"""
Arquitecturas de Modelos para Predicción de Deforestación
==========================================================

Implementación de:
- U-Net: Segmentación semántica baseline
- ConvLSTM: Modelado espacio-temporal
- ST-AttentionNet: Modelo propuesto con atención espacio-temporal

Referencias:
    - Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    - Shi et al. (2015) "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .attention import CBAM, SpatioTemporalAttention, PositionalEncoding


# ============================================================================
# U-Net Components
# ============================================================================

class DoubleConv(nn.Module):
    """
    Bloque de doble convolución (Conv -> BN -> ReLU) x 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None
    ):
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Bloque de downsampling: MaxPool -> DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Bloque de upsampling: Upsample/ConvTranspose -> Concat -> DoubleConv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True
    ):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Ajustar tamaños si es necesario
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net para segmentación semántica de deforestación.

    Args:
        in_channels (int): Canales de entrada (bandas espectrales)
        out_channels (int): Clases de salida (2: bosque/no-bosque o más)
        features (list): Lista de features en cada nivel
        bilinear (bool): Usar upsampling bilinear vs conv transpuesta

    Input shape: [B, C, H, W]
    Output shape: [B, num_classes, H, W]
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 2,
        features: List[int] = [64, 128, 256, 512, 1024],
        bilinear: bool = True
    ):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[4] // factor)

        # Decoder
        self.up1 = Up(features[4], features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)

        # Clasificador final
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

    def get_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Retorna features intermedias del encoder para interpretabilidad."""
        features = []
        x1 = self.inc(x)
        features.append(x1)

        x2 = self.down1(x1)
        features.append(x2)

        x3 = self.down2(x2)
        features.append(x3)

        x4 = self.down3(x3)
        features.append(x4)

        x5 = self.down4(x4)
        features.append(x5)

        return features


# ============================================================================
# ConvLSTM Components
# ============================================================================

class ConvLSTMCell(nn.Module):
    """
    Celda ConvLSTM individual.

    Args:
        input_dim (int): Canales de entrada
        hidden_dim (int): Canales ocultos
        kernel_size (tuple): Tamaño del kernel de convolución
        bias (bool): Usar bias en convoluciones
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int],
        bias: bool = True
    ):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )


class ConvLSTM(nn.Module):
    """
    ConvLSTM para modelado espacio-temporal de secuencias de imágenes.

    Args:
        in_channels (int): Canales de entrada
        hidden_channels (list): Canales ocultos por capa
        kernel_size (tuple): Tamaño del kernel
        num_layers (int): Número de capas
        out_channels (int): Clases de salida
        batch_first (bool): Batch como primera dimensión
        return_all_layers (bool): Retornar salidas de todas las capas

    Input shape: [B, T, C, H, W]
    Output shape: [B, num_classes, H, W]
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: List[int] = [64, 64, 64],
        kernel_size: Tuple[int, int] = (3, 3),
        num_layers: int = 3,
        out_channels: int = 2,
        batch_first: bool = True,
        return_all_layers: bool = False
    ):
        super(ConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        # Construir capas ConvLSTM
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.in_channels if i == 0 else self.hidden_channels[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_channels[i],
                    kernel_size=self.kernel_size,
                    bias=True
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Capa de salida
        self.output_conv = nn.Conv2d(
            hidden_channels[-1],
            out_channels,
            kernel_size=1
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: List = None
    ) -> torch.Tensor:

        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        B, T, C, H, W = input_tensor.size()

        # Inicializar estados ocultos
        if hidden_state is None:
            hidden_state = self._init_hidden(B, (H, W))

        # Procesar secuencia
        layer_output_list = []
        last_state_list = []

        seq_len = T
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # Usar último estado para predicción
        last_output = layer_output_list[-1][:, -1, :, :, :]  # [B, C, H, W]
        logits = self.output_conv(last_output)

        return logits

    def _init_hidden(
        self,
        batch_size: int,
        image_size: Tuple[int, int]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


# ============================================================================
# ST-AttentionNet (Modelo Propuesto)
# ============================================================================

class STAttentionNet(nn.Module):
    """
    Spatio-Temporal Attention Network para predicción de deforestación.

    Combina:
    - U-Net encoder para extracción de features espaciales
    - ConvLSTM para modelado temporal
    - Módulos CBAM para atención espacial y de canal
    - Atención temporal multi-cabeza

    Args:
        in_channels (int): Canales de entrada (bandas espectrales)
        out_channels (int): Clases de salida
        encoder_features (list): Features del encoder U-Net
        hidden_dim (int): Dimensión oculta para ConvLSTM
        num_heads (int): Cabezas de atención temporal
        dropout (float): Tasa de dropout

    Input shape: [B, T, C, H, W]
    Output shape: [B, num_classes, H, W]
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 2,
        encoder_features: List[int] = [64, 128, 256, 512],
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(STAttentionNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        # ===== Encoder Espacial (por frame) =====
        self.inc = DoubleConv(in_channels, encoder_features[0])
        self.down1 = Down(encoder_features[0], encoder_features[1])
        self.cbam1 = CBAM(encoder_features[1])
        self.down2 = Down(encoder_features[1], encoder_features[2])
        self.cbam2 = CBAM(encoder_features[2])
        self.down3 = Down(encoder_features[2], encoder_features[3])
        self.cbam3 = CBAM(encoder_features[3])

        # ===== ConvLSTM para modelado temporal =====
        self.convlstm = ConvLSTMCell(
            input_dim=encoder_features[3],
            hidden_dim=hidden_dim,
            kernel_size=(3, 3)
        )

        # ===== Atención Temporal =====
        self.temporal_pool = nn.AdaptiveAvgPool2d(1)
        self.temporal_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=20, dropout=dropout)

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # ===== Decoder =====
        self.up1 = Up(hidden_dim + encoder_features[3], encoder_features[2], bilinear=True)
        self.up2 = Up(encoder_features[2] + encoder_features[2], encoder_features[1], bilinear=True)
        self.up3 = Up(encoder_features[1] + encoder_features[1], encoder_features[0], bilinear=True)

        # ===== Capa de Predicción =====
        self.final_conv = nn.Sequential(
            nn.Conv2d(encoder_features[0] + encoder_features[0], encoder_features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_features[0], out_channels, kernel_size=1)
        )

        # Almacenar para interpretabilidad
        self.attention_weights = None
        self.spatial_features = None

    def encode_frame(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Codificar un frame individual."""
        skip_connections = []

        x1 = self.inc(x)
        skip_connections.append(x1)

        x2 = self.down1(x1)
        x2 = self.cbam1(x2)
        skip_connections.append(x2)

        x3 = self.down2(x2)
        x3 = self.cbam2(x3)
        skip_connections.append(x3)

        x4 = self.down3(x3)
        x4 = self.cbam3(x4)

        return x4, skip_connections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo ST-AttentionNet.

        Args:
            x: Tensor de entrada [B, T, C, H, W]

        Returns:
            Predicción de segmentación [B, num_classes, H, W]
        """
        B, T, C, H, W = x.shape

        # ===== Codificar todos los frames =====
        encoded_frames = []
        all_skip_connections = []

        for t in range(T):
            frame = x[:, t]  # [B, C, H, W]
            encoded, skips = self.encode_frame(frame)
            encoded_frames.append(encoded)
            all_skip_connections.append(skips)

        # ===== Procesar secuencia con ConvLSTM =====
        _, _, h_enc, w_enc = encoded_frames[0].shape
        h_state, c_state = self.convlstm.init_hidden(B, (h_enc, w_enc))

        lstm_outputs = []
        for t in range(T):
            h_state, c_state = self.convlstm(encoded_frames[t], (h_state, c_state))
            lstm_outputs.append(h_state)

        lstm_outputs = torch.stack(lstm_outputs, dim=1)  # [B, T, hidden_dim, H', W']

        # ===== Atención Temporal =====
        # Pooling espacial para atención temporal
        pooled = self.temporal_pool(lstm_outputs.view(B * T, self.hidden_dim, h_enc, w_enc))
        pooled = pooled.view(B, T, self.hidden_dim)  # [B, T, D]

        # Proyección y encoding posicional
        temporal_features = self.temporal_proj(pooled)
        temporal_features = self.pos_encoding(temporal_features)

        # Self-attention temporal
        attn_out, attn_weights = self.temporal_attention(
            temporal_features, temporal_features, temporal_features
        )
        self.attention_weights = attn_weights.detach()

        # Residual + FFN
        temporal_features = self.temporal_norm(temporal_features + attn_out)
        ffn_out = self.temporal_ffn(temporal_features)
        temporal_features = self.ffn_norm(temporal_features + ffn_out)

        # Ponderar outputs LSTM con atención temporal
        temporal_weights = F.softmax(temporal_features.mean(dim=-1), dim=1)  # [B, T]
        temporal_weights = temporal_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1, 1]

        weighted_lstm = (lstm_outputs * temporal_weights).sum(dim=1)  # [B, hidden_dim, H', W']

        # ===== Decoder con Skip Connections del último frame =====
        last_skips = all_skip_connections[-1]  # Skip connections del frame más reciente

        # Concatenar con features del encoder
        dec_input = torch.cat([weighted_lstm, encoded_frames[-1]], dim=1)

        x = self.up1(dec_input, last_skips[2])
        x = self.up2(x, last_skips[1])
        x = self.up3(x, last_skips[0])

        # Concatenar con features originales y predecir
        x = torch.cat([x, last_skips[0]], dim=1)
        logits = self.final_conv(x)

        # Ajustar tamaño si es necesario
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)

        return logits

    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """Retorna los mapas de atención temporal para interpretabilidad."""
        return self.attention_weights


# ============================================================================
# Factory Function
# ============================================================================

def build_model(
    model_name: str,
    in_channels: int = 6,
    out_channels: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function para crear modelos.

    Args:
        model_name: Nombre del modelo ('unet', 'convlstm', 'st_attention')
        in_channels: Canales de entrada
        out_channels: Clases de salida
        **kwargs: Argumentos adicionales específicos del modelo

    Returns:
        Modelo instanciado
    """
    models = {
        'unet': UNet,
        'convlstm': ConvLSTM,
        'st_attention': STAttentionNet
    }

    if model_name.lower() not in models:
        raise ValueError(f"Modelo '{model_name}' no soportado. Opciones: {list(models.keys())}")

    return models[model_name.lower()](
        in_channels=in_channels,
        out_channels=out_channels,
        **kwargs
    )
