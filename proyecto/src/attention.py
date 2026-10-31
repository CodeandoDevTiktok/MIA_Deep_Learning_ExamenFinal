"""
Módulos de Atención para Predicción Espacio-Temporal
=====================================================

Implementación de mecanismos de atención espacial y temporal para
mejorar la predicción de deforestación en series de imágenes satelitales.

Referencias:
    - Woo et al. (2018) "CBAM: Convolutional Block Attention Module"
    - Vaswani et al. (2017) "Attention Is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """
    Módulo de Atención Espacial.

    Genera un mapa de atención 2D que pondera la importancia de cada
    ubicación espacial en el feature map.

    Args:
        kernel_size (int): Tamaño del kernel para la convolución. Default: 7
    """

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "Kernel size debe ser 3 o 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del módulo de atención espacial.

        Args:
            x: Tensor de entrada [B, C, H, W]

        Returns:
            Tensor con atención aplicada [B, C, H, W]
        """
        # Calcular estadísticas a lo largo del canal
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # Concatenar y aplicar convolución
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = self.conv(concat)  # [B, 1, H, W]
        attention = self.sigmoid(attention)

        return x * attention


class ChannelAttention(nn.Module):
    """
    Módulo de Atención por Canal.

    Genera pesos de atención para cada canal del feature map,
    capturando dependencias inter-canal.

    Args:
        in_channels (int): Número de canales de entrada
        reduction_ratio (int): Ratio de reducción para el bottleneck. Default: 16
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP compartido
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del módulo de atención por canal.

        Args:
            x: Tensor de entrada [B, C, H, W]

        Returns:
            Tensor con atención aplicada [B, C, H, W]
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Combina atención por canal y espacial de forma secuencial
    para refinar feature maps.

    Args:
        in_channels (int): Número de canales de entrada
        reduction_ratio (int): Ratio de reducción para atención por canal
        kernel_size (int): Tamaño del kernel para atención espacial

    Reference:
        Woo et al. "CBAM: Convolutional Block Attention Module" ECCV 2018
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7
    ):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de CBAM.

        Args:
            x: Tensor de entrada [B, C, H, W]

        Returns:
            Tensor refinado con atención [B, C, H, W]
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class TemporalAttention(nn.Module):
    """
    Módulo de Atención Temporal.

    Aplica atención sobre la dimensión temporal para ponderar
    la importancia de cada timestep en la secuencia.

    Args:
        hidden_dim (int): Dimensión de los features ocultos
        num_heads (int): Número de cabezas de atención. Default: 8
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super(TemporalAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim debe ser divisible por num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass de atención temporal multi-cabeza.

        Args:
            x: Tensor de entrada [B, T, D] donde T es tiempo
            mask: Máscara opcional para atención

        Returns:
            tuple: (output, attention_weights)
        """
        B, T, D = x.shape

        # Proyecciones lineales
        Q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Calcular scores de atención
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Aplicar atención a valores
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, D)

        output = self.output(context)

        return output, attention_weights


class SpatioTemporalAttention(nn.Module):
    """
    Módulo de Atención Espacio-Temporal.

    Combina atención espacial (CBAM) con atención temporal para
    procesar secuencias de imágenes satelitales.

    Args:
        in_channels (int): Canales de entrada
        hidden_dim (int): Dimensión oculta para atención temporal
        num_heads (int): Número de cabezas de atención
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_heads: int = 8
    ):
        super(SpatioTemporalAttention, self).__init__()

        self.spatial_attention = CBAM(in_channels)

        # Proyección para atención temporal
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.temporal_proj = nn.Linear(in_channels, hidden_dim)
        self.temporal_attention = TemporalAttention(hidden_dim, num_heads)
        self.output_proj = nn.Linear(hidden_dim, in_channels)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass de atención espacio-temporal.

        Args:
            x: Tensor de entrada [B, T, C, H, W]

        Returns:
            tuple: (output, spatial_features, temporal_weights)
        """
        B, T, C, H, W = x.shape

        # Aplicar atención espacial a cada frame
        spatial_out = []
        for t in range(T):
            frame = x[:, t]  # [B, C, H, W]
            attended = self.spatial_attention(frame)
            spatial_out.append(attended)

        spatial_out = torch.stack(spatial_out, dim=1)  # [B, T, C, H, W]

        # Preparar para atención temporal
        pooled = self.pool(spatial_out.view(B * T, C, H, W))  # [B*T, C, 1, 1]
        pooled = pooled.view(B, T, C)  # [B, T, C]

        temporal_features = self.temporal_proj(pooled)  # [B, T, hidden_dim]
        temporal_out, temporal_weights = self.temporal_attention(temporal_features)

        # Proyectar de vuelta y modular features espaciales
        temporal_modulation = self.output_proj(temporal_out)  # [B, T, C]
        temporal_modulation = temporal_modulation.unsqueeze(-1).unsqueeze(-1)  # [B, T, C, 1, 1]

        output = spatial_out * torch.sigmoid(temporal_modulation)

        return output, spatial_out, temporal_weights


class PositionalEncoding(nn.Module):
    """
    Codificación Posicional para secuencias temporales.

    Añade información posicional a los embeddings para que el modelo
    pueda distinguir entre diferentes timesteps.

    Args:
        d_model (int): Dimensión del modelo
        max_len (int): Longitud máxima de secuencia
        dropout (float): Tasa de dropout
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Crear encodings posicionales
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Añadir encoding posicional.

        Args:
            x: Tensor de entrada [B, T, D]

        Returns:
            Tensor con encoding posicional añadido
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
