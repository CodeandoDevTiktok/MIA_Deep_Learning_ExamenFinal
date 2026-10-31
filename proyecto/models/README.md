# Pesos de Modelos Entrenados

## Modelos Disponibles

| Modelo | IoU | F1 | Enlace |
|--------|-----|----|----|
| ST-AttentionNet (mejor) | 0.781 | 0.877 | [Descargar](#) |
| U-Net | 0.701 | 0.824 | [Descargar](#) |
| ConvLSTM | 0.723 | 0.839 | [Descargar](#) |

## Instrucciones de Descarga

### Opción 1: Google Drive

```bash
# Descargar modelo ST-AttentionNet
gdown https://drive.google.com/uc?id=<FILE_ID> -O st_attention_best.pth

# Descargar modelo U-Net
gdown https://drive.google.com/uc?id=<FILE_ID> -O unet_best.pth

# Descargar modelo ConvLSTM
gdown https://drive.google.com/uc?id=<FILE_ID> -O convlstm_best.pth
```

### Opción 2: Hugging Face Hub

```python
from huggingface_hub import hf_hub_download

# Descargar modelo
model_path = hf_hub_download(
    repo_id="usuario/deforestation-prediction-peru",
    filename="st_attention_best.pth"
)
```

## Uso

```python
import torch
from src.models import STAttentionNet
from src.utils import load_checkpoint

# Crear modelo
model = STAttentionNet(
    in_channels=6,
    out_channels=2,
    encoder_features=[64, 128, 256, 512],
    hidden_dim=256,
    num_heads=8
)

# Cargar pesos
load_checkpoint('models/st_attention_best.pth', model)
model.eval()

# Inferencia
with torch.no_grad():
    input_tensor = torch.randn(1, 5, 6, 256, 256)  # [B, T, C, H, W]
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
```

## Información del Checkpoint

Cada archivo `.pth` contiene:

```python
{
    'epoch': int,                    # Época del mejor modelo
    'model_state_dict': dict,        # Pesos del modelo
    'optimizer_state_dict': dict,    # Estado del optimizador
    'metrics': {
        'iou': float,
        'f1': float,
        'precision': float,
        'recall': float,
        'loss': float
    }
}
```

## Configuración de Entrenamiento

Los modelos fueron entrenados con:
- **Hardware**: NVIDIA T4 (Google Colab)
- **Tiempo**: ~4 horas por modelo
- **Épocas**: 100 (early stopping)
- **Batch size**: 8
- **Learning rate**: 1e-4
- **Optimizador**: AdamW
