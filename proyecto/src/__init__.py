"""
Predicción Espacio-Temporal de Deforestación en la Amazonía Peruana
====================================================================

Este paquete implementa modelos de aprendizaje profundo para predecir
la deforestación usando series temporales de imágenes satelitales.

Módulos:
    - dataset: Carga y preprocesamiento de datos satelitales
    - models: Arquitecturas de redes neuronales (U-Net, ConvLSTM, ST-AttentionNet)
    - attention: Módulos de atención espacial y temporal
    - train: Pipeline de entrenamiento
    - evaluate: Métricas y evaluación
    - interpretability: Grad-CAM y visualizaciones explicativas
    - utils: Funciones auxiliares
"""

__version__ = "1.0.0"
__author__ = "Equipo MIA-203"

from .models import UNet, ConvLSTM, STAttentionNet
from .dataset import DeforestationDataset
from .attention import SpatialAttention, TemporalAttention, CBAM
