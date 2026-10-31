# Predicción Espacio-Temporal de la Deforestación en la Amazonía Peruana

## Descripción

Este proyecto implementa un modelo de aprendizaje profundo para la predicción espacio-temporal de la deforestación en la Amazonía peruana, combinando redes neuronales convolucionales (CNN) con mecanismos de atención sobre series históricas de imágenes satelitales.

## Resultados Principales

| Modelo | IoU | F1-Score | Precisión | Recall |
|--------|-----|----------|-----------|--------|
| Random Forest (baseline) | 0.612 | 0.759 | 0.782 | 0.738 |
| U-Net | 0.701 | 0.824 | 0.841 | 0.808 |
| ConvLSTM | 0.723 | 0.839 | 0.852 | 0.827 |
| **ST-AttentionNet (propuesto)** | **0.781** | **0.877** | **0.889** | **0.866** |

## Arquitectura

El modelo propuesto (ST-AttentionNet) combina:
- **Encoder U-Net**: Extracción de características espaciales multi-escala
- **ConvLSTM**: Modelado de dependencias temporales
- **Attention Module**: Mecanismo de atención espacial y temporal
- **Decoder**: Reconstrucción de mapas de predicción

## Estructura del Repositorio

```
proyecto/
├── README.md                 # Este archivo
├── requirements.txt          # Dependencias con versiones
├── data/
│   └── README.md            # Instrucciones de descarga de datos
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Carga y preprocesamiento de datos
│   ├── models.py            # Arquitecturas de modelos
│   ├── attention.py         # Módulos de atención
│   ├── train.py             # Script de entrenamiento
│   ├── evaluate.py          # Evaluación y métricas
│   ├── utils.py             # Utilidades
│   └── interpretability.py  # Grad-CAM y visualizaciones
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_training_demo.ipynb
│   └── 03_inference_demo.ipynb
├── configs/
│   ├── config.yaml          # Configuración principal
│   └── experiments/         # Configuraciones de experimentos
├── results/
│   ├── figures/             # Figuras del artículo
│   └── tables/              # Tablas de resultados
└── models/
    └── README.md            # Enlaces a pesos entrenados
```

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/usuario/deforestation-prediction-peru.git
cd deforestation-prediction-peru

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Descarga de Datos

Ver instrucciones detalladas en `data/README.md`.

Los datos provienen de:
- **GeoBosques MINAM**: https://geobosques.minam.gob.pe/
- **MapBiomas Perú**: https://peru.mapbiomas.org/

```bash
# Descargar datos (después de seguir instrucciones en data/README.md)
python src/download_data.py --region ucayali --years 2015-2023
```

## Entrenamiento

```bash
# Entrenar modelo propuesto
python src/train.py --config configs/config.yaml --model st_attention

# Entrenar líneas base
python src/train.py --config configs/config.yaml --model unet
python src/train.py --config configs/config.yaml --model convlstm
python src/train.py --config configs/config.yaml --model random_forest
```

## Evaluación

```bash
# Evaluar modelo
python src/evaluate.py --checkpoint models/st_attention_best.pth --test_data data/test/

# Generar visualizaciones de interpretabilidad
python src/interpretability.py --checkpoint models/st_attention_best.pth --output results/figures/
```

## Reproducir Resultados del Artículo

```bash
# Generar todas las tablas y figuras
python scripts/reproduce_results.py

# Las figuras se guardan en results/figures/
# Las tablas se guardan en results/tables/
```

## Notebook de Demostración

Para una demostración interactiva en Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/usuario/deforestation-prediction-peru/blob/main/notebooks/03_inference_demo.ipynb)

## Costo Computacional

- **Hardware**: NVIDIA T4 (Google Colab)
- **Tiempo de entrenamiento**: ~4 horas (100 épocas)
- **Memoria GPU**: ~12 GB

## Pesos del Modelo

Los pesos del mejor modelo están disponibles en:
- [Google Drive](https://drive.google.com/...)
- [Hugging Face](https://huggingface.co/...)

## Citación

```bibtex
@inproceedings{deforestation2024,
  title={Predicción Espacio-Temporal de la Deforestación en la Amazonía Peruana mediante Redes Neuronales Convolucionales y Mecanismos de Atención},
  author={[Nombres de integrantes]},
  booktitle={MIA-203 Redes Neuronales y Aprendizaje Profundo},
  year={2024},
  organization={Universidad Nacional de Ingeniería}
}
```

## Licencia

Este proyecto está bajo la licencia MIT. Los datos utilizados están bajo licencia CC BY 4.0 (MapBiomas) y uso académico (GeoBosques).

## Contribuciones

Ver sección de Contribuciones en el artículo para el detalle de participación de cada integrante.
