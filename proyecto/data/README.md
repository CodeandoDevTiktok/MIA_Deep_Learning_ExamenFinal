# Datos para Predicción de Deforestación

## Fuentes de Datos

### 1. GeoBosques - MINAM
**URL**: https://geobosques.minam.gob.pe/geobosque/view/servicios.php

Proporciona datos oficiales de pérdida de bosque en la Amazonía peruana:
- **Cobertura temporal**: 2001-2025
- **Fuente satelital**: Landsat 7, 8, 9
- **Formato**: Raster (GeoTIFF)
- **Tipos de datos**:
  - Línea base de no-bosque (2000)
  - Pérdida de bosque anual
  - Bosque remanente
  - Alertas tempranas

### 2. MapBiomas Perú
**URL**: https://peru.mapbiomas.org/descargas/mapas-de-las-colecciones/

Mapas de cobertura y uso del suelo:
- **Cobertura temporal**: 1985-2025 (Colección 4)
- **Formato**: GeoTIFF
- **Licencia**: CC BY 4.0
- **Resolución**: 30m

## Descarga de Datos

### Opción 1: Descarga Manual

1. **GeoBosques**:
   - Visitar: https://geobosques.minam.gob.pe/geobosque/view/descargas.php
   - Seleccionar región de interés (ej: Ucayali, Loreto, Madre de Dios)
   - Descargar mapas de pérdida de bosque por año

2. **MapBiomas**:
   - Visitar: https://peru.mapbiomas.org/descargas/mapas-de-las-colecciones/
   - Descargar colección 4 de cobertura y uso
   - URL directa para GeoTIFF: `https://storage.googleapis.com/mapbiomas-public/peru/collection-4/lclu/peru-{year}.tif`

### Opción 2: Google Earth Engine

```python
import ee
ee.Initialize()

# Definir región de interés (Amazonía peruana)
roi = ee.Geometry.Rectangle([-77.5, -13.5, -68.5, -0.5])

# Cargar datos de MapBiomas Perú
mapbiomas = ee.Image('projects/mapbiomas-workspace/public/collection4/mapbiomas_collection40_integration_v1')

# Exportar a Google Drive
task = ee.batch.Export.image.toDrive(
    image=mapbiomas.clip(roi),
    description='mapbiomas_peru_amazonia',
    scale=30,
    region=roi,
    fileFormat='GeoTIFF'
)
task.start()
```

## Estructura de Datos Esperada

```
data/
├── raw/                          # Datos descargados sin procesar
│   ├── geobosques/
│   │   ├── perdida_2015.tif
│   │   ├── perdida_2016.tif
│   │   └── ...
│   └── mapbiomas/
│       ├── peru-2015.tif
│       └── ...
├── processed/                    # Datos procesados para entrenamiento
│   ├── images/
│   │   ├── ucayali/
│   │   │   ├── 2015.tif         # Imagen multibanda (6 bandas)
│   │   │   ├── 2016.tif
│   │   │   └── ...
│   │   └── loreto/
│   │       └── ...
│   └── masks/
│       ├── ucayali/
│       │   ├── 2015.tif         # Máscara binaria de deforestación
│       │   └── ...
│       └── loreto/
│           └── ...
└── README.md
```

## Preprocesamiento

### Script de Preprocesamiento

```python
import rasterio
import numpy as np
from pathlib import Path

def preprocess_geobosques(input_path, output_path):
    """
    Preprocesa datos de GeoBosques a formato binario.

    Valores originales:
    - 0: No bosque (línea base)
    - 1: Bosque
    - 2-25: Año de pérdida (2001-2025)

    Salida:
    - 0: Sin deforestación
    - 1: Deforestación
    """
    with rasterio.open(input_path) as src:
        data = src.read(1)
        profile = src.profile.copy()

    # Crear máscara binaria
    # Deforestación = cualquier valor > 1 (años de pérdida)
    mask = (data > 1).astype(np.uint8)

    profile.update(dtype=rasterio.uint8, count=1)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mask, 1)

def extract_bands_from_landsat(input_paths, output_path):
    """
    Extrae y combina bandas espectrales de Landsat.

    Bandas utilizadas:
    - B2: Blue
    - B3: Green
    - B4: Red
    - B5: NIR
    - B6: SWIR1
    - B7: SWIR2
    """
    bands = []
    profile = None

    for path in input_paths:
        with rasterio.open(path) as src:
            bands.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()

    # Stack de bandas
    stacked = np.stack(bands, axis=0)

    profile.update(count=len(bands))

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(stacked)
```

## Regiones de Estudio Sugeridas

| Región | Departamento | Características |
|--------|--------------|-----------------|
| Ucayali | Ucayali | Alta tasa de deforestación, acceso a datos |
| Loreto | Loreto | Mayor extensión de bosque amazónico |
| Madre de Dios | Madre de Dios | Minería ilegal, biodiversidad alta |
| San Martín | San Martín | Agricultura de expansión |
| Huánuco | Huánuco | Transición sierra-selva |

## Consideraciones de Licencia

- **GeoBosques**: Uso académico permitido. Citar al MINAM.
- **MapBiomas**: Licencia CC BY 4.0. Citar según: "MapBiomas Perú (2024)"

## Notas Importantes

1. Los datos de Landsat tienen una resolución de 30m por píxel
2. Las nubes y sombras deben ser filtradas en el preprocesamiento
3. Se recomienda usar compositos anuales para reducir ruido
4. Los datos de 2001-2010 pueden tener menor calidad (Landsat 7 SLC-off)
