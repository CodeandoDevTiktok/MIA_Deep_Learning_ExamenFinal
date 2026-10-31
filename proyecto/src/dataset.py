"""
Dataset para Predicción de Deforestación
==========================================

Módulo para carga y preprocesamiento de datos satelitales
de GeoBosques y MapBiomas para la Amazonía peruana.

Formato de datos:
    - Imágenes: GeoTIFF con múltiples bandas espectrales
    - Máscaras: GeoTIFF binario (0: no-deforestación, 1: deforestación)
    - Series temporales: Secuencias de años consecutivos
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Callable
import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import random


class DeforestationDataset(Dataset):
    """
    Dataset para series temporales de imágenes satelitales.

    Args:
        root_dir (str): Directorio raíz con los datos
        regions (list): Lista de regiones a incluir
        years (list): Lista de años a incluir
        sequence_length (int): Longitud de la secuencia temporal
        patch_size (int): Tamaño del patch a extraer
        transform: Transformaciones de aumento de datos
        mode (str): 'train', 'val' o 'test'
        normalize (bool): Normalizar imágenes

    Estructura esperada del directorio:
        root_dir/
        ├── images/
        │   ├── region1/
        │   │   ├── 2015.tif
        │   │   ├── 2016.tif
        │   │   └── ...
        │   └── region2/
        │       └── ...
        └── masks/
            ├── region1/
            │   ├── 2015.tif
            │   └── ...
            └── region2/
                └── ...
    """

    # Estadísticas de normalización (calculadas sobre el dataset de entrenamiento)
    BAND_MEANS = [0.1021, 0.0987, 0.0924, 0.2145, 0.2567, 0.1834]
    BAND_STDS = [0.0523, 0.0498, 0.0612, 0.0823, 0.0945, 0.0756]

    def __init__(
        self,
        root_dir: str,
        regions: List[str] = None,
        years: List[int] = None,
        sequence_length: int = 5,
        patch_size: int = 256,
        transform: Optional[Callable] = None,
        mode: str = 'train',
        normalize: bool = True,
        stride: int = 128
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.transform = transform
        self.mode = mode
        self.normalize = normalize
        self.stride = stride

        # Descubrir regiones y años disponibles
        self.images_dir = self.root_dir / 'images'
        self.masks_dir = self.root_dir / 'masks'

        if regions is None:
            regions = [d.name for d in self.images_dir.iterdir() if d.is_dir()]
        self.regions = regions

        if years is None:
            years = list(range(2015, 2024))
        self.years = sorted(years)

        # Construir índice de samples
        self.samples = self._build_sample_index()

        # Configurar transformaciones por defecto
        if self.transform is None:
            self.transform = self._get_default_transform()

    def _build_sample_index(self) -> List[Dict]:
        """
        Construye índice de samples (patches de secuencias temporales).
        """
        samples = []

        for region in self.regions:
            region_images_dir = self.images_dir / region
            region_masks_dir = self.masks_dir / region

            if not region_images_dir.exists():
                continue

            # Obtener años disponibles para esta región
            available_years = []
            for year in self.years:
                img_path = region_images_dir / f"{year}.tif"
                mask_path = region_masks_dir / f"{year}.tif"
                if img_path.exists() and mask_path.exists():
                    available_years.append(year)

            # Crear secuencias de años consecutivos
            for i in range(len(available_years) - self.sequence_length + 1):
                seq_years = available_years[i:i + self.sequence_length]

                # Obtener dimensiones de la imagen
                with rasterio.open(region_images_dir / f"{seq_years[0]}.tif") as src:
                    height, width = src.height, src.width

                # Generar patches
                for row in range(0, height - self.patch_size + 1, self.stride):
                    for col in range(0, width - self.patch_size + 1, self.stride):
                        samples.append({
                            'region': region,
                            'years': seq_years,
                            'row': row,
                            'col': col
                        })

        return samples

    def _get_default_transform(self) -> A.Compose:
        """
        Retorna transformaciones por defecto según el modo.
        """
        if self.mode == 'train':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
            ])
        else:
            return A.Compose([])  # Sin augmentación para val/test

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna una secuencia de imágenes y la máscara objetivo.

        Returns:
            images: Tensor [T, C, H, W]
            mask: Tensor [H, W] (máscara del último año para predicción)
        """
        sample = self.samples[idx]
        region = sample['region']
        years = sample['years']
        row, col = sample['row'], sample['col']

        # Ventana para extraer el patch
        window = Window(col, row, self.patch_size, self.patch_size)

        # Cargar secuencia de imágenes
        images = []
        for year in years:
            img_path = self.images_dir / region / f"{year}.tif"
            with rasterio.open(img_path) as src:
                img = src.read(window=window).astype(np.float32)  # [C, H, W]
                images.append(img)

        images = np.stack(images, axis=0)  # [T, C, H, W]

        # Cargar máscara del último año (objetivo de predicción)
        target_year = years[-1]
        mask_path = self.masks_dir / region / f"{target_year}.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1, window=window).astype(np.int64)  # [H, W]

        # Aplicar transformaciones (mismas para toda la secuencia)
        if self.transform:
            # Transformar el primer frame y aplicar la misma transformación a todos
            transformed = self.transform(
                image=images[0].transpose(1, 2, 0),  # [H, W, C]
                mask=mask
            )

            # Obtener parámetros de la transformación y aplicar a toda la secuencia
            replay = transformed.get('replay', None)

            if replay:
                transformed_images = []
                for t in range(len(images)):
                    t_result = A.ReplayCompose.replay(
                        replay,
                        image=images[t].transpose(1, 2, 0)
                    )
                    transformed_images.append(t_result['image'].transpose(2, 0, 1))
                images = np.stack(transformed_images, axis=0)
            else:
                # Transformación simple sin replay
                images_transformed = []
                seed = random.randint(0, 2**32)
                for t in range(len(images)):
                    random.seed(seed)
                    np.random.seed(seed)
                    result = self.transform(
                        image=images[t].transpose(1, 2, 0),
                        mask=mask if t == len(images) - 1 else np.zeros_like(mask)
                    )
                    images_transformed.append(result['image'].transpose(2, 0, 1))
                    if t == len(images) - 1:
                        mask = result['mask']
                images = np.stack(images_transformed, axis=0)

        # Normalizar
        if self.normalize:
            for c in range(images.shape[1]):
                images[:, c] = (images[:, c] - self.BAND_MEANS[c]) / self.BAND_STDS[c]

        # Convertir a tensores
        images = torch.from_numpy(images).float()
        mask = torch.from_numpy(mask).long()

        return images, mask


class SyntheticDeforestationDataset(Dataset):
    """
    Dataset sintético para pruebas y demostraciones.

    Genera datos artificiales que simulan patrones de deforestación
    para validar el pipeline de entrenamiento.

    Args:
        num_samples (int): Número de muestras a generar
        sequence_length (int): Longitud de la secuencia temporal
        image_size (int): Tamaño de las imágenes
        num_channels (int): Número de canales (bandas)
        num_classes (int): Número de clases de salida
    """

    def __init__(
        self,
        num_samples: int = 1000,
        sequence_length: int = 5,
        image_size: int = 128,
        num_channels: int = 6,
        num_classes: int = 2
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Genera una muestra sintética."""
        # Semilla para reproducibilidad por sample
        np.random.seed(idx)

        # Generar secuencia de imágenes con patrón temporal
        images = []
        base_pattern = np.random.rand(self.image_size, self.image_size)

        for t in range(self.sequence_length):
            # Simular cambio temporal (deforestación progresiva)
            temporal_factor = 1 + 0.1 * t
            frame = np.zeros((self.num_channels, self.image_size, self.image_size))

            for c in range(self.num_channels):
                noise = np.random.randn(self.image_size, self.image_size) * 0.1
                frame[c] = base_pattern * temporal_factor + noise + c * 0.1

            images.append(frame)

        images = np.stack(images, axis=0).astype(np.float32)

        # Generar máscara de deforestación
        # Simular áreas de deforestación como regiones circulares
        mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)

        num_deforestation_areas = np.random.randint(1, 5)
        for _ in range(num_deforestation_areas):
            cx = np.random.randint(20, self.image_size - 20)
            cy = np.random.randint(20, self.image_size - 20)
            radius = np.random.randint(5, 25)

            y, x = np.ogrid[:self.image_size, :self.image_size]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask[dist <= radius] = 1

        images = torch.from_numpy(images)
        mask = torch.from_numpy(mask)

        return images, mask


def create_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    sequence_length: int = 5,
    patch_size: int = 256,
    num_workers: int = 4,
    train_regions: List[str] = None,
    val_regions: List[str] = None,
    test_regions: List[str] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para entrenamiento, validación y prueba.

    Args:
        root_dir: Directorio raíz de datos
        batch_size: Tamaño del batch
        sequence_length: Longitud de secuencia temporal
        patch_size: Tamaño de los patches
        num_workers: Número de workers para carga de datos
        train_regions: Regiones para entrenamiento
        val_regions: Regiones para validación
        test_regions: Regiones para prueba

    Returns:
        Tuple de (train_loader, val_loader, test_loader)
    """
    train_dataset = DeforestationDataset(
        root_dir=root_dir,
        regions=train_regions,
        sequence_length=sequence_length,
        patch_size=patch_size,
        mode='train',
        **kwargs
    )

    val_dataset = DeforestationDataset(
        root_dir=root_dir,
        regions=val_regions,
        sequence_length=sequence_length,
        patch_size=patch_size,
        mode='val',
        **kwargs
    )

    test_dataset = DeforestationDataset(
        root_dir=root_dir,
        regions=test_regions,
        sequence_length=sequence_length,
        patch_size=patch_size,
        mode='test',
        **kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def create_synthetic_dataloaders(
    num_train: int = 800,
    num_val: int = 100,
    num_test: int = 100,
    batch_size: int = 8,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders con datos sintéticos para pruebas.
    """
    train_dataset = SyntheticDeforestationDataset(num_samples=num_train, **kwargs)
    val_dataset = SyntheticDeforestationDataset(num_samples=num_val, **kwargs)
    test_dataset = SyntheticDeforestationDataset(num_samples=num_test, **kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
