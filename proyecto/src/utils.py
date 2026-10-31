"""
Utilidades para el Proyecto de Predicción de Deforestación
============================================================

Funciones auxiliares para:
- Configuración de semillas
- Checkpointing
- Early stopping
- Visualización
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def set_seed(seed: int = 42):
    """
    Fija semillas para reproducibilidad.

    Args:
        seed: Valor de la semilla
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    filepath: str,
    **kwargs
):
    """
    Guarda un checkpoint del modelo.

    Args:
        model: Modelo a guardar
        optimizer: Optimizador
        epoch: Época actual
        metrics: Métricas de evaluación
        filepath: Ruta del archivo
        **kwargs: Información adicional
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda'
) -> Dict:
    """
    Carga un checkpoint.

    Args:
        filepath: Ruta del archivo
        model: Modelo a cargar
        optimizer: Optimizador (opcional)
        device: Dispositivo

    Returns:
        Diccionario con información del checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


class EarlyStopping:
    """
    Early stopping para prevenir overfitting.

    Args:
        patience: Épocas sin mejora antes de parar
        min_delta: Cambio mínimo para considerar mejora
        mode: 'min' para pérdida, 'max' para métricas
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        """
        Verifica si se debe detener el entrenamiento.

        Args:
            value: Valor actual de la métrica

        Returns:
            True si se debe detener
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'min':
            is_improvement = value < self.best_value - self.min_delta
        else:
            is_improvement = value > self.best_value + self.min_delta

        if is_improvement:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class AverageMeter:
    """
    Calcula y almacena promedios y valores actuales.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Cuenta parámetros del modelo.

    Args:
        model: Modelo de PyTorch

    Returns:
        Diccionario con conteos
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def visualize_prediction(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    save_path: Optional[str] = None,
    title: str = ""
):
    """
    Visualiza imagen, ground truth y predicción.

    Args:
        image: Imagen de entrada [C, H, W] o [H, W, C]
        target: Ground truth [H, W]
        prediction: Predicción [H, W]
        save_path: Ruta para guardar
        title: Título de la figura
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Imagen (RGB si tiene suficientes canales)
    if image.shape[0] <= 4:  # [C, H, W]
        img_show = np.transpose(image[:3], (1, 2, 0))
    else:
        img_show = image[:, :, :3]

    img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)
    axes[0].imshow(img_show)
    axes[0].set_title('Imagen Satelital')
    axes[0].axis('off')

    # Ground truth
    cmap = mcolors.ListedColormap(['green', 'red'])
    axes[1].imshow(target, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Predicción
    axes[2].imshow(prediction, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title('Predicción')
    axes[2].axis('off')

    # Diferencia
    diff = np.zeros((*target.shape, 3))
    diff[..., 1] = (target == 0) & (prediction == 0)  # TN: verde
    diff[..., 0] = (prediction == 1) & (target == 0)  # FP: rojo
    diff[..., 2] = (prediction == 0) & (target == 1)  # FN: azul
    diff[(target == 1) & (prediction == 1)] = [1, 1, 0]  # TP: amarillo

    axes[3].imshow(diff)
    axes[3].set_title('Comparación (TP:amarillo, TN:verde, FP:rojo, FN:azul)')
    axes[3].axis('off')

    if title:
        plt.suptitle(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_temporal_sequence(
    sequence: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualiza una secuencia temporal de imágenes.

    Args:
        sequence: Secuencia [T, C, H, W]
        predictions: Predicciones opcionales [T, H, W]
        save_path: Ruta para guardar
    """
    T = sequence.shape[0]
    n_rows = 2 if predictions is not None else 1

    fig, axes = plt.subplots(n_rows, T, figsize=(4 * T, 4 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for t in range(T):
        # Imagen
        img = np.transpose(sequence[t, :3], (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[0, t].imshow(img)
        axes[0, t].set_title(f'Año t+{t}')
        axes[0, t].axis('off')

        # Predicción
        if predictions is not None:
            cmap = mcolors.ListedColormap(['green', 'red'])
            axes[1, t].imshow(predictions[t], cmap=cmap, vmin=0, vmax=1)
            axes[1, t].set_title(f'Predicción t+{t}')
            axes[1, t].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_device() -> torch.device:
    """
    Obtiene el dispositivo disponible.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Usando CPU")

    return device


def print_model_summary(model: nn.Module, input_size: tuple):
    """
    Imprime un resumen del modelo.

    Args:
        model: Modelo de PyTorch
        input_size: Tamaño de entrada (sin batch)
    """
    params = count_parameters(model)

    print("=" * 60)
    print(f"Modelo: {model.__class__.__name__}")
    print("=" * 60)
    print(f"Entrada esperada: {input_size}")
    print(f"Parámetros totales: {params['total']:,}")
    print(f"Parámetros entrenables: {params['trainable']:,}")
    print(f"Parámetros no entrenables: {params['non_trainable']:,}")
    print(f"Tamaño aproximado: {params['total'] * 4 / 1e6:.2f} MB (float32)")
    print("=" * 60)
