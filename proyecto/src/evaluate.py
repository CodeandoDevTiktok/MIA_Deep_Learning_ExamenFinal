"""
Evaluación y Métricas para Segmentación de Deforestación
==========================================================

Implementa métricas estándar de segmentación semántica:
- IoU (Intersection over Union)
- F1-Score (Dice Coefficient)
- Precisión y Recall
- Matriz de confusión
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm


class SegmentationMetrics:
    """
    Calcula métricas de segmentación de manera incremental.

    Args:
        num_classes (int): Número de clases
        ignore_index (int): Índice a ignorar en el cálculo
    """

    def __init__(self, num_classes: int = 2, ignore_index: int = -100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reinicia los contadores."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Actualiza métricas con un batch de predicciones.

        Args:
            preds: Predicciones [B, H, W]
            targets: Ground truth [B, H, W]
        """
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Filtrar índices ignorados
        valid_mask = targets != self.ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]

        # Actualizar matriz de confusión
        cm = confusion_matrix(targets, preds, labels=list(range(self.num_classes)))
        self.confusion_matrix += cm

    def compute(self) -> Dict[str, float]:
        """
        Calcula todas las métricas.

        Returns:
            Diccionario con métricas calculadas
        """
        # Evitar división por cero
        cm = self.confusion_matrix.astype(np.float64)
        eps = 1e-8

        # IoU por clase
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou_per_class = intersection / (union + eps)

        # Precisión y Recall por clase
        precision_per_class = intersection / (cm.sum(axis=0) + eps)
        recall_per_class = intersection / (cm.sum(axis=1) + eps)

        # F1-Score por clase
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + eps)

        # Métricas globales
        accuracy = np.diag(cm).sum() / (cm.sum() + eps)
        mean_iou = np.nanmean(iou_per_class)
        mean_f1 = np.nanmean(f1_per_class)

        # Métricas específicas para clase de deforestación (clase 1)
        deforestation_iou = iou_per_class[1] if self.num_classes > 1 else iou_per_class[0]
        deforestation_f1 = f1_per_class[1] if self.num_classes > 1 else f1_per_class[0]
        deforestation_precision = precision_per_class[1] if self.num_classes > 1 else precision_per_class[0]
        deforestation_recall = recall_per_class[1] if self.num_classes > 1 else recall_per_class[0]

        return {
            'accuracy': accuracy,
            'iou': mean_iou,
            'f1': mean_f1,
            'iou_per_class': iou_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'deforestation_iou': deforestation_iou,
            'deforestation_f1': deforestation_f1,
            'deforestation_precision': deforestation_precision,
            'deforestation_recall': deforestation_recall,
            'confusion_matrix': cm.tolist()
        }

    def get_iou(self) -> float:
        """Retorna IoU medio actual."""
        metrics = self.compute()
        return metrics['iou']

    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        class_names: List[str] = ['No Deforestación', 'Deforestación']
    ):
        """
        Grafica la matriz de confusión.

        Args:
            save_path: Ruta para guardar la figura
            class_names: Nombres de las clases
        """
        cm = self.confusion_matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Matriz de confusión absoluta
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0]
        )
        axes[0].set_title('Matriz de Confusión (Absoluta)')
        axes[0].set_xlabel('Predicción')
        axes[0].set_ylabel('Valor Real')

        # Matriz de confusión normalizada
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1]
        )
        axes[1].set_title('Matriz de Confusión (Normalizada)')
        axes[1].set_xlabel('Predicción')
        axes[1].set_ylabel('Valor Real')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Función de conveniencia para calcular métricas en un solo batch.

    Args:
        preds: Predicciones [B, H, W] o [B, C, H, W]
        targets: Ground truth [B, H, W]
        num_classes: Número de clases

    Returns:
        Diccionario con métricas
    """
    if preds.dim() == 4:  # [B, C, H, W]
        preds = torch.argmax(preds, dim=1)

    metrics = SegmentationMetrics(num_classes=num_classes)
    metrics.update(preds, targets)
    return metrics.compute()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    num_classes: int = 2
) -> Dict:
    """
    Evalúa un modelo en un dataloader completo.

    Args:
        model: Modelo a evaluar
        dataloader: DataLoader de evaluación
        device: Dispositivo de cómputo
        num_classes: Número de clases

    Returns:
        Diccionario con métricas y predicciones
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    metrics = SegmentationMetrics(num_classes=num_classes)
    all_preds = []
    all_targets = []

    for images, masks in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        if hasattr(model, 'in_channels') and images.dim() == 5:
            # Modelo U-Net con secuencia temporal: usar último frame
            outputs = model(images[:, -1])
        else:
            outputs = model(images)

        preds = torch.argmax(outputs, dim=1)
        metrics.update(preds, masks)

        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    results = metrics.compute()
    results['predictions'] = torch.cat(all_preds, dim=0)
    results['targets'] = torch.cat(all_targets, dim=0)

    return results


def plot_results_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Compara resultados de múltiples modelos.

    Args:
        results: Diccionario {nombre_modelo: métricas}
        save_path: Ruta para guardar la figura
    """
    models = list(results.keys())
    metrics_to_plot = ['iou', 'f1', 'deforestation_precision', 'deforestation_recall']
    metric_labels = ['IoU', 'F1-Score', 'Precisión', 'Recall']

    x = np.arange(len(metrics_to_plot))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        values = [results[model].get(m, 0) for m in metrics_to_plot]
        offset = (i - len(models) / 2) * width
        ax.bar(x + offset, values, width, label=model)

    ax.set_ylabel('Score')
    ax.set_title('Comparación de Métricas por Modelo')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_evaluation_report(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    model_name: str,
    output_dir: str,
    device: str = 'cuda'
) -> Dict:
    """
    Genera un reporte completo de evaluación.

    Args:
        model: Modelo a evaluar
        test_loader: DataLoader de prueba
        model_name: Nombre del modelo
        output_dir: Directorio de salida
        device: Dispositivo de cómputo

    Returns:
        Diccionario con resultados
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluar modelo
    results = evaluate_model(model, test_loader, device)

    # Guardar métricas
    metrics_file = output_dir / f'{model_name}_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"Reporte de Evaluación: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"IoU (mean): {results['iou']:.4f}\n")
        f.write(f"F1-Score (mean): {results['f1']:.4f}\n\n")
        f.write("Métricas por Clase:\n")
        f.write("-" * 30 + "\n")
        f.write(f"  Clase 0 (No Deforestación):\n")
        f.write(f"    IoU: {results['iou_per_class'][0]:.4f}\n")
        f.write(f"    F1: {results['f1_per_class'][0]:.4f}\n")
        f.write(f"    Precisión: {results['precision_per_class'][0]:.4f}\n")
        f.write(f"    Recall: {results['recall_per_class'][0]:.4f}\n\n")
        f.write(f"  Clase 1 (Deforestación):\n")
        f.write(f"    IoU: {results['deforestation_iou']:.4f}\n")
        f.write(f"    F1: {results['deforestation_f1']:.4f}\n")
        f.write(f"    Precisión: {results['deforestation_precision']:.4f}\n")
        f.write(f"    Recall: {results['deforestation_recall']:.4f}\n")

    # Generar matriz de confusión
    metrics_tracker = SegmentationMetrics(num_classes=2)
    metrics_tracker.confusion_matrix = np.array(results['confusion_matrix'])
    metrics_tracker.plot_confusion_matrix(
        save_path=output_dir / f'{model_name}_confusion_matrix.png'
    )

    print(f"Reporte guardado en: {output_dir}")
    return results
