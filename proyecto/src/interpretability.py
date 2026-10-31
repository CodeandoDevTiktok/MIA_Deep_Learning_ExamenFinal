"""
Interpretabilidad de Modelos de Deforestación
===============================================

Implementa métodos de explicabilidad:
- Grad-CAM: Visualización de regiones importantes
- Mapas de atención temporal
- Análisis de errores
- Visualización del espacio latente
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Genera mapas de calor que destacan las regiones de la imagen
    más importantes para la predicción del modelo.

    Args:
        model: Modelo de PyTorch
        target_layer: Capa objetivo para extraer gradientes
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registrar hooks
        self._register_hooks()

    def _register_hooks(self):
        """Registra hooks para capturar gradientes y activaciones."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Genera el mapa Grad-CAM.

        Args:
            input_tensor: Imagen de entrada [1, C, H, W] o [1, T, C, H, W]
            target_class: Clase objetivo (si None, usa la predicha)

        Returns:
            Mapa de calor normalizado [H, W]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass para la clase objetivo
        self.model.zero_grad()

        # Crear máscara one-hot
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        # Backward
        output.backward(gradient=one_hot, retain_graph=True)

        # Calcular pesos (global average pooling de gradientes)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Combinación lineal ponderada
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU y normalización
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


class TemporalAttentionVisualizer:
    """
    Visualiza los pesos de atención temporal del modelo ST-AttentionNet.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def visualize(
        self,
        input_sequence: torch.Tensor,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualiza los pesos de atención temporal.

        Args:
            input_sequence: Secuencia de entrada [1, T, C, H, W]
            save_path: Ruta para guardar la visualización

        Returns:
            Pesos de atención [T, T] o [num_heads, T, T]
        """
        self.model.eval()

        # Forward pass
        _ = self.model(input_sequence)

        # Obtener pesos de atención
        attention_weights = self.model.get_attention_maps()

        if attention_weights is None:
            raise ValueError("El modelo no tiene mapas de atención disponibles")

        attention_weights = attention_weights.squeeze().cpu().numpy()

        # Visualizar
        if attention_weights.ndim == 3:  # Multi-head: [num_heads, T, T]
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()

            for head in range(min(8, attention_weights.shape[0])):
                im = axes[head].imshow(
                    attention_weights[head],
                    cmap='viridis',
                    aspect='auto'
                )
                axes[head].set_title(f'Head {head + 1}')
                axes[head].set_xlabel('Key (año)')
                axes[head].set_ylabel('Query (año)')
                plt.colorbar(im, ax=axes[head])

            plt.suptitle('Atención Temporal Multi-Cabeza')
        else:  # Single head: [T, T]
            plt.figure(figsize=(8, 6))
            plt.imshow(attention_weights, cmap='viridis', aspect='auto')
            plt.colorbar(label='Peso de atención')
            plt.xlabel('Key (timestep)')
            plt.ylabel('Query (timestep)')
            plt.title('Mapa de Atención Temporal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        return attention_weights


class ErrorAnalyzer:
    """
    Análisis sistemático de errores del modelo.

    Identifica patrones en los errores de predicción para
    entender las limitaciones del modelo.
    """

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.errors = []

    def collect_errors(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Recolecta información sobre errores de predicción.

        Args:
            predictions: Predicciones [B, H, W]
            targets: Ground truth [B, H, W]
            images: Imágenes originales (opcional)
            metadata: Metadatos adicionales (opcional)
        """
        errors = (predictions != targets)

        for b in range(predictions.shape[0]):
            error_mask = errors[b].cpu().numpy()
            if error_mask.any():
                error_info = {
                    'prediction': predictions[b].cpu().numpy(),
                    'target': targets[b].cpu().numpy(),
                    'error_mask': error_mask,
                    'false_positives': ((predictions[b] == 1) & (targets[b] == 0)).sum().item(),
                    'false_negatives': ((predictions[b] == 0) & (targets[b] == 1)).sum().item(),
                    'total_pixels': error_mask.size,
                    'error_rate': error_mask.mean()
                }
                if images is not None:
                    error_info['image'] = images[b].cpu().numpy()
                if metadata is not None:
                    error_info['metadata'] = metadata

                self.errors.append(error_info)

    def analyze(self) -> Dict:
        """
        Analiza los errores recolectados.

        Returns:
            Diccionario con análisis de errores
        """
        if not self.errors:
            return {'message': 'No hay errores recolectados'}

        total_fp = sum(e['false_positives'] for e in self.errors)
        total_fn = sum(e['false_negatives'] for e in self.errors)
        avg_error_rate = np.mean([e['error_rate'] for e in self.errors])

        # Distribución de tasas de error
        error_rates = [e['error_rate'] for e in self.errors]

        analysis = {
            'num_samples_with_errors': len(self.errors),
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'fp_fn_ratio': total_fp / (total_fn + 1e-8),
            'avg_error_rate': avg_error_rate,
            'max_error_rate': max(error_rates),
            'min_error_rate': min(error_rates),
            'error_rate_std': np.std(error_rates)
        }

        return analysis

    def visualize_worst_predictions(
        self,
        n: int = 5,
        save_path: Optional[str] = None
    ):
        """
        Visualiza las peores predicciones.

        Args:
            n: Número de ejemplos a mostrar
            save_path: Ruta para guardar
        """
        # Ordenar por tasa de error
        sorted_errors = sorted(self.errors, key=lambda x: -x['error_rate'])[:n]

        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, error in enumerate(sorted_errors):
            # Imagen original (mostrar banda RGB si está disponible)
            if 'image' in error:
                img = error['image']
                if img.shape[0] >= 3:
                    rgb = np.transpose(img[:3], (1, 2, 0))
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                else:
                    rgb = img[0]
                axes[i, 0].imshow(rgb)
            axes[i, 0].set_title('Imagen')
            axes[i, 0].axis('off')

            # Ground truth
            axes[i, 1].imshow(error['target'], cmap='RdYlGn_r')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            # Predicción
            axes[i, 2].imshow(error['prediction'], cmap='RdYlGn_r')
            axes[i, 2].set_title('Predicción')
            axes[i, 2].axis('off')

            # Máscara de errores
            error_vis = np.zeros((*error['error_mask'].shape, 3))
            error_vis[..., 0] = (error['prediction'] == 1) & (error['target'] == 0)  # FP: rojo
            error_vis[..., 2] = (error['prediction'] == 0) & (error['target'] == 1)  # FN: azul
            axes[i, 3].imshow(error_vis)
            axes[i, 3].set_title(f'Errores (FP: {error["false_positives"]}, FN: {error["false_negatives"]})')
            axes[i, 3].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()


class LatentSpaceVisualizer:
    """
    Visualiza el espacio latente del modelo.
    """

    def __init__(self, model: nn.Module, layer_name: str = 'encoder'):
        self.model = model
        self.layer_name = layer_name
        self.features = []
        self.labels = []

    @torch.no_grad()
    def collect_features(
        self,
        dataloader,
        device: str = 'cuda',
        max_samples: int = 1000
    ):
        """
        Recolecta features del espacio latente.
        """
        self.model.eval()
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        count = 0
        for images, masks in dataloader:
            if count >= max_samples:
                break

            images = images.to(device)

            # Obtener features del encoder
            if hasattr(self.model, 'get_encoder_features'):
                features = self.model.get_encoder_features(images[:, -1])[-1]
            else:
                # Forward normal
                _ = self.model(images)
                features = self.model.activations if hasattr(self.model, 'activations') else None

            if features is not None:
                # Global average pooling
                features = F.adaptive_avg_pool2d(features, 1).squeeze()
                self.features.append(features.cpu().numpy())

                # Label: proporción de deforestación
                label = masks.float().mean(dim=(1, 2)).numpy()
                self.labels.extend(label)

                count += images.shape[0]

        self.features = np.vstack(self.features)
        self.labels = np.array(self.labels)

    def visualize(
        self,
        method: str = 'tsne',
        save_path: Optional[str] = None
    ):
        """
        Visualiza el espacio latente.

        Args:
            method: 'tsne' o 'pca'
            save_path: Ruta para guardar
        """
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2)

        embedded = reducer.fit_transform(self.features)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embedded[:, 0],
            embedded[:, 1],
            c=self.labels,
            cmap='RdYlGn_r',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter, label='Proporción de Deforestación')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.title(f'Espacio Latente ({method.upper()})')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def generate_interpretability_report(
    model: nn.Module,
    test_loader,
    output_dir: str,
    device: str = 'cuda'
):
    """
    Genera un reporte completo de interpretabilidad.

    Args:
        model: Modelo a analizar
        test_loader: DataLoader de prueba
        output_dir: Directorio de salida
        device: Dispositivo de cómputo
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 1. Análisis de errores
    print("Analizando errores...")
    error_analyzer = ErrorAnalyzer()

    for images, masks in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        error_analyzer.collect_errors(preds, masks.to(device), images[:, -1])

    error_analysis = error_analyzer.analyze()
    print(f"  Total FP: {error_analysis['total_false_positives']}")
    print(f"  Total FN: {error_analysis['total_false_negatives']}")
    print(f"  Error rate promedio: {error_analysis['avg_error_rate']:.4f}")

    error_analyzer.visualize_worst_predictions(
        n=5,
        save_path=output_dir / 'worst_predictions.png'
    )

    # 2. Atención temporal (si el modelo lo soporta)
    if hasattr(model, 'get_attention_maps'):
        print("Visualizando atención temporal...")
        attn_viz = TemporalAttentionVisualizer(model)

        sample_batch = next(iter(test_loader))
        sample_input = sample_batch[0][:1].to(device)

        attn_viz.visualize(
            sample_input,
            save_path=output_dir / 'temporal_attention.png'
        )

    # 3. Grad-CAM
    print("Generando Grad-CAM...")
    # Encontrar última capa convolucional
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module

    if target_layer is not None:
        grad_cam = GradCAM(model, target_layer)

        # Generar para algunas muestras
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        sample_batch = next(iter(test_loader))
        for i in range(min(4, sample_batch[0].shape[0])):
            sample_input = sample_batch[0][i:i+1].to(device)

            # Imagen original
            img = sample_input[0, -1, :3].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[0, i].imshow(img)
            axes[0, i].set_title('Imagen')
            axes[0, i].axis('off')

            # Grad-CAM
            cam = grad_cam.generate(sample_input, target_class=1)
            axes[1, i].imshow(img)
            axes[1, i].imshow(cam, cmap='jet', alpha=0.5)
            axes[1, i].set_title('Grad-CAM')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / 'grad_cam.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Reporte de interpretabilidad guardado en: {output_dir}")
