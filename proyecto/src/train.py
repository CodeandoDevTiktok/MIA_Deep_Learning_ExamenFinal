"""
Script de Entrenamiento para Modelos de Predicción de Deforestación
=====================================================================

Implementa el pipeline completo de entrenamiento con:
- Múltiples arquitecturas (U-Net, ConvLSTM, ST-AttentionNet)
- Logging con TensorBoard/Wandb
- Early stopping y checkpointing
- Métricas de segmentación
"""

import os
import argparse
import yaml
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import build_model, UNet, ConvLSTM, STAttentionNet
from dataset import create_dataloaders, create_synthetic_dataloaders
from evaluate import compute_metrics, SegmentationMetrics
from utils import EarlyStopping, save_checkpoint, load_checkpoint, set_seed


class Trainer:
    """
    Clase principal de entrenamiento.

    Args:
        config (dict): Configuración del experimento
        device (str): Dispositivo de cómputo ('cuda' o 'cpu')
    """

    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Configurar semillas para reproducibilidad
        set_seed(config.get('seed', 42))

        # Crear directorios de salida
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Inicializar componentes
        self._init_model()
        self._init_dataloaders()
        self._init_training()
        self._init_logging()

    def _init_model(self):
        """Inicializa el modelo."""
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'st_attention')

        self.model = build_model(
            model_name=model_name,
            in_channels=model_config.get('in_channels', 6),
            out_channels=model_config.get('out_channels', 2),
            **model_config.get('params', {})
        ).to(self.device)

        # Contar parámetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Modelo: {model_name}")
        print(f"  Parámetros totales: {total_params:,}")
        print(f"  Parámetros entrenables: {trainable_params:,}")

    def _init_dataloaders(self):
        """Inicializa los dataloaders."""
        data_config = self.config.get('data', {})

        if data_config.get('synthetic', False):
            # Usar datos sintéticos para pruebas
            self.train_loader, self.val_loader, self.test_loader = create_synthetic_dataloaders(
                num_train=data_config.get('num_train', 800),
                num_val=data_config.get('num_val', 100),
                num_test=data_config.get('num_test', 100),
                batch_size=data_config.get('batch_size', 8),
                sequence_length=data_config.get('sequence_length', 5),
                image_size=data_config.get('image_size', 128),
                num_channels=data_config.get('num_channels', 6)
            )
        else:
            self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
                root_dir=data_config.get('root_dir'),
                batch_size=data_config.get('batch_size', 8),
                sequence_length=data_config.get('sequence_length', 5),
                patch_size=data_config.get('patch_size', 256),
                num_workers=data_config.get('num_workers', 4)
            )

        print(f"Dataset:")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        print(f"  Test samples: {len(self.test_loader.dataset)}")

    def _init_training(self):
        """Inicializa componentes de entrenamiento."""
        train_config = self.config.get('training', {})

        # Función de pérdida
        self.criterion = self._get_criterion(train_config.get('loss', 'cross_entropy'))

        # Optimizador
        self.optimizer = self._get_optimizer(train_config)

        # Scheduler
        self.scheduler = self._get_scheduler(train_config)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=train_config.get('early_stopping_patience', 15),
            min_delta=train_config.get('early_stopping_delta', 1e-4)
        )

        # Mixed precision
        self.use_amp = train_config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Épocas
        self.epochs = train_config.get('epochs', 100)
        self.current_epoch = 0

    def _get_criterion(self, loss_type: str) -> nn.Module:
        """Retorna la función de pérdida."""
        if loss_type == 'cross_entropy':
            # Pesos de clase para desbalance (más peso a deforestación)
            weights = torch.tensor([1.0, 3.0]).to(self.device)
            return nn.CrossEntropyLoss(weight=weights)
        elif loss_type == 'focal':
            return FocalLoss(alpha=0.25, gamma=2.0)
        elif loss_type == 'dice':
            return DiceLoss()
        elif loss_type == 'combined':
            return CombinedLoss(ce_weight=0.5, dice_weight=0.5)
        else:
            return nn.CrossEntropyLoss()

    def _get_optimizer(self, config: Dict) -> optim.Optimizer:
        """Retorna el optimizador."""
        opt_name = config.get('optimizer', 'adamw')
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-4)

        if opt_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            return optim.AdamW(self.model.parameters(), lr=lr)

    def _get_scheduler(self, config: Dict):
        """Retorna el scheduler de learning rate."""
        scheduler_name = config.get('scheduler', 'cosine')

        if scheduler_name == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('epochs', 100),
                eta_min=config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None

    def _init_logging(self):
        """Inicializa logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.output_dir / 'logs' / timestamp
        self.writer = SummaryWriter(log_dir)

        # Métricas tracker
        self.metrics = SegmentationMetrics(num_classes=2)

    def train_epoch(self) -> Dict:
        """Entrena una época."""
        self.model.train()
        self.metrics.reset()

        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            with autocast(enabled=self.use_amp):
                # Para U-Net, usar solo el último frame
                if isinstance(self.model, UNet):
                    outputs = self.model(images[:, -1])  # [B, C, H, W]
                else:
                    outputs = self.model(images)  # [B, T, C, H, W] -> [B, num_classes, H, W]

                loss = self.criterion(outputs, masks)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Actualizar métricas
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            self.metrics.update(preds, masks)

            # Actualizar progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'IoU': f"{self.metrics.get_iou():.4f}"
            })

        # Calcular métricas de la época
        epoch_metrics = self.metrics.compute()
        epoch_metrics['loss'] = total_loss / len(self.train_loader)

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict:
        """Valida el modelo."""
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0

        for images, masks in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            if isinstance(self.model, UNet):
                outputs = self.model(images[:, -1])
            else:
                outputs = self.model(images)

            loss = self.criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            self.metrics.update(preds, masks)

        # Calcular métricas
        val_metrics = self.metrics.compute()
        val_metrics['loss'] = total_loss / len(self.val_loader)

        return val_metrics

    def train(self):
        """Ejecuta el loop de entrenamiento completo."""
        print(f"\nIniciando entrenamiento en {self.device}")
        print(f"Épocas: {self.epochs}")
        print("-" * 50)

        best_val_iou = 0.0

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Entrenar
            train_metrics = self.train_epoch()

            # Validar
            val_metrics = self.validate()

            # Actualizar scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Logging
            self._log_metrics(train_metrics, val_metrics, epoch)

            # Imprimir métricas
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")

            # Guardar mejor modelo
            if val_metrics['iou'] > best_val_iou:
                best_val_iou = val_metrics['iou']
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    self.checkpoint_dir / 'best_model.pth'
                )
                print(f"  Nuevo mejor modelo guardado (IoU: {best_val_iou:.4f})")

            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping en época {epoch + 1}")
                break

        # Guardar modelo final
        save_checkpoint(
            self.model,
            self.optimizer,
            self.current_epoch,
            val_metrics,
            self.checkpoint_dir / 'final_model.pth'
        )

        print(f"\nEntrenamiento completado. Mejor IoU: {best_val_iou:.4f}")
        self.writer.close()

        return best_val_iou

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log de métricas a TensorBoard."""
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)

        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)

        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)


# ============================================================================
# Funciones de Pérdida Adicionales
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss para clases desbalanceadas."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss para segmentación."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(inputs, dim=1)[:, 1]  # Probabilidad de clase 1
        targets_float = targets.float()

        intersection = (probs * targets_float).sum()
        union = probs.sum() + targets_float.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combinación de Cross Entropy y Dice Loss."""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(inputs, targets) + self.dice_weight * self.dice(inputs, targets)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo de predicción de deforestación')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Ruta al archivo de configuración')
    parser.add_argument('--model', type=str, default=None,
                        help='Nombre del modelo (sobrescribe config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Número de épocas (sobrescribe config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Dispositivo de cómputo')
    args = parser.parse_args()

    # Cargar configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Sobrescribir con argumentos de línea de comandos
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs

    # Crear trainer y entrenar
    trainer = Trainer(config, device=args.device)
    trainer.train()


if __name__ == '__main__':
    main()
