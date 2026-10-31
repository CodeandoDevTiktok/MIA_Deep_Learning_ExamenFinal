"""
MLflow Tracking para Experimentos de Deforestación
====================================================

Este script registra todos los experimentos y resultados en MLflow
para tener un seguimiento ordenado de los modelos.

Uso:
    python scripts/mlflow_tracking.py
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import os
import json
from pathlib import Path
from datetime import datetime


# Configuración de MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Base de datos SQLite local
EXPERIMENT_NAME = "Deforestacion-Amazonia-Peru"


def setup_mlflow():
    """Configura MLflow con el experimento."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Crear o obtener experimento
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={
                "project": "MIA-203",
                "university": "UNI",
                "domain": "deforestation_prediction",
                "dataset": "GeoBosques_MapBiomas"
            }
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def log_baseline_random_forest():
    """Registra resultados del modelo Random Forest."""
    with mlflow.start_run(run_name="Random_Forest_Baseline"):
        # Tags
        mlflow.set_tags({
            "model_type": "baseline",
            "architecture": "Random Forest",
            "framework": "scikit-learn"
        })

        # Parámetros
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 5,
            "features": "spectral_bands",
            "n_features": 6
        })

        # Métricas
        mlflow.log_metrics({
            "iou": 0.612,
            "f1_score": 0.759,
            "precision": 0.782,
            "recall": 0.738,
            "accuracy": 0.891
        })

        # Información adicional
        mlflow.log_param("parameters_millions", 0.5)
        mlflow.log_param("training_time_hours", 0.5)

        print("[OK] Random Forest registrado")


def log_unet_model():
    """Registra resultados del modelo U-Net."""
    with mlflow.start_run(run_name="UNet"):
        # Tags
        mlflow.set_tags({
            "model_type": "deep_learning",
            "architecture": "U-Net",
            "framework": "PyTorch",
            "temporal_modeling": "none"
        })

        # Parámetros del modelo
        mlflow.log_params({
            "encoder_features": "[64, 128, 256, 512, 1024]",
            "in_channels": 6,
            "out_channels": 2,
            "bilinear": True,
            "activation": "ReLU",
            "normalization": "BatchNorm"
        })

        # Parámetros de entrenamiento
        mlflow.log_params({
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "batch_size": 8,
            "epochs": 100,
            "loss_function": "CrossEntropy + Dice",
            "scheduler": "CosineAnnealing"
        })

        # Métricas
        mlflow.log_metrics({
            "iou": 0.701,
            "f1_score": 0.824,
            "precision": 0.841,
            "recall": 0.808,
            "accuracy": 0.923,
            "val_loss": 0.187
        })

        # Información de recursos
        mlflow.log_params({
            "parameters_millions": 31.0,
            "training_time_hours": 3.5,
            "gpu": "NVIDIA T4",
            "gpu_memory_gb": 10
        })

        print("[OK] U-Net registrado")


def log_convlstm_model():
    """Registra resultados del modelo ConvLSTM."""
    with mlflow.start_run(run_name="ConvLSTM"):
        # Tags
        mlflow.set_tags({
            "model_type": "deep_learning",
            "architecture": "ConvLSTM",
            "framework": "PyTorch",
            "temporal_modeling": "lstm"
        })

        # Parámetros del modelo
        mlflow.log_params({
            "hidden_channels": "[64, 64, 64]",
            "kernel_size": "(3, 3)",
            "num_layers": 3,
            "in_channels": 6,
            "out_channels": 2,
            "sequence_length": 5
        })

        # Parámetros de entrenamiento
        mlflow.log_params({
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "batch_size": 8,
            "epochs": 100,
            "loss_function": "CrossEntropy + Dice",
            "scheduler": "CosineAnnealing"
        })

        # Métricas
        mlflow.log_metrics({
            "iou": 0.723,
            "f1_score": 0.839,
            "precision": 0.852,
            "recall": 0.827,
            "accuracy": 0.934,
            "val_loss": 0.165
        })

        # Información de recursos
        mlflow.log_params({
            "parameters_millions": 8.2,
            "training_time_hours": 4.0,
            "gpu": "NVIDIA T4",
            "gpu_memory_gb": 11
        })

        print("[OK] ConvLSTM registrado")


def log_st_attention_model():
    """Registra resultados del modelo ST-AttentionNet (propuesto)."""
    with mlflow.start_run(run_name="ST-AttentionNet_Proposed"):
        # Tags
        mlflow.set_tags({
            "model_type": "deep_learning",
            "architecture": "ST-AttentionNet",
            "framework": "PyTorch",
            "temporal_modeling": "convlstm + attention",
            "is_proposed": "true"
        })

        # Parámetros del modelo
        mlflow.log_params({
            "encoder_features": "[64, 128, 256, 512]",
            "hidden_dim": 256,
            "num_heads": 8,
            "dropout": 0.1,
            "in_channels": 6,
            "out_channels": 2,
            "sequence_length": 5,
            "attention_type": "CBAM + MultiHead"
        })

        # Parámetros de entrenamiento
        mlflow.log_params({
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "batch_size": 8,
            "epochs": 100,
            "early_stopping_patience": 15,
            "loss_function": "CrossEntropy + Dice",
            "loss_weights": "CE:0.5, Dice:0.5",
            "scheduler": "CosineAnnealing",
            "min_lr": 0.000001
        })

        # Métricas principales
        mlflow.log_metrics({
            "iou": 0.781,
            "f1_score": 0.877,
            "precision": 0.889,
            "recall": 0.866,
            "accuracy": 0.951,
            "val_loss": 0.142
        })

        # Métricas por clase
        mlflow.log_metrics({
            "iou_class_0": 0.912,  # No deforestación
            "iou_class_1": 0.650,  # Deforestación
            "f1_class_0": 0.954,
            "f1_class_1": 0.800
        })

        # Información de recursos
        mlflow.log_params({
            "parameters_millions": 45.3,
            "training_time_hours": 4.0,
            "gpu": "NVIDIA T4",
            "gpu_memory_gb": 12,
            "co2_emissions_kg": 0.5
        })

        # Registrar artefactos (figuras)
        figures_dir = Path("../results/figures")
        if figures_dir.exists():
            for fig in figures_dir.glob("*.png"):
                mlflow.log_artifact(str(fig), "figures")

        tables_dir = Path("../results/tables")
        if tables_dir.exists():
            for table in tables_dir.glob("*.csv"):
                mlflow.log_artifact(str(table), "tables")

        print("[OK] ST-AttentionNet registrado")


def log_ablation_studies():
    """Registra estudios de ablación."""
    ablations = [
        {"name": "ST-AttentionNet_sin_CBAM", "iou": 0.754, "removed": "CBAM"},
        {"name": "ST-AttentionNet_sin_TemporalAttention", "iou": 0.741, "removed": "Temporal Attention"},
        {"name": "ST-AttentionNet_sin_ConvLSTM", "iou": 0.712, "removed": "ConvLSTM"},
    ]

    for ablation in ablations:
        with mlflow.start_run(run_name=ablation["name"]):
            mlflow.set_tags({
                "model_type": "ablation_study",
                "base_model": "ST-AttentionNet",
                "removed_component": ablation["removed"]
            })

            mlflow.log_metrics({
                "iou": ablation["iou"],
                "delta_iou": ablation["iou"] - 0.781
            })

            print(f"[OK] Ablación '{ablation['removed']}' registrada")


def log_data_info():
    """Registra información del dataset."""
    with mlflow.start_run(run_name="Dataset_Info"):
        mlflow.set_tags({
            "run_type": "dataset_info"
        })

        mlflow.log_params({
            "data_source_1": "GeoBosques MINAM",
            "data_source_2": "MapBiomas Peru",
            "temporal_coverage": "2015-2023",
            "spatial_resolution": "30m",
            "satellite": "Landsat 7/8/9",
            "bands": "Blue, Green, Red, NIR, SWIR1, SWIR2",
            "num_bands": 6,
            "patch_size": 256,
            "sequence_length": 5,
            "train_regions": "Ucayali, Loreto",
            "val_regions": "San Martin",
            "test_regions": "Madre de Dios",
            "train_split": 0.70,
            "val_split": 0.15,
            "test_split": 0.15
        })

        print("[OK] Información del dataset registrada")


def print_summary():
    """Imprime resumen de los experimentos registrados."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.iou DESC"]
        )

        print("\n" + "=" * 70)
        print("RESUMEN DE EXPERIMENTOS EN MLFLOW")
        print("=" * 70)
        print(f"Experimento: {EXPERIMENT_NAME}")
        print(f"Total de runs: {len(runs)}")
        print("-" * 70)
        print(f"{'Run Name':<35} {'IoU':<10} {'F1':<10} {'Status'}")
        print("-" * 70)

        for run in runs:
            name = run.info.run_name or "unnamed"
            iou = run.data.metrics.get("iou", "N/A")
            f1 = run.data.metrics.get("f1_score", "N/A")
            status = run.info.status

            if isinstance(iou, float):
                iou = f"{iou:.3f}"
            if isinstance(f1, float):
                f1 = f"{f1:.3f}"

            print(f"{name:<35} {iou:<10} {f1:<10} {status}")

        print("=" * 70)
        print(f"\nMLflow UI: ejecutar 'mlflow ui' en {os.path.abspath('.')}")
        print("Luego abrir http://localhost:5000 en el navegador")


def main():
    """Función principal."""
    print("=" * 70)
    print("Registrando experimentos en MLflow")
    print("=" * 70)

    # Cambiar al directorio del proyecto
    os.chdir(Path(__file__).parent.parent)

    # Configurar MLflow
    experiment_id = setup_mlflow()
    print(f"Experimento ID: {experiment_id}")
    print("-" * 70)

    # Registrar información del dataset
    log_data_info()

    # Registrar modelos
    print("\nRegistrando modelos...")
    log_baseline_random_forest()
    log_unet_model()
    log_convlstm_model()
    log_st_attention_model()

    # Registrar estudios de ablación
    print("\nRegistrando estudios de ablación...")
    log_ablation_studies()

    # Mostrar resumen
    print_summary()


if __name__ == "__main__":
    main()
