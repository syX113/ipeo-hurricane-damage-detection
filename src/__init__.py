"""
Modular pipeline for hurricane damage detection.

Submodules:
- data: ImageFolder datamodule + transforms
- models: torchvision backbones with swappable heads
- training: trainer orchestration
- validation: metrics, evaluation, calibration
- utils: logging, seeding, checkpoints
"""

from .config import TrainConfig, build_config_from_dict
from .data import DataModule, build_transforms
from .models import build_model
from .training import Trainer
from .validation import (
    TemperatureScaler,
    classification_metrics,
    evaluate,
    expected_calibration_error,
    reliability_bins,
)
from .utils import get_logger, init_wandb, load_checkpoint, set_seed

__all__ = [
    "TrainConfig",
    "build_config_from_dict",
    "DataModule",
    "build_transforms",
    "build_model",
    "Trainer",
    "TemperatureScaler",
    "classification_metrics",
    "evaluate",
    "expected_calibration_error",
    "reliability_bins",
    "get_logger",
    "init_wandb",
    "set_seed",
    "load_checkpoint",
]
