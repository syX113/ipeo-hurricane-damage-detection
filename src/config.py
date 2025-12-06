from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TrainConfig:
    # Data
    data_root: str = "data"
    batch_size: int = 64
    num_workers: int = 4
    image_size: int = 256
    mean: Tuple[float, float, float] = (0.430, 0.411, 0.374)
    std: Tuple[float, float, float] = (0.218, 0.207, 0.203)
    horizontal_flip: float = 0.5
    max_rotation: float = 15.0
    color_jitter: Tuple[float, float, float, float] = (0.15, 0.15, 0.15, 0.05)
    balance_strategy: Optional[str] = "weighted_sampler"  # None, weighted_sampler, class_weights

    # Model
    model_name: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.1
    in_channels: int = 3
    freeze_backbone: bool = False

    # Optimization
    optimizer: str = "adamw"  # adamw or sgd
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9
    epochs: int = 20
    label_smoothing: float = 0.05
    amp: bool = True
    grad_clip_norm: Optional[float] = 1.0
    early_stopping: int = 5
    checkpoint_metric: str = "macro_f1"
    progress_bar: bool = True

    # Evaluation
    threshold: float = 0.5
    reliability_bins: int = 15
    apply_temperature: bool = True

    # Logging
    checkpoints_dir: str = "artifacts/checkpoints"
    wandb_mode: str = "disabled"  # disabled, online, offline
    wandb_project: str = "hurricane-damage"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    tensorboard: bool = False
    tensorboard_dir: str = "artifacts/tensorboard"

    # Misc
    seed: int = 42
    data_integrity_check: bool = False  # optional coordinate/leakage summary during setup

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_config_from_dict(overrides: Dict[str, Any]) -> TrainConfig:
    defaults = TrainConfig().to_dict()
    filtered = {k: v for k, v in overrides.items() if k in defaults}
    merged = {**defaults, **filtered}
    return TrainConfig(**merged)
