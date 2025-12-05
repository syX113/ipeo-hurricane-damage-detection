import logging
from typing import Optional

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from hurricane.config import TrainConfig


def get_logger(name: str = "hurricane") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def init_wandb(cfg: TrainConfig) -> Optional["wandb.sdk.wandb_run.Run"]:
    if cfg.wandb_mode == "disabled" or wandb is None:
        return None
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        mode=cfg.wandb_mode,
        tags=cfg.wandb_tags,
        config=cfg.to_dict(),
        save_code=True,
    )
