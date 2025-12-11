def get_logger(name: str = "hurricane"):
    from .logging import get_logger as _get_logger

    return _get_logger(name=name)


def init_wandb(cfg):
    from .logging import init_wandb as _init_wandb

    return _init_wandb(cfg)


def set_seed(seed: int):
    from .reproducibility import set_seed as _set_seed

    return _set_seed(seed)


def load_checkpoint(checkpoint_path, map_location="cpu"):
    """
    Lazy import to avoid pulling torch/numpy when only light utils are needed.
    """
    from .checkpoint import load_checkpoint as _load_checkpoint

    return _load_checkpoint(checkpoint_path, map_location=map_location)


def resolve_project_root():
    from .paths import resolve_project_root as _resolve_project_root

    return _resolve_project_root()


__all__ = [
    "get_logger",
    "init_wandb",
    "set_seed",
    "load_checkpoint",
    "resolve_project_root",
]
