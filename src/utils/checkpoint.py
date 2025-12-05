from typing import Tuple

import torch

from config import TrainConfig, build_config_from_dict
from models.builder import build_model


def load_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Tuple[torch.nn.Module, TrainConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    cfg = build_config_from_dict(checkpoint.get("config", {}))
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state"])
    return model, cfg
