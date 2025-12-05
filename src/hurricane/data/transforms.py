from typing import Dict, List

from torchvision import transforms

from hurricane.config import TrainConfig


def build_transforms(cfg: TrainConfig, train: bool = True) -> transforms.Compose:
    steps: List = [transforms.Resize((cfg.image_size, cfg.image_size))]
    if train:
        if cfg.horizontal_flip:
            steps.append(transforms.RandomHorizontalFlip(cfg.horizontal_flip))
        if cfg.max_rotation:
            steps.append(transforms.RandomRotation(cfg.max_rotation))
        if any(cfg.color_jitter):
            b, c, s, h = cfg.color_jitter
            steps.append(transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )
    return transforms.Compose(steps)


def describe_transforms(cfg: TrainConfig) -> Dict:
    return {
        "image_size": cfg.image_size,
        "horizontal_flip": cfg.horizontal_flip,
        "max_rotation": cfg.max_rotation,
        "color_jitter": cfg.color_jitter,
    }
