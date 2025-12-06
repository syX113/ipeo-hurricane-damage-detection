from functools import partial
from typing import Callable, Dict

import torch
import torch.nn as nn
from torchvision import models

from config import TrainConfig
from .custom import CustomCNN


def _get_weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    return {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
    }.get(name)


def _adapt_first_conv(module: nn.Module, in_channels: int) -> None:
    """Expand/shrink first conv to support non-RGB inputs without reinitializing everything."""
    if not hasattr(module, "conv1"):
        return
    conv1 = module.conv1  # type: ignore[attr-defined]
    if conv1.in_channels == in_channels:
        return
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, : conv1.in_channels] = conv1.weight
        if in_channels > conv1.in_channels:
            extra = in_channels - conv1.in_channels
            new_conv.weight[:, conv1.in_channels :, :, :] = conv1.weight[:, :extra, :, :]
    module.conv1 = new_conv  # type: ignore[attr-defined]


def build_model(cfg: TrainConfig) -> nn.Module:
    registry: Dict[str, Callable] = {
        "resnet18": partial(models.resnet18),
        "resnet34": partial(models.resnet34),
        "efficientnet_b0": partial(models.efficientnet_b0),
        "convnext_tiny": partial(models.convnext_tiny),
    }
    if cfg.model_name == "custom_cnn":
        model = CustomCNN(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
        )
        if cfg.freeze_backbone and hasattr(model, "features"):
            for name, param in model.features.named_parameters():  # type: ignore[attr-defined]
                param.requires_grad = False
        return model
    if cfg.model_name not in registry:
        raise ValueError(f"Model {cfg.model_name} not supported")

    weights = _get_weights(cfg.model_name, cfg.pretrained)
    model = registry[cfg.model_name](weights=weights)

    if hasattr(model, "fc"):
        in_features = model.fc.in_features  # type: ignore[attr-defined]
        head = []
        if cfg.dropout:
            head.append(nn.Dropout(cfg.dropout))
        head.append(nn.Linear(in_features, cfg.num_classes))
        model.fc = nn.Sequential(*head) if len(head) > 1 else head[0]  # type: ignore[attr-defined]
    elif hasattr(model, "classifier"):
        classifier = model.classifier  # type: ignore[attr-defined]
        if isinstance(classifier, nn.Sequential):
            in_features = classifier[-1].in_features  # type: ignore[index]
            classifier[-1] = nn.Linear(in_features, cfg.num_classes)  # type: ignore[index]
            model.classifier = classifier  # type: ignore[attr-defined]
        else:
            in_features = classifier.in_features  # type: ignore[attr-defined]
            model.classifier = nn.Linear(in_features, cfg.num_classes)  # type: ignore[attr-defined]
    else:
        raise RuntimeError("Unhandled head for selected model.")

    _adapt_first_conv(model, cfg.in_channels)

    if cfg.freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" in name or "classifier" in name:
                continue
            param.requires_grad = False

    return model
