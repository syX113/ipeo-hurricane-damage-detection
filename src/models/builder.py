from functools import partial
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

from config import TrainConfig
from .custom import CustomCNN
from .custom_time_detector import CustomTimeDetector


def _get_weights(name: str, pretrained: bool):
    if not pretrained:
        return None

    # Map model name to torchvision default weights
    return {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "efficientnet_b1": models.EfficientNet_B1_Weights.DEFAULT,
        "efficientnet_b2": models.EfficientNet_B2_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "convnext_small": models.ConvNeXt_Small_Weights.DEFAULT,
    }.get(name)


def _find_first_conv(module: nn.Module) -> Optional[Tuple[nn.Module, str, nn.Conv2d]]:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            return module, name, child
        found = _find_first_conv(child)
        if found is not None:
            return found
    return None


def _adapt_first_conv(module: nn.Module, in_channels: int) -> None:
    """Expand/shrink first conv to support non-RGB inputs without reinitializing everything."""
    target: Optional[Tuple[nn.Module, str, nn.Conv2d]] = None
    if hasattr(module, "conv1"):
        target = (module, "conv1", module.conv1)
    elif hasattr(module, "features"):
        target = _find_first_conv(module.features)
    if target is None:
        return
    parent, name, conv1 = target
    if conv1.in_channels == in_channels:
        return

    # Recreate first conv to match requested channels while keeping learned weights
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        copy_channels = min(conv1.in_channels, in_channels)
        new_conv.weight[:, :copy_channels] = conv1.weight[:, :copy_channels]
        if in_channels > copy_channels:
            extra = in_channels - copy_channels
            fill = conv1.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, copy_channels:, :, :] = fill.repeat(1, extra, 1, 1)
        if conv1.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv1.bias)
    setattr(parent, name, new_conv)


def build_model(cfg: TrainConfig) -> nn.Module:
    registry: Dict[str, Callable] = {
        "resnet18": partial(models.resnet18),
        "resnet34": partial(models.resnet34),
        "resnet50": partial(models.resnet50),
        "efficientnet_b0": partial(models.efficientnet_b0),
        "efficientnet_b1": partial(models.efficientnet_b1),
        "efficientnet_b2": partial(models.efficientnet_b2),
        "convnext_tiny": partial(models.convnext_tiny),
        "convnext_small": partial(models.convnext_small),
    }
    if cfg.model_name == "custom_cnn":

        # Lightweight baseline
        model = CustomCNN(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
        )
        if cfg.freeze_backbone and hasattr(model, "features"):
            for name, param in model.features.named_parameters():
                param.requires_grad = False
        return model
    if cfg.model_name == "custom_time_detector":

        # Variant that can destroy spatial structure for ablation
        model = CustomTimeDetector(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            shuffle_pixels=cfg.custom_shuffle_pixels,
            pointwise_conv=cfg.custom_pointwise_conv,
            shuffle_seed=cfg.seed,
        )
        if cfg.freeze_backbone and hasattr(model, "features"):
            for name, param in model.features.named_parameters():
                param.requires_grad = False
        return model
    if cfg.model_name not in registry:
        raise ValueError(f"Model {cfg.model_name} not supported")

    weights = _get_weights(cfg.model_name, cfg.pretrained)
    model = registry[cfg.model_name](weights=weights)

    if hasattr(model, "fc"):
        # Replace classification head for ResNet-like models
        in_features = model.fc.in_features
        head = []
        if cfg.dropout:
            head.append(nn.Dropout(cfg.dropout))
        head.append(nn.Linear(in_features, cfg.num_classes))
        model.fc = nn.Sequential(*head) if len(head) > 1 else head[0]
    elif hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):

            # EfficientNet/ConvNeXt style sequential head
            in_features = classifier[-1].in_features
            classifier[-1] = nn.Linear(in_features, cfg.num_classes)
            model.classifier = classifier
        else:
            in_features = classifier.in_features
            model.classifier = nn.Linear(in_features, cfg.num_classes)
    else:
        raise RuntimeError("Unhandled head for selected model.")

    _adapt_first_conv(model, cfg.in_channels)

    if cfg.freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" in name or "classifier" in name:
                continue

            # Only finetune head if freezing backbone
            param.requires_grad = False

    return model
