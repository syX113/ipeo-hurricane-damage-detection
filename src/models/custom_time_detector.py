from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PixelShuffleDestroyer(nn.Module):
    """
    Randomly shuffles pixels within each image (keeps channels aligned).
    Used to test whether the model relies on global color statistics vs spatial structure.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed
        self._generators = {}

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            gen = torch.Generator(device=device)
            if self.seed is not None:
                gen.manual_seed(self.seed)
            self._generators[device] = gen
        return self._generators[device]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.flatten(2)  # (b, c, hw)
        gen = self._get_generator(x.device)
        perm = torch.randperm(h * w, generator=gen, device=x.device)
        flat = flat[:, :, perm]
        return flat.view(b, c, h, w)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, p_drop: float = 0.0, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ]
        if p_drop > 0:
            layers.append(nn.Dropout2d(p_drop))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomTimeDetector(nn.Module):
    """
    Variant of CustomCNN with options to destroy spatial structure (pixel shuffle or 1x1 convs).
    Captures activations for Grad-CAM style visualizations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
        shuffle_pixels: bool = False,
        pointwise_conv: bool = False,
        shuffle_seed: Optional[int] = None,
    ):
        super().__init__()
        self.shuffle_pixels = shuffle_pixels
        self.shuffle = PixelShuffleDestroyer(seed=shuffle_seed) if shuffle_pixels else None
        k = 1 if pointwise_conv else 3
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, p_drop=0.05, kernel_size=k),
            ConvBlock(32, 64, p_drop=0.05, kernel_size=k),
            ConvBlock(64, 128, p_drop=0.05, kernel_size=k),
            ConvBlock(128, 256, p_drop=0.05, kernel_size=k),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        head = [nn.Flatten(), nn.Linear(256, 128), nn.ReLU(inplace=True)]
        if dropout:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(128, num_classes))
        self.classifier = nn.Sequential(*head)
        self._cam_feats: Optional[torch.Tensor] = None
        self._cam_grads: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shuffle is not None:
            x = self.shuffle(x)
        x = self.features(x)
        self._cam_feats = x
        if x.requires_grad:
            x.register_hook(self._save_cam_grads)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def _save_cam_grads(self, grad: torch.Tensor) -> None:
        self._cam_grads = grad

    def get_cam_features(self) -> Optional[torch.Tensor]:
        return self._cam_feats

    def get_cam_grads(self) -> Optional[torch.Tensor]:
        return self._cam_grads
