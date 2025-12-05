from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from hurricane.config import TrainConfig
from hurricane.data.transforms import build_transforms


class DataModule:
    """
    ImageFolder-based datamodule for data/train|validation|test/<class>.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.train_dataset: Optional[ImageFolder] = None
        self.val_dataset: Optional[ImageFolder] = None
        self.test_dataset: Optional[ImageFolder] = None

    def setup(self) -> None:
        root = Path(self.cfg.data_root)
        train_tfms = build_transforms(self.cfg, train=True)
        eval_tfms = build_transforms(self.cfg, train=False)

        self.train_dataset = ImageFolder(root / "train", transform=train_tfms)
        self.val_dataset = ImageFolder(root / "validation", transform=eval_tfms)
        self.test_dataset = ImageFolder(root / "test", transform=eval_tfms)

    def _build_sampler(self, dataset: ImageFolder):
        if self.cfg.balance_strategy != "weighted_sampler":
            return None
        counts = {}
        for label in dataset.targets:
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        class_weights = {cls: total / count for cls, count in counts.items()}
        sample_weights = [class_weights[label] for label in dataset.targets]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    def _loader(self, dataset: ImageFolder, sampler=None, shuffle=False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup() first."
        sampler = self._build_sampler(self.train_dataset)
        return self._loader(self.train_dataset, sampler=sampler, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call setup() first."
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "Call setup() first."
        return self._loader(self.test_dataset, shuffle=False)

    def num_classes(self) -> int:
        if self.train_dataset is None:
            return 0
        return len(self.train_dataset.classes)
