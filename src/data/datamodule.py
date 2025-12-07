from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from config import TrainConfig
from data.transforms import build_transforms


class DataModule:
    """
    ImageFolder-based datamodule for data/train|validation|test/<class>.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.train_dataset: Optional[ImageFolder] = None
        self.val_dataset: Optional[ImageFolder] = None
        self.test_dataset: Optional[ImageFolder] = None
        self._logger = None

    def setup(self) -> None:
        root = Path(self.cfg.data_root)
        train_tfms = build_transforms(self.cfg, train=True)
        eval_tfms = build_transforms(self.cfg, train=False)

        self.train_dataset = ImageFolder(root / "train", transform=train_tfms)
        self.val_dataset = ImageFolder(root / "validation", transform=eval_tfms)
        self.test_dataset = ImageFolder(root / "test", transform=eval_tfms)
        if self.cfg.data_integrity_check:
            self._report_integrity()

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
        persistent = self.cfg.num_workers > 0
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=persistent,
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

    def _report_integrity(self) -> None:
        """
        Lightweight integrity summary to spot split leakage/conflicting labels by coordinate stem.
        """
        import logging

        logger = self._logger or logging.getLogger(__name__)

        def coord_key(path: Path) -> str:
            return path.stem

        split_datasets = {
            "train": self.train_dataset,
            "validation": self.val_dataset,
            "test": self.test_dataset,
        }
        coord_labels = {}
        coord_counts = {}
        class_counts = {}
        for split, ds in split_datasets.items():
            if ds is None:
                continue
            from collections import Counter

            labels_for_coord = {}
            counts_for_coord = {}
            class_counter = Counter()
            for path, cls_idx in ds.samples:  # type: ignore[attr-defined]
                key = coord_key(Path(path))
                lbl = ds.classes[cls_idx]
                labels_for_coord.setdefault(key, set()).add(lbl)
                counts_for_coord[key] = counts_for_coord.get(key, 0) + 1
                class_counter[lbl] += 1
            coord_labels[split] = labels_for_coord
            coord_counts[split] = counts_for_coord
            class_counts[split] = class_counter

        def conflicting_count(labels_map):
            return sum(1 for labels in labels_map.values() if len(labels) > 1)

        conflict_train = conflicting_count(coord_labels.get("train", {}))
        conflict_val = conflicting_count(coord_labels.get("validation", {}))
        conflict_test = conflicting_count(coord_labels.get("test", {}))

        coords_train = set(coord_labels.get("train", {}))
        coords_val = set(coord_labels.get("validation", {}))
        coords_test = set(coord_labels.get("test", {}))

        overlap_train_val = len(coords_train & coords_val)
        overlap_train_test = len(coords_train & coords_test)
        overlap_val_test = len(coords_val & coords_test)

        logger.info(
            "Data integrity (by coordinate): "
            f"train coords={len(coords_train)}, val coords={len(coords_val)}, test coords={len(coords_test)}, "
            f"conflicts(train/val/test)={conflict_train}/{conflict_val}/{conflict_test}, "
            f"overlap train∩val={overlap_train_val}, train∩test={overlap_train_test}, val∩test={overlap_val_test}"
        )
        logger.info(
            "Split class counts: "
            f"train={dict(class_counts.get('train', {}))}, "
            f"val={dict(class_counts.get('validation', {}))}, "
            f"test={dict(class_counts.get('test', {}))}"
        )
