from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from config import TrainConfig
from validation.calibration import TemperatureScaler
from validation.metrics import classification_metrics, expected_calibration_error


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    temperature: Optional[TemperatureScaler] = None,
) -> Tuple[Dict, Dict]:
    model.eval()
    logits_list = []
    labels_list = []
    for batch in dataloader:
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels, _ = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        if temperature is not None:
            outputs = temperature(outputs)
        logits_list.append(outputs)
        labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    metrics = classification_metrics(logits, labels)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    metrics["ece"] = expected_calibration_error(probs, labels.cpu().numpy(), cfg.reliability_bins)
    return metrics, {"logits": logits, "labels": labels, "probs": probs}
