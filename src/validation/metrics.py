from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, precision_recall_fscore_support


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    probs_cpu = probs.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    y_pred = preds.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    if probs.shape[1] <= 2:
        # Binary Brier score uses probability of positive class
        y_prob = probs[:, -1].detach().cpu().numpy() if probs.shape[1] > 1 else probs_cpu.squeeze()
        brier = brier_score_loss(y_true, y_prob)
    else:
        # Multiclass Brier via one-hot targets
        y_true_oh = np.eye(probs.shape[1])[y_true]
        brier = float(np.mean(np.sum((probs_cpu - y_true_oh) ** 2, axis=1)))
    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "brier": brier,
    }


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = preds == labels
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)
    for i in range(n_bins):
        # Include upper edge on last bin to cover conf==1.0
        upper_inclusive = i == n_bins - 1
        mask = (conf >= bins[i]) & (conf <= bins[i + 1] if upper_inclusive else conf < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf[i] = conf[mask].mean()
        bin_acc[i] = acc[mask].mean()
        bin_count[i] = mask.sum()
    return bin_conf, bin_acc, bin_count


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bin_conf, bin_acc, bin_count = reliability_bins(probs, labels, n_bins)
    if bin_count.sum() == 0:
        return 0.0
    return float(np.sum(np.abs(bin_acc - bin_conf) * bin_count) / np.sum(bin_count))
