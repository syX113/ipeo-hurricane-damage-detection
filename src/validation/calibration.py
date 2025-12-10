from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        # Single learnable scalar applied to logits
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-6)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01) -> None:
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        self.eval()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"temperature": self.temperature.detach().cpu()}, path)

    @staticmethod
    def load(path: str, device: Optional[torch.device] = None) -> "TemperatureScaler":
        state = torch.load(path, map_location=device or "cpu")
        scaler = TemperatureScaler()
        scaler.temperature = nn.Parameter(state["temperature"])
        return scaler
