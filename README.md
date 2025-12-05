# IPEO: Hurricane Damage Detection

## Data layout
```
data/
  train/damage
  train/no_damage
  validation/damage
  validation/no_damage
  test/damage
  test/no_damage
```

## Quick start
```python
from hurricane import TrainConfig, Trainer

cfg = TrainConfig(data_root="data", epochs=5, wandb_mode="disabled")
trainer = Trainer(cfg)
trainer.fit()
print(trainer.test())
```

## Module overview
- `hurricane.config`: `TrainConfig` and `build_config_from_dict` for easy overrides.
- `hurricane.data`: ImageFolder datamodule + transforms.
- `hurricane.models`: torchvision builders (swap via `model_name`: resnet18/34, efficientnet_b0, convnext_tiny).
- `hurricane.training`: Trainer orchestrating train/val/test, checkpoints, optional temperature scaling.
- `hurricane.validation`: metrics (accuracy/F1/Brier/ECE), evaluation loop, reliability bins, temperature scaler.
- `hurricane.utils`: logging, reproducibility, checkpoint loader.

## Calibration & tuning
- Set `apply_temperature=True` to fit a temperature scaler on the best val logits.
- Use `balance_strategy` (`weighted_sampler` or `class_weights`) to handle imbalance.
- Compose W&B sweeps from the config fields (see `TrainConfig`).
