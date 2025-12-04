# Hurricane Damage Detection — Modeling Plan (EDA-Driven)

## Objectives
- Build a damage vs. no_damage classifier with reliable probabilities (calibration is explicit).
- Maintain reproducibility and clear evaluation: accuracy/F1 plus calibration metrics on the provided val/test splits.
- Mitigate data issues discovered in EDA: class imbalance, co-located/conflicting labels, potential leakage, and noise.

## Key EDA Findings Driving the Approach
- **Class balance:** Train skewed (~13k damage / 6k no_damage); val/test balanced (1k/1k). → Need imbalance handling to avoid bias toward damage.
- **Co-located/conflicting labels:** Thousands of coordinates appear with both labels; many coords have multiple images. → Expect label noise; guard against overconfidence and leakage.
- **Duplicates:** Exact duplicates exist across splits. → Remove or exclude train duplicates overlapping val/test to avoid leakage.
- **Spatial distribution:** Clusters in a few regions; val/test noisier per brief. → Mild augmentation and regularization for shift robustness.
- **Image size:** Uniform 256×256 RGB. → Keep native size; no aggressive rescaling.
- **Pixel stats/brightness/contrast:** Moderate variation; val/test noisier. → Mild color jitter; dataset-specific normalization (not ImageNet).

## Data Curation & Filters (reduce leakage/noise)
1. **Corruption check** (Pillow verify) – avoid training crashes.
2. **Duplicate hashing** (MD5) across all splits – drop train images that duplicate val/test to prevent leakage; deduplicate within train to avoid overweighting.
3. **Conflicting-label coords** – identify coords with mixed labels. Options:
   - Keep but use label smoothing + calibrated training.
   - Downweight samples from highly conflicted coords.
   - For a small, noisy subset, remove after visual review (use the “same coord” panels).
4. **Train/val/test integrity** – never mix val/test into training; if you need a tuning split, carve it from train only.

## Data Loading & Transforms (respect semantics, mitigate noise)
- **Normalization:** Use dataset means/stds from EDA (not ImageNet) → stabilizes training on true distribution.
- **Resize:** Keep 256×256; avoids extra interpolation.
- **Train-time augmentations:** RandomHorizontalFlip, small Rotation (≤20°), mild ColorJitter (consistent with brightness/contrast spread). Avoid heavy crops/cutouts early to keep damage cues intact.
- **Eval transforms:** Resize + Normalize only.
- **Sampling:** WeightedRandomSampler or class-weighted loss to counter imbalance; still monitor per-class metrics.

## Models to Compare (capacity vs. overfit vs. calibration)
- **Baselines:** ResNet-18/34 pretrained on ImageNet — strong starting point, fast to train, decent calibration with temperature scaling.
- **Medium capacity:** EfficientNet-B0/B2, ConvNeXt-T — better accuracy potential; watch overfitting on noisy labels.
- **Optional robustness variants:** MixUp/CutMix after baseline if overfitting appears (but monitor calibration as these can miscalibrate if unchecked).

## Losses & Regularization (noise + calibration)
- **CrossEntropy with class weights** from train distribution.
- **Label smoothing** (ε≈0.05–0.1) to soften impact of noisy/conflicting labels and improve calibration.
- **Weight decay** (~1e-4) and early stopping on val F1/ECE to limit overfit.
- **Optimizer:** AdamW (lr≈1e-3 with cosine/step decay; batch size ≈64; AMP if available).

## Calibration Strategy (Guo et al., ICML 2017)
- Fit **temperature scaling** on val logits after training; reuse on test.
- Report **ECE + Brier score + reliability diagrams** for each candidate model.
- Optional: shallow TTA (horizontal flip), average logits, then re-apply temperature if TTA used at inference.

## Evaluation Protocol
- **Primary metrics:** Accuracy, macro-F1, per-class precision/recall.
- **Calibration:** ECE, Brier, reliability diagram.
- **Threshold tuning:** On val only if operational needs favor recall (e.g., damage detection sensitivity).
- **Slices for robustness:** Metrics by brightness/contrast deciles, by lon/lat bins (to see spatial shift sensitivity), and on the “conflicting coords” subset.
- **Confusion matrix & score histograms:** To spot systematic errors and overconfident mistakes.

## Project Roadmap
1. **Data curation:** Run corruption check, de-duplication, conflict summary; document any removals/downweighting.
2. **Baseline:** ResNet-18 (https://www.geeksforgeeks.org/deep-learning/resnet18-from-scratch-using-pytorch/) with weighted sampling + smoothing; train 25–40 epochs; log metrics; fit temperature scaling on val; evaluate on test.
3. **Medium model:** EfficientNet-B0/B2 (https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py) (or ConvNeXt-T) with same pipeline; compare accuracy/F1/ECE/Brier.
4. **Robustness add-ons (if needed):** Slightly stronger color jitter; MixUp/CutMix trials; re-check calibration.
5. **Select best model:** Choose based on balanced accuracy/F1 and calibrated ECE/Brier; prefer the best-calibrated if accuracy is close (per brief’s emphasis on trustworthiness).
6. **Error analysis:** Inspect high-confidence errors via grids; review performance on conflicting coords; report limitations.

## Reproducibility & Reporting (Why: evaluation criteria require clarity)
- Fix seeds; log all hyperparameters, normalization stats, and data filtering decisions.
- Save: model weights, temperature parameter, training/val logs, evaluation scripts, and generated figures (class counts, spatial scatter/map, density plots, conflict highlights, reliability diagrams).
- Keep official val/test untouched; if an internal tuning split is needed, draw only from train and report it.

## Default Answer: Which splits to include in analysis?
Use **all splits** for EDA to reveal domain shift, imbalance, and label noise patterns. For training, restrict to the train split (optionally with an internal holdout) and keep val/test purely for evaluation and calibration fitting.
