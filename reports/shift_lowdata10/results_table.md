### data/synthetic_shift / test (n=241)

Segmentation: Dice muscle **0.971** (5th pct 0.950), Dice fat **0.897**, frames QC-flagged (muscle touches border): 0

| Measurement | Method | MAE | RMSE | Bias | R² | CCC | LoA (95%) |
|---|---|---|---|---|---|---|---|
| Eye muscle area (mm²) | U-Net → measure | 46.14 mm² | 118.54 | -36.54 | 0.892 | 0.947 | [-258.02, +184.94] |
| Eye muscle depth (mm) | U-Net → measure | 0.35 mm | 0.85 | +0.19 | 0.967 | 0.983 | [-1.43, +1.81] |
| Eye muscle width (mm) | U-Net → measure | 2.57 mm | 3.95 | -1.39 | 0.679 | 0.846 | [-8.66, +5.88] |
| Fat depth C (mm) | U-Net → measure | 1.08 mm | 1.32 | +1.03 | 0.540 | 0.796 | [-0.59, +2.65] |
