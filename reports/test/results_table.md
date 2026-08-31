### data/synthetic / test (n=264)

Segmentation: Dice muscle **0.994** (5th pct 0.991), Dice fat **0.961**, frames QC-flagged (muscle touches border): 15

| Measurement | Method | MAE | RMSE | Bias | R² | CCC | LoA (95%) |
|---|---|---|---|---|---|---|---|
| Eye muscle area (mm²) | U-Net → measure | 6.87 mm² | 8.75 | +0.05 | 0.999 | 1.000 | [-17.14, +17.24] |
| Eye muscle area (mm²) | CNN regression | 68.99 mm² | 82.97 | -64.57 | 0.954 | 0.976 | [-166.89, +37.75] |
| Eye muscle depth (mm) | U-Net → measure | 0.13 mm | 0.24 | +0.07 | 0.998 | 0.999 | [-0.39, +0.52] |
| Eye muscle depth (mm) | CNN regression | 0.84 mm | 1.10 | -0.56 | 0.957 | 0.977 | [-2.42, +1.30] |
| Eye muscle width (mm) | U-Net → measure | 0.58 mm | 0.78 | +0.15 | 0.987 | 0.993 | [-1.34, +1.65] |
| Eye muscle width (mm) | CNN regression | 2.11 mm | 2.63 | -1.69 | 0.847 | 0.920 | [-5.65, +2.27] |
| Fat depth C (mm) | U-Net → measure | 0.41 mm | 0.69 | +0.33 | 0.853 | 0.922 | [-0.86, +1.52] |
| Fat depth C (mm) | CNN regression | 0.41 mm | 0.55 | +0.14 | 0.905 | 0.946 | [-0.91, +1.19] |
