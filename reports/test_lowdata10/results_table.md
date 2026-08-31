### data/synthetic / test (n=264)

Segmentation: Dice muscle **0.980** (5th pct 0.968), Dice fat **0.905**, frames QC-flagged (muscle touches border): 0

| Measurement | Method | MAE | RMSE | Bias | R² | CCC | LoA (95%) |
|---|---|---|---|---|---|---|---|
| Eye muscle area (mm²) | U-Net → measure | 21.51 mm² | 27.73 | -8.77 | 0.995 | 0.997 | [-60.43, +42.88] |
| Eye muscle depth (mm) | U-Net → measure | 0.31 mm | 0.44 | +0.25 | 0.993 | 0.997 | [-0.46, +0.96] |
| Eye muscle width (mm) | U-Net → measure | 1.82 mm | 2.25 | -0.67 | 0.888 | 0.946 | [-4.89, +3.54] |
| Fat depth C (mm) | U-Net → measure | 1.09 mm | 1.32 | +1.06 | 0.461 | 0.762 | [-0.49, +2.60] |
