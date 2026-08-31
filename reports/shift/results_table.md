### data/synthetic_shift / test (n=241)

Segmentation: Dice muscle **0.992** (5th pct 0.987), Dice fat **0.959**, frames QC-flagged (muscle touches border): 10

| Measurement | Method | MAE | RMSE | Bias | R² | CCC | LoA (95%) |
|---|---|---|---|---|---|---|---|
| Eye muscle area (mm²) | U-Net → measure | 13.21 mm² | 16.32 | -9.99 | 0.998 | 0.999 | [-35.32, +15.34] |
| Eye muscle area (mm²) | CNN regression | 80.62 mm² | 96.28 | -74.65 | 0.929 | 0.962 | [-194.08, +44.79] |
| Eye muscle depth (mm) | U-Net → measure | 0.15 mm | 0.27 | +0.04 | 0.997 | 0.998 | [-0.47, +0.56] |
| Eye muscle depth (mm) | CNN regression | 0.88 mm | 1.14 | -0.66 | 0.940 | 0.968 | [-2.48, +1.16] |
| Eye muscle width (mm) | U-Net → measure | 0.69 mm | 0.87 | -0.31 | 0.984 | 0.992 | [-1.90, +1.28] |
| Eye muscle width (mm) | CNN regression | 2.58 mm | 3.12 | -2.20 | 0.800 | 0.895 | [-6.55, +2.16] |
| Fat depth C (mm) | U-Net → measure | 0.43 mm | 0.69 | +0.34 | 0.877 | 0.937 | [-0.83, +1.51] |
| Fat depth C (mm) | CNN regression | 0.51 mm | 0.65 | +0.17 | 0.890 | 0.935 | [-1.06, +1.40] |
