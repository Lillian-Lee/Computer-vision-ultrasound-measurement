# 03 · Results

All numbers are on **synthetic** data (see `docs/02-model-card.md` for what that does and
does not tell you). Reference values are the exact geometric measurements of the ground-truth
masks. Raw outputs: `reports/<set>/metrics.json`, `per_frame.csv`, `figures/`.
Everything was trained on CPU (2 cores): U-Net 12 epochs ≈ 35 min, regressor 20 epochs ≈ 20 min.

## 1. Segmentation quality

| Test set | n frames | Dice eye muscle (mean / 5th pct) | Dice fat | IoU muscle |
|---|---|---|---|---|
| In-distribution test split | 264 | **0.994** / 0.991 | 0.961 | 0.988 |
| Domain-shift set (harsher scanner settings) | 241 | **0.992** / 0.987 | 0.959 | 0.985 |
| Low-data ablation (10 % of training animals), test | 264 | 0.980 / 0.968 | 0.905 | – |
| Low-data ablation, domain-shift | 241 | 0.971 / 0.950 | 0.897 | – |

The muscle contour is recovered almost perfectly even where its lateral walls are invisible
(incidence-angle effect) or partly hidden by a rib shadow; the fat layer is harder because
its lower boundary coincides with the faint muscle roof and its lateral lobes are ambiguous
by construction. Training curves: `reports/figures/training_curves.png`.

## 2. Measurement agreement (segment → measure vs. direct regression)

In-distribution test split (n = 264):

| Measurement | Method | MAE | RMSE | Bias | R² | CCC | 95 % LoA |
|---|---|---|---|---|---|---|---|
| Eye muscle area (mm²) | **U-Net → measure** | **6.9** | 8.8 | +0.1 | 0.999 | 1.000 | [−17.1, +17.2] |
| Eye muscle area (mm²) | CNN regression | 69.0 | 83.0 | −64.6 | 0.954 | 0.976 | [−166.9, +37.8] |
| Eye muscle depth (mm) | **U-Net → measure** | **0.13** | 0.24 | +0.07 | 0.998 | 0.999 | [−0.39, +0.52] |
| Eye muscle depth (mm) | CNN regression | 0.84 | 1.10 | −0.56 | 0.957 | 0.977 | [−2.42, +1.30] |
| Eye muscle width (mm) | **U-Net → measure** | **0.58** | 0.78 | +0.15 | 0.987 | 0.993 | [−1.34, +1.65] |
| Eye muscle width (mm) | CNN regression | 2.11 | 2.63 | −1.69 | 0.847 | 0.920 | [−5.65, +2.27] |
| Fat depth C (mm) | U-Net → measure | 0.41 | 0.69 | +0.33 | 0.853 | 0.922 | [−0.86, +1.52] |
| Fat depth C (mm) | CNN regression | 0.41 | 0.55 | +0.14 | 0.905 | 0.946 | [−0.91, +1.19] |

Domain-shift set (n = 241):

| Measurement | Method | MAE | RMSE | Bias | R² | CCC | 95 % LoA |
|---|---|---|---|---|---|---|---|
| Eye muscle area (mm²) | **U-Net → measure** | **13.2** | 16.3 | −10.0 | 0.998 | 0.999 | [−35.3, +15.3] |
| Eye muscle area (mm²) | CNN regression | 80.6 | 96.3 | −74.7 | 0.929 | 0.962 | [−194.1, +44.8] |
| Eye muscle depth (mm) | **U-Net → measure** | **0.15** | 0.27 | +0.04 | 0.997 | 0.998 | [−0.47, +0.56] |
| Eye muscle depth (mm) | CNN regression | 0.88 | 1.14 | −0.66 | 0.940 | 0.968 | [−2.48, +1.16] |
| Eye muscle width (mm) | **U-Net → measure** | **0.69** | 0.87 | −0.31 | 0.984 | 0.992 | [−1.90, +1.28] |
| Eye muscle width (mm) | CNN regression | 2.58 | 3.12 | −2.20 | 0.800 | 0.895 | [−6.55, +2.16] |
| Fat depth C (mm) | U-Net → measure | 0.43 | 0.69 | +0.34 | 0.877 | 0.937 | [−0.83, +1.51] |
| Fat depth C (mm) | CNN regression | 0.51 | 0.65 | +0.17 | 0.890 | 0.935 | [−1.06, +1.40] |

Figures: `reports/test/figures/agreement.png`, `bland_altman.png`, `qualitative_best.png`,
`qualitative_worst.png`, `error_vs_condition.png` (and the same under `reports/shift/`).

## 3. What the numbers say

Segment-then-measure is better on every muscle trait by a factor of 4-10 and degrades
gracefully under domain shift: EMA MAE goes from 6.9 to 13.2 mm², which is still under 1 % of
a typical 1 400 mm² eye muscle, and the EMD error stays around a third of a pixel. Eye-muscle
depth is quantised to the 0.42 mm pixel pitch - visible as horizontal bands in the
Bland-Altman plot - so sub-pixel contour fitting is an obvious next step.

The regression CNN shows the usual shrinkage toward the mean: a negative bias that grows with
animal size, R² between 0.85 and 0.96. More epochs, a bigger backbone or a linear calibration
afterwards would help, but the structural problems remain - no contour to audit, no QC flag,
and an implicitly memorised training mm/px. The one trait where it is competitive is fat
depth C, because the geometric route carries a systematic +0.3 mm over-call there.

Fat depth C is the hard trait for both routes (CCC 0.92-0.95). The predicted fat/muscle
boundary sits about one pixel too deep on average, i.e. the bright muscle-roof fascia gets
attributed to fat. That is a systematic and correctable bias, but it is also exactly the kind
of thing I cannot settle on synthetic data, because the simulator is what decides where "fat
ends". It needs real technician traces.

The QC flags behave: the "muscle touches border" flag agreed with ground truth on 98.5 % of
test frames and 97.5 % of domain-shift frames, and every truly out-of-field frame in the test
split was caught. In practice a frame that fails QC would be re-scanned rather than averaged
into a breeding value.

On data efficiency: with 10 % of the training animals (63 animals, 126 frames) Dice(muscle) is
still 0.980 in-distribution and 0.971 under shift, but the fat-C error more than doubles (MAE
1.1 mm). That is a useful hint for how many real annotated frames to budget for - the muscle
contour is learnt from little data, the fat boundary needs more.

Failure modes (`qualitative_worst.png`, `error_vs_condition.png`) are the ones I expected:
a rib shadow cutting the muscle floor, contact loss on the side where the muscle wall is
already faint, and thick fat under a weak roof fascia. None of them produce gross failures on
synthetic data; on real data these are the frames I would collect first.

## 4. Next steps

See `docs/04-design-notes.md`. In short: annotate a few hundred real frames from at least a
hundred animals, fine-tune from the synthetic-pretrained U-Net, and validate against technician
A/B/C values and then carcass/CT data with CCC and Bland-Altman rather than Dice alone.
