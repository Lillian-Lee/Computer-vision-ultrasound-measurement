# 02 · Model card — cvmeasure v0.1 (sheep eye-muscle case study)

## Intended use
Personal research prototype for automated measurement of sheep eye-muscle traits (eye muscle area,
depth, width; subcutaneous fat depth C) from transverse B-mode ultrasound frames taken at
the 12th/13th rib. Intended to be retrained on real, expert-annotated frames before any
use in genetic evaluation. **Not** a veterinary or commercial product.

## Models
| Model | Task | Input | Output | Params |
|---|---|---|---|---|
| `UNet(base=16, depth=4)` | semantic segmentation | 1×192×192 grayscale, [0,1] | 3-class logits (background / eye muscle / subcutaneous fat) | 1.94 M |
| `MeasurementRegressor` | direct regression baseline | same | 4 standardised targets → mm / mm² | ~0.9 M |
| `ResNetUNet` (optional) | segmentation with ImageNet-pretrained ResNet-18 encoder | same | same | 14 M |

Post-processing for the segmentation route: keep the largest connected muscle component,
fill holes, then compute measurements geometrically with the frame's `pixel_spacing_mm`
(`src/cvmeasure/measure/measurements.py`).

## Training data
Synthetic frames from `cvmeasure.synth` (see `docs/01-domain-research.md` §3 and the module
docstring): 900 simulated animals → 1 795 frames, split **by animal** 70/15/15. A separate
domain-shift test set (120 animals, 241 frames) uses harsher acquisition priors (lower gain,
coarser speckle, stronger attenuation, more shadowing and probe-contact loss).
Augmentation: horizontal flip, ±8 px shift, ±6 % scale (with mm/px corrected and targets
recomputed), gain/gamma jitter, additive noise, lateral drop-out band. No vertical flips.

## Evaluation
Dice / IoU per class; measurement agreement as bias, MAE, RMSE, R², Lin's CCC and
Bland-Altman 95 % limits of agreement, on the in-distribution test split and on the
domain-shift set. Numbers: `docs/03-results.md`, raw JSON under `reports/`.

## Limitations
* **Synthetic-to-real gap.** The simulator reproduces the main physics (speckle, specular
  boundaries with incidence-angle dependence, attenuation, shadowing) but not every real
  effect (reverberation, refraction, heterogeneous fat, wool/gel artefacts, probe curvature).
  Real annotated frames are required; the import path exists (`cvmeasure.data.annotation`).
* **Pixel spacing must be supplied.** Exported ultrasound PNGs rarely carry calibration.
* **Priors are hand-set.** Anatomy ranges are plausible for lambs at scanning weight, not a
  measured population; the model has never seen a very lean, very fat or adult animal.
* **Single frame, single site.** GR site and multi-frame averaging are out of scope.
* **Not validated against carcass data.** Real deployments validate against slaughter
  measurements or CT, not only against a technician's trace.

## Data provenance
No animal data were used. Nothing in this repository is derived from any scanning service,
breeder's records or commercial product; company and product names appear only as context
for the research question.
