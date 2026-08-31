# 01 · Domain research: why eye-muscle ultrasound, and what the model must output

Background reading I did before writing any code. I kept it short and practical: the point
was to make sure the model outputs the numbers a sheep-breeding programme actually uses,
rather than stopping at a segmentation mask.

## 1. What is measured on a live sheep, and why

Meat-breeding programmes in New Zealand and Australia select rams partly on **carcass
traits measured on the live animal by ultrasound**. A B-mode scanner (linear probe,
typically 3.5-7.5 MHz) is placed transversely over the loin at the **12th/13th rib**, and
the technician traces or clicks on the frozen frame to record:

| Symbol | Trait | What it is |
|---|---|---|
| **A** | Eye muscle width (EMW) | maximum lateral extent of the *longissimus dorsi* ("eye muscle") cross-section |
| **B** | Eye muscle depth (EMD) | maximum vertical extent of the eye muscle |
| **C** | Fat depth C | subcutaneous fat thickness over the eye muscle (over its deepest point) |
| **EMA** | Eye muscle area | traced cross-sectional area (cm²), strongly correlated with lean meat yield |
| **GR** | GR tissue depth | total tissue depth 110 mm from the midline over the 12th rib - a separate site, *not* in the same frame |

NZ scanning services (e.g. Stockscan, WRS Livestock Scanning) describe exactly the A/B/C
convention above; the numbers are submitted to the national genetic evaluation (SIL /
nProve) where they contribute to muscling and fat breeding values. Eye muscle area can vary
by ~30 % within a flock before selection, which is why the trait is worth measuring. (UK
services such as Signet scan at the 3rd lumbar vertebra instead and take three fat readings
at 1 cm intervals - site conventions differ between countries, so a model has to be
validated per protocol.)

The ranges I used for the synthetic priors - EMD 22-38 mm, EMW 48-72 mm, fat C 1.5-8 mm,
EMA roughly 8-22 cm² (`AnatomyPriors` in `src/cvmeasure/synth/generator.py`) - are plausible
values for lambs at scanning weight, pieced together from the scanning-service pages and papers
below. They are hand-set ranges, not a measured population.

## 2. Why automate it

* Manual tracing is slow (a technician can scan a few hundred animals a day) and has
  operator effects - repeatability between technicians is a known limitation of ultrasound
  carcass traits.
* MEQ Solutions has commercialised AI measurement of carcass/meat-quality traits in
  **cattle** (MEQ Camera / Probe at the plant, **MEQ Live** on the live animal via
  ultrasound + AI). The same idea for **sheep** - automated eye-muscle and fat measurement
  from ultrasound frames - does not seem to exist as a product yet, and NZ has a large
  ram-breeding sector that already scans routinely. That gap is what got me interested.
* Recent papers show the direction of travel: MPG-SwinUMamba (Animals, 2025) segments live
  sheep eye-muscle ultrasound (710 images from 230 sheep; Dice 0.955, automated EMA r = 0.96
  against experts, MAPE 4 %); UNet++ has been used for cattle rib-eye ultrasound (Computers
  & Electronics in Agriculture, 2022); a dual-branch model predicts sheep IMF from the
  ultrasound image plus eye-muscle depth/area and fat depth (Scientific Reports, 2025,
  1 728 sheep, R² 0.83) - i.e. the measurements this project produces are also inputs to
  meat-quality prediction.

## 3. Design decisions that follow from the domain

1. **Segment, then measure** (primary) rather than only regress numbers. Technicians and
   geneticists need to *see* the traced muscle to trust it, and QC flags (muscle cut off,
   fragmented mask) must be possible. A direct-regression CNN is kept as a baseline for
   comparison.
2. **Millimetres are first-class.** Every function takes `pixel_spacing_mm`; the model
   never learns a hidden mm/px. Changing scanner depth setting must not silently break
   measurements.
3. **Split by animal, not by frame.** Several frames per animal are collected in practice;
   splitting by frame would leak animal identity into the test set and inflate the results.
4. **Method-comparison statistics.** Report bias, MAE, Lin's CCC and Bland-Altman limits of
   agreement against the reference - the way animal-science method papers do - not only
   Dice.
5. **Domain-shift test set.** Scanner gain, speckle, shadowing and probe contact vary
   between operators/days; a harder held-out set is generated to measure robustness.
6. **Physically motivated synthetic data** because I have no annotated real frames, with a
   LabelMe/COCO import path so the same code trains on real data the day I get some.

## 4. Sources consulted

* Stockscan NZ - Eye muscle scanning: https://www.stockscan.co.nz/services/eye-muscle-scanning/
* WRS Livestock Scanning - Muscle scanning: http://www.wrslivestockscanning.co.nz/muscle-scanning-g-219.html
* Signet Breeding (UK) - Ultrasound scanning service: https://signetdata.com/technical/sheep-recording/ultrasound-scanning/
* MEQ Solutions - MEQ Live: https://meqsolutions.com/meq-live ; company/funding overview: https://www.businessnewsaustralia.com/articles/meq-solutions-raises-23m.html
* MPG-SwinUMamba: high-precision segmentation and automated measurement of eye muscle area in live sheep (Animals, 2025): https://doi.org/10.3390/ani15243509
* Automatic segmentation of cattle rib-eye area in ultrasound images using UNet++ (Comput. Electron. Agric., 2022): https://www.sciencedirect.com/science/article/abs/pii/S0168169922001351
* Dual-branch multi-modal deep learning for sheep intramuscular fat (Sci. Rep., 2025): https://www.nature.com/articles/s41598-025-32208-2
* Ultrasound image segmentation with shape priors for cattle rib-eye area (classic reference): https://www.researchgate.net/publication/6289717
