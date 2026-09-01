# Reproducible LabelMe annotation workflow

This walkthrough shows how polygon annotations move from LabelMe into the training dataset. It uses generated images only. It does not represent manual annotation of real livestock ultrasound.

## 1. Create the reproducibility fixture

From the repository root, install the project and run:

```bash
pip install -e ".[dev]"
python scripts/labelme_annotation_walkthrough.py
```

The command deterministically creates three synthetic frames and LabelMe-compatible JSON files under `artifacts/labelme_walkthrough/labelme_raw/`. Each JSON file contains `eye_muscle` and `subcut_fat` polygons and an explicit `synthetic_example` flag.

It then imports those files into the standard dataset layout:

```text
artifacts/labelme_walkthrough/
  labelme_raw/                 source PNG and LabelMe JSON pairs
  imported_dataset/
    images/                    greyscale training images
    masks/                     class-index masks: 0 background, 1 muscle, 2 fat
    metadata.csv               calibration, measurements, animal IDs and splits
  annotation_overlay.png       visual check of the imported polygons
```

The three animal IDs are deliberately assigned to different train, validation and test splits. This makes the no-leakage rule easy to inspect.

## 2. Inspect or edit an annotation in LabelMe

Install and launch LabelMe, then open the generated directory:

```bash
pip install labelme
labelme artifacts/labelme_walkthrough/labelme_raw
```

Use polygon shapes and one of these labels:

| Label | Imported class |
|---|---:|
| `eye_muscle` | 1 |
| `subcut_fat` | 2 |

Other labels containing `muscle`, `eye`, `loin`, `ld`, `fat` or `subcut` are also recognised. Save one JSON file beside each image and keep image names in the form `<animal>_<frame>.png`. The prefix before the first underscore becomes the animal ID used for grouped splitting.

## 3. Supply calibration rather than guessing it

The example passes `pixel_spacing_mm=0.42`, meaning 0.42 mm per pixel. For real data, obtain this value from the scanner export or a documented depth calibration. Do not infer it from this synthetic fixture.

For one scanner preset, pass a constant:

```python
from cvmeasure.data.annotation import import_labelme

import_labelme("raw_frames", "data/annotated", pixel_spacing_mm=0.10)
```

For mixed presets, pass a dictionary keyed by image stem:

```python
spacing = {"animal01_frame01": 0.10, "animal02_frame01": 0.12}
import_labelme("raw_frames", "data/annotated", pixel_spacing_mm=spacing)
```

The importer writes physical measurements only when calibration is supplied. It does not guess missing spacing.

## 4. Verify the import

Check all four outputs before training:

1. Open `annotation_overlay.png` and confirm that the green muscle and orange fat boundaries follow the intended regions.
2. Open a mask and confirm that its only pixel values are 0, 1 and 2.
3. Inspect `metadata.csv` for the expected animal ID, split and pixel spacing.
4. Confirm that no animal ID occurs in more than one split.

The automated test covers the same contract:

```bash
pytest -q tests/test_labelme_walkthrough.py
```

## 5. Use the imported dataset

Point `data.root` in a training config to `artifacts/labelme_walkthrough/imported_dataset` to exercise the data path. The three-frame fixture is only a format and integration check. It is too small and too synthetic for model training or performance claims.

For real annotation work, add an independent expert review step, document the label guide, track annotator and revision metadata, and measure inter-annotator agreement on a shared subset. Those activities are outside this synthetic demonstration.
