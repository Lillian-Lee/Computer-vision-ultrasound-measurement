"""Import real annotated ultrasound frames into the cvmeasure dataset layout.

Supported annotation formats
----------------------------
* **LabelMe** JSON (one file per image, polygon shapes) - the tool most small research
  teams use for medical/vet ultrasound because polygons are quick to trace.
* **COCO** instance JSON (single file) - the export format of CVAT, Label Studio,
  Roboflow, etc.

Label mapping: any label containing "muscle" / "eye" / "loin" / "ld" -> class 1;
"fat" / "subcut" -> class 2. Override with ``label_map`` if your annotators used other names.

Pixel spacing (mm/px)
---------------------
Ultrasound exports (PNG/JPG/DICOM screenshots) usually do **not** carry calibration.
Provide it via ``pixel_spacing_mm`` (constant for one scanner preset) or a CSV column
``pixel_spacing_mm`` keyed by image id. Without it the pipeline still trains and reports
Dice, but every mm/mm^2 figure would be wrong, so the importer refuses to guess.

Everything written here is compatible with :class:`cvmeasure.data.dataset.LoinUltrasoundDataset`.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cvmeasure.measure.measurements import MEASUREMENT_KEYS, measure_from_label_map

DEFAULT_LABEL_MAP = {
    "muscle": 1, "eye": 1, "loin": 1, "ld": 1, "longissimus": 1, "ema": 1,
    "fat": 2, "subcut": 2, "subcutaneous": 2,
}


def _class_of(label: str, label_map: dict) -> int | None:
    low = label.lower()
    for key, cls in label_map.items():
        if key in low:
            return cls
    return None


def polygons_to_label_map(shape_hw: tuple[int, int], polygons: Iterable[tuple[int, np.ndarray]]) -> np.ndarray:
    """polygons: iterable of (class_id, Nx2 float array of x,y). Muscle is drawn last so it wins overlaps."""
    lab = np.zeros(shape_hw, np.uint8)
    for cls, pts in sorted(polygons, key=lambda t: t[0], reverse=True):   # 2 (fat) first, then 1 (muscle)
        cv2.fillPoly(lab, [np.round(pts).astype(np.int32)], int(cls))
    return lab


def import_labelme(image_dir: str | Path, out_root: str | Path, pixel_spacing_mm: float | dict | None,
                   label_map: dict = DEFAULT_LABEL_MAP, animal_id_fn=lambda stem: stem.split("_")[0],
                   split_fn=None, seed: int = 0) -> pd.DataFrame:
    """Read every ``<image_dir>/*.json`` (LabelMe) with its image and write a dataset."""
    image_dir, out_root = Path(image_dir), Path(out_root)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for js in sorted(image_dir.glob("*.json")):
        d = json.loads(js.read_text(encoding="utf-8"))
        img_path = image_dir / d.get("imagePath", js.stem + ".png")
        if not img_path.exists():
            cands = list(image_dir.glob(js.stem + ".*"))
            img_path = next((c for c in cands if c.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif")), None)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        polys = []
        for sh in d.get("shapes", []):
            cls = _class_of(sh.get("label", ""), label_map)
            if cls is None or sh.get("shape_type", "polygon") != "polygon":
                continue
            polys.append((cls, np.asarray(sh["points"], np.float32)))
        lab = polygons_to_label_map(img.shape, polys)
        rows.append(_write_frame(out_root, js.stem, img, lab, pixel_spacing_mm, animal_id_fn))
    return _finish(rows, out_root, split_fn, seed)


def import_coco(coco_json: str | Path, image_dir: str | Path, out_root: str | Path,
                pixel_spacing_mm: float | dict | None, label_map: dict = DEFAULT_LABEL_MAP,
                animal_id_fn=lambda stem: stem.split("_")[0], split_fn=None, seed: int = 0) -> pd.DataFrame:
    coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    image_dir, out_root = Path(image_dir), Path(out_root)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)
    cat_cls = {c["id"]: _class_of(c["name"], label_map) for c in coco["categories"]}
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    rows = []
    for im in coco["images"]:
        p = image_dir / im["file_name"]
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        polys = []
        for a in anns_by_img.get(im["id"], []):
            cls = cat_cls.get(a["category_id"])
            if cls is None:
                continue
            for seg in a.get("segmentation", []):
                if isinstance(seg, list) and len(seg) >= 6:
                    polys.append((cls, np.asarray(seg, np.float32).reshape(-1, 2)))
        lab = polygons_to_label_map(img.shape, polys)
        rows.append(_write_frame(out_root, Path(im["file_name"]).stem, img, lab, pixel_spacing_mm, animal_id_fn))
    return _finish(rows, out_root, split_fn, seed)


def _write_frame(out_root, stem, img, lab, pixel_spacing_mm, animal_id_fn):
    sp = pixel_spacing_mm.get(stem) if isinstance(pixel_spacing_mm, dict) else pixel_spacing_mm
    cv2.imwrite(str(out_root / "images" / f"{stem}.png"), img)
    cv2.imwrite(str(out_root / "masks" / f"{stem}.png"), lab)
    row = {"id": stem, "animal_id": animal_id_fn(stem), "pixel_spacing_mm": sp}
    if sp is not None:
        m = measure_from_label_map(lab, sp, clean=False)
        row.update({k: m[k] for k in MEASUREMENT_KEYS})
    else:
        row.update({k: float("nan") for k in MEASUREMENT_KEYS})
    return row


def _finish(rows, out_root, split_fn, seed):
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No annotated frames found - check paths and label names.")
    if split_fn is None:
        rng = np.random.default_rng(seed)
        animals = list(df.animal_id.unique())
        rng.shuffle(animals)
        n = len(animals)
        split_of = {a: ("train" if i < 0.7 * n else "val" if i < 0.85 * n else "test") for i, a in enumerate(animals)}
        df["split"] = df.animal_id.map(split_of)
    else:
        df["split"] = df.id.map(split_fn)
    df.to_csv(out_root / "metadata.csv", index=False)
    return df
