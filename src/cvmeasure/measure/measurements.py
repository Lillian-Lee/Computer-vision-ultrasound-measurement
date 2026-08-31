"""Convert a segmentation mask into the carcass-trait measurements a scanning technician records.

Measurements (all in mm / mm^2, computed with the frame's pixel spacing):

* ``ema_mm2``  - eye muscle area (longissimus dorsi cross-section area)
* ``emd_mm``   - eye muscle depth: maximum vertical (axial) extent of the muscle ("B")
* ``emw_mm``   - eye muscle width: maximum lateral extent of the muscle ("A")
* ``fat_c_mm`` - subcutaneous fat depth over the deepest point of the eye muscle ("C").
                 Measured as the fat thickness in the column where the muscle is deepest,
                 which is the operational definition technicians follow on-screen.

The same function is applied to ground-truth masks and to predicted masks, so any
geometric bias in the extraction cancels in the comparison and the reported error is
attributable to segmentation quality alone.

QC flags are returned so downstream code can reject frames where the muscle is cut off by
the field of view or the mask is fragmented (a real-world requirement: a bad frame should
be re-scanned, not silently averaged into a breeding value).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, label

MEASUREMENT_KEYS = ["ema_mm2", "emd_mm", "emw_mm", "fat_c_mm"]
MEASUREMENT_LABELS = {
    "ema_mm2": "Eye muscle area (mm²)",
    "emd_mm": "Eye muscle depth (mm)",
    "emw_mm": "Eye muscle width (mm)",
    "fat_c_mm": "Fat depth C (mm)",
}


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (post-processing for predicted masks)."""
    lab, n = label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def clean_muscle_mask(mask: np.ndarray) -> np.ndarray:
    return binary_fill_holes(largest_component(mask.astype(bool)))


def measure_from_masks(muscle: np.ndarray, fat: np.ndarray, pixel_spacing_mm: float,
                       clean: bool = False) -> dict:
    """Compute EMA / EMD / EMW / fat-C from binary masks. Returns NaNs + flags if no muscle."""
    muscle = muscle.astype(bool)
    fat = fat.astype(bool)
    if clean:
        muscle = clean_muscle_mask(muscle)
    sp = pixel_spacing_mm
    H, W = muscle.shape
    out = {k: float("nan") for k in MEASUREMENT_KEYS}
    flags = {"no_muscle": False, "muscle_touches_border": False, "n_components": 0}
    if not muscle.any():
        flags["no_muscle"] = True
        out["qc"] = flags
        return out

    _, n_comp = label(muscle)
    flags["n_components"] = int(n_comp)
    ys, xs = np.where(muscle)
    flags["muscle_touches_border"] = bool(xs.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1)

    out["ema_mm2"] = float(muscle.sum()) * sp * sp
    out["emw_mm"] = float(xs.max() - xs.min() + 1) * sp
    # per-column vertical extent -> depth at deepest column
    col_any = muscle.any(axis=0)
    cols = np.where(col_any)[0]
    top = np.argmax(muscle, axis=0)                          # first True row per column
    bottom = H - 1 - np.argmax(muscle[::-1, :], axis=0)      # last True row per column
    depth_px = np.where(col_any, bottom - top + 1, 0)
    c_star = int(cols[np.argmax(depth_px[cols])])
    out["emd_mm"] = float(depth_px[c_star]) * sp
    # fat depth C: fat pixels directly above the muscle in the deepest column
    # (average over a 5-column window for robustness to single-column noise)
    win = range(max(0, c_star - 2), min(W, c_star + 3))
    fat_thick = []
    for c in win:
        if not col_any[c]:
            continue
        above = fat[: top[c], c]
        fat_thick.append(above.sum())
    out["fat_c_mm"] = float(np.mean(fat_thick)) * sp if fat_thick else float("nan")
    out["deepest_column_px"] = c_star
    out["qc"] = flags
    return out


def measure_from_label_map(label_map: np.ndarray, pixel_spacing_mm: float, clean: bool = True,
                           muscle_id: int = 1, fat_id: int = 2) -> dict:
    return measure_from_masks(label_map == muscle_id, label_map == fat_id, pixel_spacing_mm, clean=clean)
