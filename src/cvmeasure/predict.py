"""Batch inference: folder of B-mode frames -> measurements CSV + annotated overlays.

    python -m cvmeasure.predict --seg runs/seg/best.pt --images data/synthetic/images \
        --pixel-spacing 0.42 --out predictions/ --limit 20

The pixel spacing (mm/px) MUST be supplied - it comes from the scanner depth setting and
is what turns pixels into the millimetres that feed a breeding-value analysis. Frames that
fail QC (muscle cut off by the field of view / no muscle found) are written with a flag
rather than silently dropped, so a technician can re-scan the animal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from cvmeasure.evaluate import load_checkpoint
from cvmeasure.measure.measurements import MEASUREMENT_KEYS, measure_from_label_map


@torch.no_grad()
def predict_folder(seg_ckpt: str, images: str | Path, pixel_spacing_mm: float, out: str | Path,
                   image_size: int | None = None, limit: int | None = None, device: str = "cpu",
                   save_overlays: bool = True) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    from cvmeasure.viz import overlay
    out = Path(out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    model, ck = load_checkpoint(seg_ckpt, device)
    image_size = image_size or ck["cfg"]["data"].get("image_size")
    paths = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif"))
    if limit:
        paths = paths[:limit]
    rows = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        sp = pixel_spacing_mm
        if image_size and img.shape[0] != image_size:
            sp = sp * img.shape[0] / image_size
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)
        probs = model(x).softmax(1)[0]
        mask = probs.argmax(0).cpu().numpy().astype(np.uint8)
        conf = float(probs.max(0).values[mask == 1].mean()) if (mask == 1).any() else float("nan")
        m = measure_from_label_map(mask, sp, clean=True)
        row = {"image": p.name, "pixel_spacing_mm": sp, "mean_muscle_confidence": conf}
        row.update({k: m[k] for k in MEASUREMENT_KEYS})
        row.update({f"qc_{k}": v for k, v in m["qc"].items()})
        row["qc_pass"] = not (m["qc"]["no_muscle"] or m["qc"]["muscle_touches_border"])
        rows.append(row)
        cv2.imwrite(str(out / "overlays" / f"{p.stem}_mask.png"), mask * 120)
        if save_overlays:
            fig, ax = plt.subplots(figsize=(4, 4))
            overlay(ax, img / 255.0, mask, sp, p.name + ("" if row["qc_pass"] else "  [QC FAIL]"))
            fig.savefig(out / "overlays" / f"{p.stem}_overlay.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(out / "measurements.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--pixel-spacing", type=float, required=True, help="mm per pixel of the input frames")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    df = predict_folder(a.seg, a.images, a.pixel_spacing, a.out, limit=a.limit, device=a.device)
    print(df.head(10).to_string(index=False))
    print(f"\n{len(df)} frames -> {a.out}/measurements.csv ; QC pass rate {df.qc_pass.mean():.1%}")


if __name__ == "__main__":
    main()
