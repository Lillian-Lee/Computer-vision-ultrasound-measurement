"""Evaluate trained models on a dataset split and write a full report.

    python -m cvmeasure.evaluate --seg runs/seg/best.pt --reg runs/reg/best.pt \
        --data data/synthetic --split test --out reports/test
    python -m cvmeasure.evaluate --seg runs/seg/best.pt --reg runs/reg/best.pt \
        --data data/synthetic_shift --split test --out reports/shift

Outputs: metrics.json, per_frame.csv, figures/*.png (Bland-Altman, scatter, qualitative
overlays, worst cases) and a markdown table you can paste into a report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cvmeasure import viz
from cvmeasure.data.dataset import LoinUltrasoundDataset
from cvmeasure.measure.measurements import MEASUREMENT_KEYS, MEASUREMENT_LABELS, measure_from_label_map
from cvmeasure.metrics import agreement_stats, dice_iou_per_class
from cvmeasure.train import build_model


def _console_safe(text: str, encoding: str | None = None) -> str:
    """Return text that can be printed by a legacy Windows console encoding."""
    target_encoding = encoding or sys.stdout.encoding or "utf-8"
    try:
        text.encode(target_encoding)
    except (LookupError, UnicodeEncodeError):
        text = text.translate(str.maketrans({"→": "->", "²": "^2", "−": "-"}))
    return text.encode(target_encoding, errors="replace").decode(target_encoding)


def load_checkpoint(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["task"], ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def run_eval(seg_ckpt: str | None, reg_ckpt: str | None, data_root: str, split: str, out_dir: Path,
             device: str = "cpu", image_size: int | None = None, batch_size: int = 16, n_qual: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    seg_model = reg_model = None
    stats = None
    if seg_ckpt:
        seg_model, ck = load_checkpoint(seg_ckpt, device)
        image_size = image_size or ck["cfg"]["data"].get("image_size")
    if reg_ckpt:
        reg_model, ckr = load_checkpoint(reg_ckpt, device)
        image_size = image_size or ckr["cfg"]["data"].get("image_size")
        cand = [Path(reg_ckpt).parent / "target_stats.json", Path(reg_ckpt).parent / "regressor_target_stats.json"]
        stats = json.loads(next(p for p in cand if p.exists()).read_text(encoding="utf-8"))

    ds = LoinUltrasoundDataset(data_root, split, augment=False, image_size=image_size, target_stats=stats)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    rows, qual = [], []
    for b in dl:
        x = b["image"].to(device)
        sp = b["pixel_spacing_mm"].numpy()
        truth = b["targets"].numpy()
        pred_masks = seg_model(x).argmax(1).cpu().numpy() if seg_model is not None else None
        pred_reg = None
        if reg_model is not None:
            mu, sd = np.array(stats["mean"]), np.array(stats["std"])
            pred_reg = reg_model(x).cpu().numpy() * sd + mu
        for i in range(x.shape[0]):
            row = {"id": b["id"][i], "pixel_spacing_mm": float(sp[i])}
            row.update({f"true_{k}": float(truth[i, j]) for j, k in enumerate(MEASUREMENT_KEYS)})
            if pred_masks is not None:
                t = b["mask"][i].numpy()
                row.update(dice_iou_per_class(pred_masks[i], t))
                m = measure_from_label_map(pred_masks[i], float(sp[i]), clean=True)
                row.update({f"seg_{k}": m[k] for k in MEASUREMENT_KEYS})
                row.update({f"qc_{k}": v for k, v in m["qc"].items()})
                qual.append((b["id"][i], (x[i, 0].cpu().numpy() * 255).astype(np.uint8), t.astype(np.uint8),
                             pred_masks[i].astype(np.uint8), float(sp[i]), row["dice_c1"]))
            if pred_reg is not None:
                row.update({f"reg_{k}": float(pred_reg[i, j]) for j, k in enumerate(MEASUREMENT_KEYS)})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_frame.csv", index=False)

    metrics: dict = {"n_frames": len(df), "data_root": str(data_root), "split": split}
    if seg_model is not None:
        metrics["segmentation"] = {
            "dice_muscle": float(df.dice_c1.mean()), "dice_fat": float(df.dice_c2.mean()),
            "iou_muscle": float(df.iou_c1.mean()), "iou_fat": float(df.iou_c2.mean()),
            "dice_muscle_p05": float(df.dice_c1.quantile(0.05)),
            "frames_flagged_border": int(df.qc_muscle_touches_border.sum()),
            "frames_no_muscle": int(df.qc_no_muscle.sum()),
        }
        metrics["seg_measurements"] = {k: agreement_stats(df[f"seg_{k}"], df[f"true_{k}"]) for k in MEASUREMENT_KEYS}
    if reg_model is not None:
        metrics["reg_measurements"] = {k: agreement_stats(df[f"reg_{k}"], df[f"true_{k}"]) for k in MEASUREMENT_KEYS}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ---- figures ---------------------------------------------------------------------
    methods = [m for m, ok in (("seg", seg_model is not None), ("reg", reg_model is not None)) if ok]
    viz.agreement_panel(df, methods, fig_dir / "agreement.png")
    viz.bland_altman_panel(df, methods, fig_dir / "bland_altman.png")
    if seg_model is not None and qual:
        qual_sorted = sorted(qual, key=lambda q: q[-1])
        qual_sorted = [(f, im / 255.0, t, p, sp_, d) for f, im, t, p, sp_, d in qual_sorted]
        viz.qualitative_grid(qual_sorted[len(qual_sorted) // 2: len(qual_sorted) // 2 + n_qual], fig_dir / "qualitative_best.png", "Typical predictions (median Dice)")
        viz.qualitative_grid(qual_sorted[:n_qual], fig_dir / "qualitative_worst.png", "Hardest frames (lowest Dice)")
        viz.error_vs_condition(df, ds.df, fig_dir / "error_vs_condition.png")

    # ---- markdown table ------------------------------------------------------------
    lines = [f"### {data_root} / {split} (n={len(df)})", ""]
    if seg_model is not None:
        s = metrics["segmentation"]
        segmentation_summary = (
            f"Segmentation: Dice muscle **{s['dice_muscle']:.3f}** "
            f"(5th pct {s['dice_muscle_p05']:.3f}), Dice fat **{s['dice_fat']:.3f}**, "
            f"frames QC-flagged (muscle touches border): {s['frames_flagged_border']}"
        )
        lines += [segmentation_summary, ""]
    lines += ["| Measurement | Method | MAE | RMSE | Bias | R² | CCC | LoA (95%) |", "|---|---|---|---|---|---|---|---|"]
    for k in MEASUREMENT_KEYS:
        for m in methods:
            a = metrics[f"{m}_measurements"][k]
            if "mae" not in a:
                continue
            unit = "mm²" if k == "ema_mm2" else "mm"
            lines.append(f"| {MEASUREMENT_LABELS[k]} | {'U-Net → measure' if m == 'seg' else 'CNN regression'} | "
                         f"{a['mae']:.2f} {unit} | {a['rmse']:.2f} | {a['bias']:+.2f} | {a['r2']:.3f} | {a['ccc']:.3f} | "
                         f"[{a['loa_low']:+.2f}, {a['loa_high']:+.2f}] |")
    (out_dir / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(_console_safe("\n".join(lines)))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=None)
    ap.add_argument("--reg", default=None)
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="reports/test")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    run_eval(a.seg, a.reg, a.data, a.split, Path(a.out), a.device)


if __name__ == "__main__":
    main()
