"""Validation metrics for segmentation quality and measurement agreement.

Measurement agreement is reported the way method-comparison studies in animal science do:
bias, MAE, RMSE, R^2, Lin's concordance correlation coefficient (CCC) and Bland-Altman
limits of agreement. CCC is the statistic to look at when the question is "can this
measurement replace the technician's?" because it penalises both scatter and systematic
offset, unlike Pearson r.
"""
from __future__ import annotations

import numpy as np


def dice_iou_per_class(pred: np.ndarray, target: np.ndarray, n_classes: int = 3, eps: float = 1e-7) -> dict:
    out = {}
    for c in range(1, n_classes):
        p, t = pred == c, target == c
        inter = np.logical_and(p, t).sum()
        ps, ts = p.sum(), t.sum()
        out[f"dice_c{c}"] = float((2 * inter + eps) / (ps + ts + eps))
        out[f"iou_c{c}"] = float((inter + eps) / (ps + ts - inter + eps))
    return out


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    return float(2 * cov / (vx + vy + (mx - my) ** 2 + 1e-12))


def agreement_stats(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Method-comparison statistics between predicted and reference measurements."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ok = ~(np.isnan(pred) | np.isnan(truth))
    n_bad = int((~ok).sum())
    p, t = pred[ok], truth[ok]
    if p.size < 3:
        return {"n": int(p.size), "n_failed": n_bad}
    diff = p - t
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    return {
        "n": int(p.size),
        "n_failed": n_bad,
        "bias": float(diff.mean()),
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "mape_pct": float((np.abs(diff) / np.clip(np.abs(t), 1e-6, None)).mean() * 100),
        "r2": float(1 - ss_res / (ss_tot + 1e-12)),
        "pearson_r": float(np.corrcoef(p, t)[0, 1]),
        "ccc": lins_ccc(p, t),
        "loa_low": float(diff.mean() - 1.96 * diff.std(ddof=1)),
        "loa_high": float(diff.mean() + 1.96 * diff.std(ddof=1)),
    }
