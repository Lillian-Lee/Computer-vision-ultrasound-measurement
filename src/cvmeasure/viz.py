"""Plotting helpers for reports (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cvmeasure.measure.measurements import MEASUREMENT_KEYS, MEASUREMENT_LABELS  # noqa: E402

COLORS = {"seg": "#2563EB", "reg": "#F97316", "ref": "#6B7280", "muscle": "#22C55E", "fat": "#FACC15"}
NAMES = {"seg": "U-Net → measure", "reg": "CNN regression"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def agreement_panel(df: pd.DataFrame, methods: list[str], out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, k in zip(axes, MEASUREMENT_KEYS):
        t = df[f"true_{k}"]
        lo, hi = np.nanmin(t), np.nanmax(t)
        ax.plot([lo, hi], [lo, hi], color=COLORS["ref"], lw=1, ls="--", label="identity")
        for m in methods:
            p = df[f"{m}_{k}"]
            ax.scatter(t, p, s=8, alpha=0.5, color=COLORS[m], label=NAMES[m], edgecolors="none")
        ax.set_xlabel(f"reference {MEASUREMENT_LABELS[k]}")
        ax.set_ylabel("predicted")
        ax.set_title(MEASUREMENT_LABELS[k])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Predicted vs reference measurements", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def bland_altman_panel(df: pd.DataFrame, methods: list[str], out: Path):
    fig, axes = plt.subplots(len(methods), 4, figsize=(15, 3.6 * len(methods)), squeeze=False)
    for r, m in enumerate(methods):
        for ax, k in zip(axes[r], MEASUREMENT_KEYS):
            t, p = df[f"true_{k}"].to_numpy(), df[f"{m}_{k}"].to_numpy()
            ok = ~(np.isnan(t) | np.isnan(p))
            t, p = t[ok], p[ok]
            if t.size < 3:
                ax.set_title(f"{NAMES[m]}: {MEASUREMENT_LABELS[k]}\n(too few valid frames)", fontsize=8.5)
                continue
            mean, diff = (t + p) / 2, p - t
            bias, sd = diff.mean(), diff.std(ddof=1)
            ax.scatter(mean, diff, s=8, alpha=0.5, color=COLORS[m], edgecolors="none")
            ax.axhline(bias, color=COLORS["ref"], lw=1)
            for y in (bias - 1.96 * sd, bias + 1.96 * sd):
                ax.axhline(y, color=COLORS["ref"], lw=1, ls="--")
            ax.set_title(f"{NAMES[m]}: {MEASUREMENT_LABELS[k]}\nbias {bias:+.2f}, LoA ±{1.96*sd:.2f}", fontsize=8.5)
            ax.set_xlabel("mean of methods")
            ax.set_ylabel("predicted − reference")
    fig.suptitle("Bland–Altman agreement", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def overlay(ax, img: np.ndarray, mask: np.ndarray, sp: float, title: str, annotate: bool = True):
    from cvmeasure.measure.measurements import measure_from_label_map
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask == 1] = matplotlib.colors.to_rgba(COLORS["muscle"], 0.35)
    rgba[mask == 2] = matplotlib.colors.to_rgba(COLORS["fat"], 0.35)
    ax.imshow(rgba)
    if annotate:
        m = measure_from_label_map(mask, sp, clean=True)
        if not np.isnan(m["emd_mm"]):
            muscle = mask == 1
            ys, xs = np.where(muscle)
            c = m["deepest_column_px"]
            col = np.where(muscle[:, c])[0]
            ax.plot([c, c], [col.min(), col.max()], color="#EF4444", lw=1.5)          # EMD
            ymid = int(np.median(ys))
            ax.plot([xs.min(), xs.max()], [ymid, ymid], color="#EF4444", lw=1.5, ls=":")   # EMW
            fat_above = np.where(mask[: col.min(), c] == 2)[0]
            if fat_above.size:
                ax.plot([c + 3, c + 3], [fat_above.min(), fat_above.max()], color="#F59E0B", lw=2)   # fat C
            title += f"\nEMA {m['ema_mm2']:.0f} mm² · EMD {m['emd_mm']:.1f}\nEMW {m['emw_mm']:.1f} · C {m['fat_c_mm']:.1f} mm"
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def qualitative_grid(items, out: Path, suptitle: str):
    n = len(items)
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8.2))
    for j, (fid, img, gt, pred, sp, dice) in enumerate(items):
        axes[0, j].imshow(img, cmap="gray", vmin=0, vmax=1); axes[0, j].set_title(fid, fontsize=8); axes[0, j].axis("off")
        overlay(axes[1, j], img, gt, sp, "reference")
        overlay(axes[2, j], img, pred, sp, f"prediction (Dice {dice:.3f})")
    fig.suptitle(suptitle, y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def error_vs_condition(df: pd.DataFrame, meta: pd.DataFrame, out: Path):
    """Segmentation quality vs acquisition nuisance factors (uses generator params if present)."""
    m = df.merge(meta, on="id", how="left")
    conds = [c for c in ("p_shadow", "p_contact_loss", "p_burn_in_ui") if c in m.columns]
    conts = [c for c in ("p_gain_db", "p_attenuation_db_per_mm", "p_speckle_lateral_px", "true_fat_c_mm") if c in m.columns]
    if not conds and not conts:
        return
    fig, axes = plt.subplots(1, len(conds) + len(conts), figsize=(3.2 * (len(conds) + len(conts)), 3.4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conds):
        groups = [m.loc[m[c] == v, "dice_c1"].dropna() for v in (False, True)]
        ax.boxplot(groups, tick_labels=["no", "yes"], widths=0.5)
        ax.set_title(f"Dice(muscle) vs {c[2:]}", fontsize=8.5)
    for ax, c in zip(axes[len(conds):], conts):
        ax.scatter(m[c], m["dice_c1"], s=6, alpha=0.5, color=COLORS["seg"], edgecolors="none")
        ax.set_title(f"Dice(muscle) vs {c.replace('p_', '').replace('true_', '')}", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def training_curves(hist_paths: dict[str, Path], out: Path):
    fig, axes = plt.subplots(1, len(hist_paths), figsize=(5 * len(hist_paths), 3.4), squeeze=False)
    for ax, (name, p) in zip(axes[0], hist_paths.items()):
        h = pd.read_csv(p)
        ax.plot(h.epoch, h.train_loss, label="train loss", color=COLORS["ref"])
        ax.plot(h.epoch, h.val_loss, label="val loss", color=COLORS["seg"])
        ax.set_xlabel("epoch"); ax.set_title(name)
        if "val_dice_muscle" in h:
            ax2 = ax.twinx(); ax2.plot(h.epoch, h.val_dice_muscle, color=COLORS["muscle"], label="val Dice muscle")
            ax2.set_ylabel("Dice"); ax2.legend(loc="center right", frameon=False, fontsize=8)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
