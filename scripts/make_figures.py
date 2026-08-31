"""Static figures for the README / report: synthetic-data gallery, imaging-chain
walkthrough and training curves.

    python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cvmeasure.synth.generator import GeneratorConfig, SheepLoinUltrasoundGenerator
from cvmeasure.viz import overlay, training_curves

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)


def gallery():
    gen = SheepLoinUltrasoundGenerator(GeneratorConfig(), seed=2024)
    samples = [gen.generate_animal(i)[0] for i in range(8)]
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    for j, s in enumerate(samples):
        axes[0, j].imshow(s.image, cmap="gray", vmin=0, vmax=255); axes[0, j].axis("off")
        axes[0, j].set_title(f"animal {j}", fontsize=8)
        overlay(axes[1, j], s.image / 255.0, s.mask, s.pixel_spacing_mm, "ground truth", annotate=True)
    fig.suptitle("Synthetic sheep loin B-mode frames (top) with pixel-perfect labels and measurements (bottom)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_gallery.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def imaging_chain():
    """Show the same anatomy under different acquisition settings (nuisance factors)."""
    gen = SheepLoinUltrasoundGenerator(GeneratorConfig(), seed=11)
    animal = gen.sample_animal()
    base = gen.sample_frame_params()
    base.update(shadow=False, contact_loss=False, burn_in_ui=False, gain_db=-4, tilt_deg=0, lateral_shift_mm=0)
    variants = [
        ("baseline", {}),
        ("low gain −10 dB", {"gain_db": -10}),
        ("coarse speckle", {"speckle_lateral_px": 5.0, "speckle_axial_px": 1.6}),
        ("strong attenuation", {"attenuation_db_per_mm": 0.14}),
        ("rib shadow", {"shadow": True, "shadow_strength": 0.9}),
        ("poor contact", {"contact_loss": True}),
        ("probe tilt 6°", {"tilt_deg": 6}),
        ("faint fascia", {"rim_strength": 0.6}),
    ]
    fig, axes = plt.subplots(1, len(variants), figsize=(2.5 * len(variants), 2.9))
    for ax, (name, kw) in zip(axes, variants):
        gen.rng = np.random.default_rng(5)     # same speckle realisation
        s = gen.render(animal, {**base, **kw})
        ax.imshow(s.image, cmap="gray", vmin=0, vmax=255); ax.set_title(name, fontsize=9); ax.axis("off")
    fig.suptitle("Same animal, different acquisition nuisance factors (what the model must be invariant to)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "acquisition_factors.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    gallery()
    imaging_chain()
    hp = {n: Path(f"runs/{t}/history.csv") for n, t in (("U-Net segmentation", "seg"), ("CNN regression", "reg"))}
    hp = {n: p for n, p in hp.items() if p.exists()}
    if hp:
        training_curves(hp, OUT / "training_curves.png")
    print("figures ->", OUT)
