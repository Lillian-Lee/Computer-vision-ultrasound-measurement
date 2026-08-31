"""Physics-inspired synthetic B-mode ultrasound generator for sheep loin cross-sections.

Why synthetic data?
-------------------
There is no public dataset of sheep loin (longissimus dorsi, "eye muscle") ultrasound
images with expert annotations. To build and *validate* a full measurement pipeline
end-to-end, this module simulates the imaging chain that a linear-array veterinary
scanner produces at the 12th/13th-rib site:

    anatomy (skin / subcutaneous fat / eye muscle / bone)  ->  echogenicity map
    -> diffuse scatterers  ->  point-spread-function convolution (speckle)
    -> depth attenuation + TGC  ->  acoustic shadowing  ->  log-compression + gain
    -> 8-bit B-mode frame (+ optional scanner UI burn-in)

Every frame ships with a pixel-perfect ground-truth mask (0 background, 1 eye muscle,
2 subcutaneous fat) and the true anatomical measurements in millimetres, so the
segmentation model *and* the mask->measurement geometry can be validated exactly.

The generator is intentionally parameterised by the same quantities a scanning
technician records (eye-muscle depth/width/area, fat depth at the C-site), and by
acquisition nuisance factors (gain, speckle scale, shadowing, probe tilt), so that
domain-shift experiments are one config change away.

Nothing here should be mistaken for real animals: the anatomy distributions are
chosen to be plausible for NZ lambs at scanning weight (see docs/01-domain-research.md)
but they are hand-set priors, not measured populations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter
from scipy.signal import fftconvolve

BACKGROUND, MUSCLE, FAT = 0, 1, 2
CLASS_NAMES = {BACKGROUND: "background", MUSCLE: "eye_muscle", FAT: "subcut_fat"}


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class AnatomyPriors:
    """Population priors (mm) for lamb loin cross-sections. Hand-set, plausible ranges."""
    emw_mm: tuple[float, float] = (48.0, 72.0)     # eye muscle width  (A measurement)
    emd_mm: tuple[float, float] = (22.0, 38.0)     # eye muscle depth  (B measurement)
    fat_c_mm: tuple[float, float] = (1.5, 8.0)     # fat depth over deepest point of eye muscle (C site)
    skin_mm: tuple[float, float] = (1.5, 3.0)
    top_flatness: tuple[float, float] = (0.45, 0.8)  # ratio of upper to lower semi-depth (D-shape)
    tilt_deg: tuple[float, float] = (-6.0, 6.0)
    lateral_shift_mm: tuple[float, float] = (-6.0, 6.0)


@dataclass
class AcquisitionPriors:
    """Scanner nuisance factors."""
    gain_db: tuple[float, float] = (-10.0, 0.0)
    dynamic_range_db: tuple[float, float] = (45.0, 65.0)
    speckle_axial_px: tuple[float, float] = (0.9, 1.6)
    speckle_lateral_px: tuple[float, float] = (2.0, 4.0)
    attenuation_db_per_mm: tuple[float, float] = (0.03, 0.09)   # residual after TGC
    shadow_prob: float = 0.7                                     # rib / transverse-process shadow
    shadow_strength: tuple[float, float] = (0.35, 0.85)
    contact_loss_prob: float = 0.15                              # poor probe contact -> lateral dropout
    electronic_noise: tuple[float, float] = (0.005, 0.03)
    burn_in_ui_prob: float = 0.5                                 # depth ticks / text like a real console


@dataclass
class GeneratorConfig:
    image_size: int = 192
    pixel_spacing_mm: float = 0.42          # 192 px * 0.42 mm ~ 80 mm field of view
    frames_per_animal: tuple[int, int] = (1, 3)
    anatomy: AnatomyPriors = field(default_factory=AnatomyPriors)
    acquisition: AcquisitionPriors = field(default_factory=AcquisitionPriors)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Sample:
    image: np.ndarray            # uint8 [H, W]
    mask: np.ndarray             # uint8 [H, W] in {0,1,2}
    measurements: dict           # true anatomical values in mm / mm^2
    params: dict                 # all sampled anatomy + acquisition parameters
    pixel_spacing_mm: float


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------
def _uniform(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(*lo_hi))


def _eye_muscle_mask(H: int, W: int, cx: float, cy: float, a: float, b_lower: float,
                     b_upper: float, tilt_rad: float, harmonics: np.ndarray) -> np.ndarray:
    """D-shaped (flattened-top) ellipse with smooth low-frequency boundary perturbation.

    Real longissimus cross-sections sit under a fairly flat fat/fascia plane and bulge
    downward/laterally, so the upper semi-depth is shorter than the lower one.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    x = xx - cx
    y = yy - cy
    ct, st = np.cos(tilt_rad), np.sin(tilt_rad)
    xr = ct * x + st * y
    yr = -st * x + ct * y
    b = np.where(yr < 0, b_upper, b_lower)
    theta = np.arctan2(yr / b, xr / a)
    # smooth radial modulation r(theta) = 1 + sum_k A_k cos(k theta + phi_k)
    mod = np.ones_like(theta)
    for k, (amp, phase) in enumerate(harmonics, start=2):
        mod += amp * np.cos(k * theta + phase)
    r = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    return r <= mod


def _measure_mask_mm(mask_muscle: np.ndarray, mask_fat: np.ndarray, spacing: float) -> dict:
    """Ground-truth measurements computed directly from the analytic masks (mm)."""
    from cvmeasure.measure.measurements import measure_from_masks
    return measure_from_masks(mask_muscle, mask_fat, spacing)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def _render_bmode(echo: np.ndarray, shadow_gain: np.ndarray, rng: np.random.Generator,
                  acq: dict, spacing: float) -> np.ndarray:
    """Turn an echogenicity map into a speckled, attenuated, log-compressed B-mode frame."""
    H, W = echo.shape
    # 1. diffuse scatterers: complex Gaussian field weighted by local echogenicity
    scat = (rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))) * (echo + 1e-6)
    # 2. PSF: axial gaussian with carrier (RF-like) x lateral gaussian
    sa, sl = acq["speckle_axial_px"], acq["speckle_lateral_px"]
    ky = np.arange(-int(4 * sa) - 2, int(4 * sa) + 3)
    kx = np.arange(-int(4 * sl) - 2, int(4 * sl) + 3)
    ax = np.exp(-0.5 * (ky / sa) ** 2) * np.cos(2 * np.pi * ky / (2.2 * sa))
    lat = np.exp(-0.5 * (kx / sl) ** 2)
    psf = np.outer(ax, lat)
    rf = fftconvolve(scat, psf, mode="same")
    env = np.abs(rf)
    # 3. depth attenuation (residual after TGC) and acoustic shadow
    depth_mm = np.arange(H)[:, None] * spacing
    env = env * (10 ** (-acq["attenuation_db_per_mm"] * depth_mm / 20.0)) * shadow_gain
    # 4. specular boundaries: add coherent component from echogenicity gradient (bright fascia)
    spec = np.abs(np.gradient(gaussian_filter(echo, 1.0), axis=0))
    env = env + 6.0 * spec * env.mean()
    # 5. electronic noise, log compression, gain, dynamic range
    env = env + rng.standard_normal((H, W)) * acq["electronic_noise"] * env.mean()
    env = np.clip(env, 1e-6, None)
    db = 20 * np.log10(env / (env.max() + 1e-9)) + acq["gain_db"]
    dr = acq["dynamic_range_db"]
    img = np.clip((db + dr) / dr, 0, 1)
    return img


def _burn_in_ui(img: np.ndarray, rng: np.random.Generator, spacing: float) -> np.ndarray:
    """Mimic scanner console overlays: depth ruler ticks and a small text block."""
    H, W = img.shape
    out = img.copy()
    x0 = W - 4
    every_px = max(1, round(10.0 / spacing))     # 1 cm ticks
    for y in range(0, H, every_px):
        out[y, x0 - 3:x0] = 1.0
    for y in range(0, H, max(1, every_px // 2)):
        out[y, x0 - 1:x0] = 1.0
    # small "text" block top-left made of random bright pixels (like a frozen label)
    if rng.random() < 0.8:
        h, w = 5, int(rng.integers(20, 40))
        block = rng.random((h, w)) < 0.35
        out[2:2 + h, 3:3 + w] = np.where(block, 0.9, out[2:2 + h, 3:3 + w])
    return out


# --------------------------------------------------------------------------------------
# Main generator
# --------------------------------------------------------------------------------------
class SheepLoinUltrasoundGenerator:
    def __init__(self, config: GeneratorConfig | None = None, seed: int = 0):
        self.cfg = config or GeneratorConfig()
        self.rng = np.random.default_rng(seed)

    # ---- anatomy sampling per animal --------------------------------------------------
    def sample_animal(self) -> dict:
        p, rng = self.cfg.anatomy, self.rng
        emw = _uniform(rng, p.emw_mm)
        # depth correlated with width (bigger animals are bigger in both), plus noise
        t = (emw - p.emw_mm[0]) / (p.emw_mm[1] - p.emw_mm[0])
        emd = p.emd_mm[0] + t * (p.emd_mm[1] - p.emd_mm[0]) + rng.normal(0, 2.5)
        emd = float(np.clip(emd, *p.emd_mm))
        # fatter animals tend to have more fat, weakly correlated with size
        fat_c = float(np.clip(rng.uniform(*p.fat_c_mm) * (0.75 + 0.5 * t), *p.fat_c_mm))
        harmonics = np.stack([rng.uniform(0.0, 0.05, size=3), rng.uniform(0, 2 * np.pi, size=3)], 1)
        return {
            "emw_mm": emw,
            "emd_mm": emd,
            "fat_c_mm": fat_c,
            "skin_mm": _uniform(rng, p.skin_mm),
            "top_flatness": _uniform(rng, p.top_flatness),
            "harmonics": harmonics.tolist(),
        }

    def sample_frame_params(self) -> dict:
        p, a, rng = self.cfg.anatomy, self.cfg.acquisition, self.rng
        return {
            "tilt_deg": _uniform(rng, p.tilt_deg),
            "lateral_shift_mm": _uniform(rng, p.lateral_shift_mm),
            "probe_pressure": float(rng.uniform(0.9, 1.05)),   # compresses fat slightly
            "gain_db": _uniform(rng, a.gain_db),
            "dynamic_range_db": _uniform(rng, a.dynamic_range_db),
            "speckle_axial_px": _uniform(rng, a.speckle_axial_px),
            "speckle_lateral_px": _uniform(rng, a.speckle_lateral_px),
            "attenuation_db_per_mm": _uniform(rng, a.attenuation_db_per_mm),
            "shadow": bool(rng.random() < a.shadow_prob),
            "shadow_strength": _uniform(rng, a.shadow_strength),
            "contact_loss": bool(rng.random() < a.contact_loss_prob),
            "electronic_noise": _uniform(rng, a.electronic_noise),
            "burn_in_ui": bool(rng.random() < a.burn_in_ui_prob),
            "rim_strength": float(rng.uniform(0.8, 2.4)),
        }

    # ---- rendering one frame -------------------------------------------------------------
    def render(self, animal: dict, frame: dict) -> Sample:
        cfg, rng = self.cfg, self.rng
        H = W = cfg.image_size
        sp = cfg.pixel_spacing_mm
        mm = lambda v: v / sp

        # geometry (pixels)
        skin_px = mm(animal["skin_mm"])
        fat_px = mm(animal["fat_c_mm"] * frame["probe_pressure"])
        a = mm(animal["emw_mm"]) / 2.0
        depth_px = mm(animal["emd_mm"])
        b_upper = depth_px * animal["top_flatness"] / (1.0 + animal["top_flatness"])
        b_lower = depth_px - b_upper
        cx = W / 2.0 + mm(frame["lateral_shift_mm"])
        cy = skin_px + fat_px + b_upper          # muscle top touches bottom of fat layer at centre
        tilt = np.deg2rad(frame["tilt_deg"])
        muscle = _eye_muscle_mask(H, W, cx, cy, a, b_lower, b_upper, tilt, np.array(animal["harmonics"]))
        muscle = binary_fill_holes(muscle)

        # fat layer: follows the top of the muscle; outside the muscle it drapes down laterally
        yy, _xx = np.mgrid[0:H, 0:W]
        cols = np.arange(W)
        top_of_muscle = np.full(W, np.nan)
        for c in cols:
            ys = np.where(muscle[:, c])[0]
            if ys.size:
                top_of_muscle[c] = ys[0]
        # smooth fat thickness variation along the width
        thick = fat_px * (1.0 + 0.15 * np.sin(cols / W * np.pi * rng.uniform(1, 3) + rng.uniform(0, 6)))
        # where there is muscle, fat sits between skin and muscle top; laterally it thickens (fat lobes)
        fat_bottom = np.where(np.isnan(top_of_muscle),
                              skin_px + thick * rng.uniform(1.4, 2.6),
                              top_of_muscle)
        fat_bottom = gaussian_filter(fat_bottom, 5.0)
        fat = (yy >= skin_px) & (yy < fat_bottom[None, :]) & (~muscle)

        mask = np.zeros((H, W), np.uint8)
        mask[fat] = FAT
        mask[muscle] = MUSCLE

        # ---- echogenicity map ------------------------------------------------------------
        echo = np.full((H, W), 0.22)                       # generic connective / inter-muscular tissue
        echo[yy < skin_px] = 0.95                          # skin: very bright
        echo[fat] = 0.48                                    # subcutaneous fat: mid-grey with fascia lines
        echo[muscle] = 0.07                                 # eye muscle: hypoechoic with fine texture
        # intra-fat fascia line (a bright horizontal band midway through the fat)
        mid_fat = (skin_px + fat_bottom[None, :]) / 2.0
        echo[fat & (np.abs(yy - mid_fat) < 0.8)] = 0.85
        # muscle boundary fascia: bright rim just outside the muscle
        # (specular reflection is strongest where the boundary is perpendicular to the beam,
        #  so the lateral walls of the muscle are much fainter than its roof/floor - as in real scans)
        sm = gaussian_filter(muscle.astype(float), 1.5)
        rim = np.clip(sm - gaussian_filter(muscle.astype(float), 0.5), 0, None)
        gy, gx = np.gradient(sm)
        incidence = np.abs(gy) / (np.hypot(gx, gy) + 1e-6)
        echo += frame["rim_strength"] * rim * incidence ** 1.5
        # muscle internal texture: faint fibrous streaks
        streak = gaussian_filter(rng.standard_normal((H, W)), (0.8, 6.0))
        echo[muscle] += 0.03 * streak[muscle]
        # bone / transverse process beneath the muscle: bright interface then shadow
        shadow_gain = np.ones((H, W))
        muscle_bottom = np.array([np.where(muscle[:, c])[0].max() if muscle[:, c].any() else -1 for c in cols])
        if frame["shadow"]:
            # a rib-like bright arc under part of the muscle
            c0 = int(np.clip(cx + rng.uniform(-a * 0.6, a * 0.6), 5, W - 6))
            half_w = int(rng.uniform(mm(4), mm(9)))
            for c in range(max(0, c0 - half_w), min(W, c0 + half_w)):
                if muscle_bottom[c] > 0:
                    y0 = muscle_bottom[c] + int(mm(rng.uniform(1.0, 3.0)))
                    if y0 < H:
                        echo[y0:min(H, y0 + 2), c] = 1.0
                        shadow_gain[y0 + 2:, c] *= (1.0 - frame["shadow_strength"])
        # deep tissue: darker with depth (below muscle) to mimic bone/other muscle
        deep = (yy > (muscle_bottom[None, :] + mm(2))) & (muscle_bottom[None, :] > 0)
        echo[deep] *= 0.75
        echo = np.clip(echo, 0.0, 1.2)

        # poor probe contact: one lateral edge drops out
        if frame["contact_loss"]:
            side = rng.choice([-1, 1])
            width = int(rng.uniform(mm(4), mm(12)))
            ramp = np.linspace(0.05, 1.0, width)
            if side < 0:
                shadow_gain[:, :width] *= ramp[None, :]
            else:
                shadow_gain[:, W - width:] *= ramp[None, ::-1]

        img = _render_bmode(echo, shadow_gain, rng, frame, sp)
        if frame["burn_in_ui"]:
            img = _burn_in_ui(img, rng, sp)
        img8 = (img * 255).round().astype(np.uint8)

        meas = _measure_mask_mm(mask == MUSCLE, mask == FAT, sp)
        params = {**{k: v for k, v in animal.items()}, **frame}
        return Sample(image=img8, mask=mask, measurements=meas, params=params, pixel_spacing_mm=sp)

    def generate_animal(self, animal_id: int) -> list[Sample]:
        animal = self.sample_animal()
        n = int(self.rng.integers(self.cfg.frames_per_animal[0], self.cfg.frames_per_animal[1] + 1))
        out = []
        for f in range(n):
            s = self.render(animal, self.sample_frame_params())
            s.params["animal_id"] = animal_id
            s.params["frame_id"] = f
            out.append(s)
        return out
