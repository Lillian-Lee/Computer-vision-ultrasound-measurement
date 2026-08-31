"""Dataset writing/loading for both synthetic and real (annotated) ultrasound frames.

On-disk layout (identical for synthetic and real data, so the training code never cares
which one it is):

    <root>/
      images/<id>.png          8-bit grayscale B-mode frame
      masks/<id>.png           uint8 label map: 0 background, 1 eye muscle, 2 subcut fat
      metadata.csv             one row per frame: id, animal_id, split, pixel_spacing_mm,
                               ema_mm2, emd_mm, emw_mm, fat_c_mm, + generator params
      generator_config.json    (synthetic only) the priors used

Splits are assigned *by animal*, never by frame - the same sheep must not appear in both
train and test, otherwise the reported error is optimistic (easy to get wrong when several
frames per animal are collected).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cvmeasure.measure.measurements import MEASUREMENT_KEYS


# --------------------------------------------------------------------------------------
# Writing a synthetic dataset
# --------------------------------------------------------------------------------------
def write_synthetic_dataset(root: str | Path, n_animals: int, seed: int = 0,
                            splits: tuple[float, float, float] = (0.7, 0.15, 0.15),
                            config=None, progress: bool = True) -> pd.DataFrame:
    from cvmeasure.synth.generator import GeneratorConfig, SheepLoinUltrasoundGenerator
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    cfg = config or GeneratorConfig()
    gen = SheepLoinUltrasoundGenerator(cfg, seed=seed)

    rng = np.random.default_rng(seed + 1)
    animal_ids = np.arange(n_animals)
    rng.shuffle(animal_ids)
    n_tr = round(splits[0] * n_animals)
    n_va = round(splits[1] * n_animals)
    split_of = {int(a): "train" for a in animal_ids[:n_tr]}
    split_of.update({int(a): "val" for a in animal_ids[n_tr:n_tr + n_va]})
    split_of.update({int(a): "test" for a in animal_ids[n_tr + n_va:]})

    rows = []
    it = range(n_animals)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc=f"generating -> {root}")
    for aid in it:
        for s in gen.generate_animal(aid):
            fid = f"a{aid:05d}_f{s.params['frame_id']}"
            cv2.imwrite(str(root / "images" / f"{fid}.png"), s.image)
            cv2.imwrite(str(root / "masks" / f"{fid}.png"), s.mask)
            row = {"id": fid, "animal_id": aid, "split": split_of[aid],
                   "pixel_spacing_mm": s.pixel_spacing_mm}
            row.update({k: s.measurements[k] for k in MEASUREMENT_KEYS})
            row.update({f"p_{k}": (json.dumps(v) if isinstance(v, list) else v)
                        for k, v in s.params.items() if k not in ("animal_id", "frame_id")})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(root / "metadata.csv", index=False)
    (root / "generator_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return df


# --------------------------------------------------------------------------------------
# Augmentation (kept explicit and dependency-free so it is easy to audit)
# --------------------------------------------------------------------------------------
class UltrasoundAugment:
    """Photometric + mild geometric augmentation appropriate for B-mode ultrasound.

    Note what is *not* here: no vertical flips (depth direction is physically meaningful:
    skin is always at the top) and no large rotations (probe orientation is controlled).
    Horizontal flips are fine (left/right side of the animal). Scale jitter is small
    because it changes the true mm/px calibration - the same warp is applied to the mask
    so label geometry stays consistent, and the regression targets are recomputed from the
    warped mask with the corrected spacing.
    """

    def __init__(self, p_hflip=0.5, max_shift_px=8, max_scale=0.06, gamma=(0.7, 1.4),
                 gain=(0.85, 1.15), noise_std=0.02, p_dropout_band=0.15):
        self.p_hflip, self.max_shift_px, self.max_scale = p_hflip, max_shift_px, max_scale
        self.gamma, self.gain, self.noise_std, self.p_dropout_band = gamma, gain, noise_std, p_dropout_band

    def __call__(self, img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
        H, W = img.shape
        if rng.random() < self.p_hflip:
            img, mask = img[:, ::-1], mask[:, ::-1]
        # small affine (shift + isotropic scale)
        s = 1.0 + rng.uniform(-self.max_scale, self.max_scale)
        tx, ty = rng.uniform(-self.max_shift_px, self.max_shift_px, size=2)
        M = np.array([[s, 0, (1 - s) * W / 2 + tx], [0, s, (1 - s) * H / 2 + ty]], np.float32)
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        # photometric
        x = img.astype(np.float32) / 255.0
        x = np.clip(x * rng.uniform(*self.gain), 0, 1) ** rng.uniform(*self.gamma)
        x = x + rng.normal(0, self.noise_std, x.shape).astype(np.float32)
        if rng.random() < self.p_dropout_band:      # simulate a lateral contact-loss band
            w = int(rng.integers(4, W // 6))
            if rng.random() < 0.5:
                x[:, :w] *= np.linspace(0.1, 1, w)[None, :]
            else:
                x[:, W - w:] *= np.linspace(1, 0.1, w)[None, :]
        return np.clip(x, 0, 1).astype(np.float32), mask.astype(np.int64), s


# --------------------------------------------------------------------------------------
# Torch dataset
# --------------------------------------------------------------------------------------
class LoinUltrasoundDataset(Dataset):
    """Yields dict(image [1,H,W] float, mask [H,W] long, targets [4] float (mm units),
    pixel_spacing_mm, id). Works for synthetic and imported real data alike."""

    def __init__(self, root: str | Path, split: str | None = None, augment: bool = False,
                 seed: int = 0, image_size: int | None = None,
                 target_stats: dict | None = None):
        self.root = Path(root)
        df = pd.read_csv(self.root / "metadata.csv")
        if split is not None:
            df = df[df.split == split].reset_index(drop=True)
        self.df = df
        self.aug = UltrasoundAugment() if augment else None
        self.rng = np.random.default_rng(seed)
        self.image_size = image_size
        self.target_stats = target_stats  # {"mean": [4], "std": [4]} for normalised regression

    def __len__(self):
        return len(self.df)

    @staticmethod
    def compute_target_stats(df: pd.DataFrame) -> dict:
        v = df[MEASUREMENT_KEYS].to_numpy(np.float32)
        return {"mean": v.mean(0).tolist(), "std": (v.std(0) + 1e-6).tolist()}

    def __getitem__(self, i):
        from cvmeasure.measure.measurements import measure_from_label_map
        r = self.df.iloc[i]
        img = cv2.imread(str(self.root / "images" / f"{r.id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.root / "masks" / f"{r.id}.png"), cv2.IMREAD_GRAYSCALE)
        sp = float(r.pixel_spacing_mm)
        if self.image_size and img.shape[0] != self.image_size:
            f = self.image_size / img.shape[0]
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            sp = sp / f
        if self.aug is not None:
            x, mask, scale = self.aug(img, mask, self.rng)
            sp_eff = sp / scale     # zooming in makes each pixel represent fewer mm
            m = measure_from_label_map(mask, sp_eff, clean=False)
            targets = np.array([m[k] for k in MEASUREMENT_KEYS], np.float32)
            sp = sp_eff
        else:
            x = img.astype(np.float32) / 255.0
            mask = mask.astype(np.int64)
            targets = np.array([r[k] for k in MEASUREMENT_KEYS], np.float32)
        if np.isnan(targets).any():   # muscle warped out of frame - fall back to original labels
            targets = np.array([r[k] for k in MEASUREMENT_KEYS], np.float32)
        item = {
            "image": torch.from_numpy(x)[None],
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "targets": torch.from_numpy(targets),
            "pixel_spacing_mm": torch.tensor(sp, dtype=torch.float32),
            "id": str(r.id),
        }
        if self.target_stats is not None:
            mu = torch.tensor(self.target_stats["mean"])
            sd = torch.tensor(self.target_stats["std"])
            item["targets_norm"] = (item["targets"] - mu) / sd
        return item
