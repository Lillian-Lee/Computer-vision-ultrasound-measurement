"""Training loop for both tasks.

    python -m cvmeasure.train --config configs/default.yaml --task seg
    python -m cvmeasure.train --config configs/default.yaml --task reg

Writes to <run_dir>/: best.pt, last.pt, history.csv, config.yaml, target_stats.json (reg).
Model selection is on the validation split (by-animal split, see data/dataset.py).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from cvmeasure.data.dataset import LoinUltrasoundDataset
from cvmeasure.metrics import dice_iou_per_class
from cvmeasure.models.regressor import MeasurementRegressor
from cvmeasure.models.unet import DiceCELoss, build_segmentation_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(task: str, cfg: dict) -> torch.nn.Module:
    if task == "seg":
        m = cfg["seg_model"]
        return build_segmentation_model(m["name"], **m.get("kwargs", {}))
    return MeasurementRegressor(**cfg["reg_model"].get("kwargs", {}))


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run(cfg: dict, task: str, run_dir: Path, device: str = "cpu", max_epochs: int | None = None):
    set_seed(cfg.get("seed", 0))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    root = cfg["data"]["root"]
    tr_meta = pd.read_csv(Path(root) / "metadata.csv")
    stats = LoinUltrasoundDataset.compute_target_stats(tr_meta[tr_meta.split == "train"])
    (run_dir / "target_stats.json").write_text(json.dumps(stats))

    ds_tr = LoinUltrasoundDataset(root, "train", augment=cfg["train"].get("augment", True),
                                  seed=cfg.get("seed", 0), image_size=cfg["data"].get("image_size"), target_stats=stats)
    frac = cfg["train"].get("subset_fraction")          # low-data ablation: keep a fraction of *animals*
    if frac and frac < 1.0:
        animals = ds_tr.df.animal_id.unique()
        keep = np.random.default_rng(cfg.get("seed", 0)).choice(animals, size=max(1, int(len(animals) * frac)), replace=False)
        ds_tr.df = ds_tr.df[ds_tr.df.animal_id.isin(keep)].reset_index(drop=True)
    ds_va = LoinUltrasoundDataset(root, "val", augment=False, image_size=cfg["data"].get("image_size"), target_stats=stats)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                       num_workers=cfg["train"].get("num_workers", 0), drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"], shuffle=False,
                       num_workers=cfg["train"].get("num_workers", 0))

    model = build_model(task, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    epochs = max_epochs or cfg["train"]["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["train"]["lr"], total_steps=epochs * len(dl_tr),
                                                pct_start=0.15)
    criterion = DiceCELoss() if task == "seg" else torch.nn.SmoothL1Loss()
    print(f"[{task}] model={model.__class__.__name__} params={n_params/1e6:.2f}M "
          f"train={len(ds_tr)} val={len(ds_va)} epochs={epochs} device={device}")

    history, best_score = [], -np.inf
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in dl_tr:
            x = b["image"].to(device)
            y = b["mask"].to(device) if task == "seg" else b["targets_norm"].to(device)
            out = model(x)
            loss = criterion(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = tot / max(n, 1)

        # ---- validation --------------------------------------------------------------
        model.eval()
        vtot, vn, dices, maes = 0.0, 0, [], []
        with torch.no_grad():
            for b in dl_va:
                x = b["image"].to(device)
                out = model(x)
                if task == "seg":
                    y = b["mask"].to(device)
                    vtot += criterion(out, y).item() * x.size(0)
                    pred = out.argmax(1).cpu().numpy()
                    for p, t in zip(pred, y.cpu().numpy()):
                        d = dice_iou_per_class(p, t)
                        dices.append((d["dice_c1"], d["dice_c2"]))
                else:
                    y = b["targets_norm"].to(device)
                    vtot += criterion(out, y).item() * x.size(0)
                    mu, sd = torch.tensor(stats["mean"]), torch.tensor(stats["std"])
                    pred_mm = out.cpu() * sd + mu
                    maes.append((pred_mm - b["targets"]).abs().numpy())
                vn += x.size(0)
        val_loss = vtot / max(vn, 1)
        rec = {"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "lr": sched.get_last_lr()[0],
               "sec": time.time() - t0}
        if task == "seg":
            d = np.array(dices).mean(0)
            rec.update(val_dice_muscle=float(d[0]), val_dice_fat=float(d[1]))
            score = float(d.mean())
        else:
            m = np.concatenate(maes).mean(0)
            rec.update(val_mae_ema=float(m[0]), val_mae_emd=float(m[1]), val_mae_emw=float(m[2]), val_mae_fatc=float(m[3]))
            score = -float(val_loss)
        history.append(rec)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        torch.save({"model": model.state_dict(), "task": task, "cfg": cfg, "epoch": ep}, run_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "task": task, "cfg": cfg, "epoch": ep, "score": score},
                       run_dir / "best.pt")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--task", choices=["seg", "reg"], required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run_dir = Path(a.run_dir or Path(cfg["runs_dir"]) / a.task)
    run(cfg, a.task, run_dir, a.device, a.epochs)


if __name__ == "__main__":
    main()
