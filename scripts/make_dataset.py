"""Generate the synthetic training set and the domain-shift test set.

    python scripts/make_dataset.py --config configs/default.yaml
"""
import argparse
from dataclasses import replace
from pathlib import Path

import yaml

from cvmeasure.data.dataset import write_synthetic_dataset
from cvmeasure.synth.generator import AcquisitionPriors, GeneratorConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-animals", type=int, default=None)
    a = ap.parse_args()
    with Path(a.config).open(encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    root = Path(cfg["data"]["root"])
    size = cfg["data"]["image_size"]
    n = a.n_animals or cfg["synthetic"]["n_animals"]

    gcfg = GeneratorConfig(image_size=size)
    df = write_synthetic_dataset(root, n_animals=n, seed=cfg["synthetic"]["seed"], config=gcfg)
    print(df.split.value_counts().to_dict(), "frames written to", root)

    st = cfg["synthetic"].get("shift_test")
    if st:
        acq = replace(AcquisitionPriors(), **{k: (tuple(v) if isinstance(v, list) else v)
                                              for k, v in st["acquisition"].items()})
        gshift = GeneratorConfig(image_size=size, acquisition=acq)
        df2 = write_synthetic_dataset(root.parent / (root.name + "_shift"), n_animals=st["n_animals"],
                                      seed=st["seed"], config=gshift, splits=(0.0, 0.0, 1.0))
        print(len(df2), "domain-shift test frames written to", root.parent / (root.name + "_shift"))


if __name__ == "__main__":
    main()
