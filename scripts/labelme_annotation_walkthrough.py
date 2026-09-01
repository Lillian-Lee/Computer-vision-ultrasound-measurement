"""Run the reproducible synthetic LabelMe annotation walkthrough."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvmeasure.data.labelme_walkthrough import build_walkthrough


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/labelme_walkthrough"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    build_walkthrough(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
