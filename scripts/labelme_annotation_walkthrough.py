"""Build and import a small, explicitly synthetic LabelMe annotation example."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cvmeasure.data.annotation import import_labelme
from cvmeasure.synth.generator import GeneratorConfig, SheepLoinUltrasoundGenerator

LABELS = {1: "eye_muscle", 2: "subcut_fat"}
SPLITS = {"synthetic01": "train", "synthetic02": "val", "synthetic03": "test"}


def mask_to_shapes(mask: np.ndarray) -> list[dict]:
    """Convert the two semantic classes into LabelMe polygon shapes."""
    shapes = []
    for class_id, label in LABELS.items():
        contours, _ = cv2.findContours(
            (mask == class_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            points = contour.reshape(-1, 2)
            if len(points) < 3:
                continue
            shapes.append(
                {
                    "label": label,
                    "points": points.astype(float).tolist(),
                    "group_id": None,
                    "description": "synthetic ground-truth polygon for workflow demonstration",
                    "shape_type": "polygon",
                    "flags": {},
                }
            )
    return shapes


def write_labelme_example(raw_dir: Path, seed: int = 17) -> None:
    """Write three synthetic frames and their LabelMe-compatible JSON files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    generator = SheepLoinUltrasoundGenerator(GeneratorConfig(image_size=128), seed=seed)
    for index, animal_id in enumerate(SPLITS, start=1):
        sample = generator.render(generator.sample_animal(), generator.sample_frame_params())
        stem = f"{animal_id}_frame01"
        image_name = f"{stem}.png"
        cv2.imwrite(str(raw_dir / image_name), sample.image)
        annotation = {
            "version": "5.5.0",
            "flags": {"synthetic_example": True},
            "shapes": mask_to_shapes(sample.mask),
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": int(sample.image.shape[0]),
            "imageWidth": int(sample.image.shape[1]),
            "description": (
                "Synthetic cvmeasure frame. This is a reproducibility fixture, not a real livestock annotation."
            ),
        }
        (raw_dir / f"{stem}.json").write_text(json.dumps(annotation, indent=2), encoding="utf-8")


def write_overlay(image_path: Path, mask_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colours = {1: (50, 210, 50), 2: (40, 170, 255)}
    for class_id, colour in colours.items():
        contours, _ = cv2.findContours(
            (mask == class_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, colour, 2)
    cv2.putText(canvas, "SYNTHETIC LABELME EXAMPLE", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def build_walkthrough(output_root: Path, seed: int = 17) -> None:
    raw_dir = output_root / "labelme_raw"
    dataset_dir = output_root / "imported_dataset"
    write_labelme_example(raw_dir, seed=seed)
    metadata = import_labelme(
        raw_dir,
        dataset_dir,
        pixel_spacing_mm=0.42,
        split_fn=lambda frame_id: SPLITS[frame_id.split("_")[0]],
    )
    first_id = metadata.iloc[0]["id"]
    write_overlay(
        dataset_dir / "images" / f"{first_id}.png",
        dataset_dir / "masks" / f"{first_id}.png",
        output_root / "annotation_overlay.png",
    )
    print(metadata[["id", "animal_id", "split", "pixel_spacing_mm", "ema_mm2", "emd_mm", "emw_mm", "fat_c_mm"]].to_string(index=False))
    print(f"\nWalkthrough written to: {output_root.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/labelme_walkthrough"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    build_walkthrough(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
