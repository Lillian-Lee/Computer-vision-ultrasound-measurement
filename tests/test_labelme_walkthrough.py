import json

import cv2
import pytest

from cvmeasure.data.labelme_walkthrough import SPLITS, build_walkthrough


def test_synthetic_labelme_walkthrough_roundtrip(tmp_path):
    build_walkthrough(tmp_path, seed=17)

    raw_dir = tmp_path / "labelme_raw"
    dataset_dir = tmp_path / "imported_dataset"
    annotations = sorted(raw_dir.glob("*.json"))
    assert len(annotations) == 3
    assert all(json.loads(path.read_text(encoding="utf-8"))["flags"]["synthetic_example"] for path in annotations)

    import pandas as pd

    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    assert set(metadata["split"]) == set(SPLITS.values())
    assert metadata.groupby("animal_id")["split"].nunique().max() == 1
    assert metadata["pixel_spacing_mm"].tolist() == pytest.approx([0.42, 0.42, 0.42])
    assert metadata[["ema_mm2", "emd_mm", "emw_mm", "fat_c_mm"]].notna().all().all()

    for mask_path in (dataset_dir / "masks").glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert set(mask.ravel()) == {0, 1, 2}
    assert (tmp_path / "annotation_overlay.png").exists()
