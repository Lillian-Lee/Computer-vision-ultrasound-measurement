import json
import numpy as np
import torch

from cvmeasure.models.unet import UNet, DiceCELoss
from cvmeasure.models.regressor import MeasurementRegressor
from cvmeasure.data.dataset import write_synthetic_dataset, LoinUltrasoundDataset
from cvmeasure.data.annotation import import_labelme
from cvmeasure.synth.generator import GeneratorConfig
from cvmeasure.metrics import agreement_stats, lins_ccc, dice_iou_per_class


def test_unet_shapes_and_loss():
    m = UNet(base=4, depth=3)
    x = torch.rand(2, 1, 64, 64)
    y = m(x)
    assert y.shape == (2, 3, 64, 64)
    loss = DiceCELoss()(y, torch.randint(0, 3, (2, 64, 64)))
    assert torch.isfinite(loss)


def test_regressor_shape():
    m = MeasurementRegressor(widths=(4, 8, 16))
    assert m(torch.rand(2, 1, 64, 64)).shape == (2, 4)


def test_dataset_roundtrip_and_split_by_animal(tmp_path):
    df = write_synthetic_dataset(tmp_path, n_animals=12, seed=0, config=GeneratorConfig(image_size=64), progress=False)
    assert (tmp_path / "metadata.csv").exists()
    for split_a in ("train", "val", "test"):
        for split_b in ("train", "val", "test"):
            if split_a != split_b:
                assert not set(df[df.split == split_a].animal_id) & set(df[df.split == split_b].animal_id)
    ds = LoinUltrasoundDataset(tmp_path, "train", augment=True, seed=0)
    item = ds[0]
    assert item["image"].shape == (1, 64, 64) and item["mask"].shape == (64, 64)
    assert item["targets"].shape == (4,) and torch.isfinite(item["targets"]).all()


def test_labelme_import(tmp_path):
    import cv2
    img_dir = tmp_path / "raw"; img_dir.mkdir()
    img = (np.random.rand(80, 80) * 255).astype(np.uint8)
    cv2.imwrite(str(img_dir / "sheep01_f1.png"), img)
    ann = {"imagePath": "sheep01_f1.png", "shapes": [
        {"label": "eye_muscle", "shape_type": "polygon", "points": [[10, 30], [70, 30], [70, 60], [10, 60]]},
        {"label": "fat", "shape_type": "polygon", "points": [[0, 20], [80, 20], [80, 30], [0, 30]]}]}
    (img_dir / "sheep01_f1.json").write_text(json.dumps(ann))
    df = import_labelme(img_dir, tmp_path / "ds", pixel_spacing_mm=0.5)
    assert len(df) == 1 and df.animal_id[0] == "sheep01"
    assert abs(df.emd_mm[0] - 15.5) < 1.0 and abs(df.fat_c_mm[0] - 5.0) < 1.0


def test_metrics():
    t = np.linspace(20, 40, 50); p = t + np.random.default_rng(0).normal(0, 0.5, 50)
    s = agreement_stats(p, t)
    assert s["ccc"] > 0.98 and s["mae"] < 1.0
    assert abs(lins_ccc(t, t) - 1.0) < 1e-9
    a = np.zeros((10, 10), int); a[2:6, 2:6] = 1
    d = dice_iou_per_class(a, a)
    assert d["dice_c1"] > 0.999
