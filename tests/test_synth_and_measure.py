import numpy as np
import pytest

from cvmeasure.measure.measurements import measure_from_masks, measure_from_label_map, MEASUREMENT_KEYS
from cvmeasure.synth.generator import SheepLoinUltrasoundGenerator, GeneratorConfig, MUSCLE, FAT


def test_measure_known_rectangle():
    H = W = 100
    lab = np.zeros((H, W), np.uint8)
    lab[20:30, 10:90] = FAT          # 10 px thick fat
    lab[30:60, 10:90] = MUSCLE       # 30 px deep, 80 px wide
    m = measure_from_label_map(lab, pixel_spacing_mm=0.5, clean=False)
    assert m["emd_mm"] == pytest.approx(15.0)
    assert m["emw_mm"] == pytest.approx(40.0)
    assert m["ema_mm2"] == pytest.approx(30 * 80 * 0.25)
    assert m["fat_c_mm"] == pytest.approx(5.0)
    assert m["qc"]["muscle_touches_border"] is False


def test_measure_handles_empty_and_border():
    lab = np.zeros((50, 50), np.uint8)
    m = measure_from_label_map(lab, 0.4)
    assert np.isnan(m["emd_mm"]) and m["qc"]["no_muscle"]
    lab[10:20, 0:30] = MUSCLE
    m = measure_from_label_map(lab, 0.4)
    assert m["qc"]["muscle_touches_border"] is True


def test_cleaning_removes_fragments():
    lab = np.zeros((60, 60), np.uint8)
    lab[20:40, 10:40] = MUSCLE
    lab[50:52, 50:52] = MUSCLE       # spurious blob
    m = measure_from_label_map(lab, 1.0, clean=True)
    assert m["emw_mm"] == pytest.approx(30.0)
    assert m["ema_mm2"] == pytest.approx(600.0)


def test_generator_reproducible_and_consistent():
    cfg = GeneratorConfig(image_size=96)
    a = SheepLoinUltrasoundGenerator(cfg, seed=7).generate_animal(0)
    b = SheepLoinUltrasoundGenerator(cfg, seed=7).generate_animal(0)
    assert np.array_equal(a[0].image, b[0].image)
    s = a[0]
    assert s.image.dtype == np.uint8 and s.image.shape == (96, 96)
    assert set(np.unique(s.mask)) <= {0, 1, 2}
    # measurements are recomputable from the mask and within the priors
    m = measure_from_masks(s.mask == MUSCLE, s.mask == FAT, s.pixel_spacing_mm)
    for k in MEASUREMENT_KEYS:
        assert m[k] == pytest.approx(s.measurements[k])
    assert 15 < s.measurements["emd_mm"] < 45
    assert 35 < s.measurements["emw_mm"] < 85


def test_frames_of_one_animal_share_anatomy():
    gen = SheepLoinUltrasoundGenerator(GeneratorConfig(image_size=96, frames_per_animal=(3, 3)), seed=1)
    frames = gen.generate_animal(3)
    ids = {f.params["animal_id"] for f in frames}
    assert ids == {3} and len(frames) == 3
    emd = [f.measurements["emd_mm"] for f in frames]
    assert max(emd) - min(emd) < 6.0     # tilt / pressure change the view slightly, not the animal
