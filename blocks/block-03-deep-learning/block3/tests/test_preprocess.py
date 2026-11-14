"""Unit tests for block3/src/preprocess.py (windowing E1 modes, HU checks, slice ordering, orig-grid Dice/HD95)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocess import (  # noqa: E402
     apply_window,
    apply_window_mode,
    compute_binary_dice,
    compute_hd95_mm,
    evaluate_mask_on_orig_grid,
    order_slices_by_z,
)


def test_apply_window_brain_and_subdural_bounds():
    hu = np.array([-1000, -20, 0, 40, 80, 180, 1000], dtype=np.int16)
    brain = apply_window(hu, level=40.0, width=80.0)     # [0, 80]
    subdural = apply_window(hu, level=80.0, width=200.0) # [-20, 180]

    assert np.allclose(brain, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], atol=1e-5)
    assert np.allclose(subdural, [0.0, 0.0, 0.1, 0.3, 0.5, 1.0, 1.0], atol=1e-5)
    # Acute SDH around 130 HU is saturated (1.0) in brain window but resolved (0.75) in subdural window
    sdh_130 = np.array([130], dtype=np.int16)
    assert float(apply_window(sdh_130, 40.0, 80.0)[0]) == 1.0
    assert np.isclose(float(apply_window(sdh_130, 80.0, 200.0)[0]), 0.75)


def test_apply_window_mode_shapes_and_double_offset_guard():
    vol = np.random.randint(-1024, 1500, size=(6, 64, 64), dtype=np.int16)
    for mode in ("brain_only", "brain_subdural", "brain_subdural_bone"):
        out_3ch = apply_window_mode(vol, mode=mode, expand_to_3ch=True)
        assert out_3ch.shape == (6, 3, 64, 64) and out_3ch.dtype == np.float32
        assert 0.0 <= float(out_3ch.min()) <= float(out_3ch.max()) <= 1.0

    out_raw_2ch = apply_window_mode(vol, mode="brain_subdural", expand_to_3ch=False)
    assert out_raw_2ch.shape == (6, 2, 64, 64)

    # Guard against double -1024 offset
    corrupt_vol = vol - 1500
    with pytest.raises(ValueError, match="double -1024 offset"):
        apply_window_mode(corrupt_vol, mode="brain_only")


def test_order_slices_by_z_sorts_strictly_ascending():
    s0 = np.full((16, 16), 10, dtype=np.int16)
    s1 = np.full((16, 16), 20, dtype=np.int16)
    s2 = np.full((16, 16), 30, dtype=np.int16)
    shuffled = [(15.0, 3, s2), (5.0, 1, s0), (10.0, 2, s1)]
    vol, z_mm = order_slices_by_z(shuffled)
    assert list(z_mm) == [5.0, 10.0, 15.0]
    assert int(vol[0, 0, 0]) == 10 and int(vol[1, 0, 0]) == 20 and int(vol[2, 0, 0]) == 30


def test_evaluate_mask_on_orig_grid_restores_512_and_computes_metrics():
    gt_512 = np.zeros((8, 512, 512), dtype=np.uint8)
    gt_512[2:6, 120:200, 160:240] = 1
    pred_256 = np.zeros((8, 256, 256), dtype=np.uint8)
    pred_256[2:6, 60:100, 80:120] = 1

    res = evaluate_mask_on_orig_grid(pred_256, gt_512, orig_spacing_xyz_mm=(0.4883, 0.4883, 5.0))
    assert res["evaluated_shape_zyx"] == [8, 512, 512]
    assert res["dice_on_orig_grid"] == 1.0
    assert res["hd95_mm_on_orig_grid"] == 0.0

    # Shifted by 10 pixels in X at 512x512 (overhang distances ramp 0.5..5.0 mm -> 95th percentile = 4.0 mm)
    gt_shifted = np.zeros((8, 512, 512), dtype=np.uint8)
    gt_shifted[2:6, 120:200, 170:250] = 1
    res_shift = evaluate_mask_on_orig_grid(pred_256, gt_shifted, orig_spacing_xyz_mm=(0.5, 0.5, 5.0))
    assert 0.80 < res_shift["dice_on_orig_grid"] < 0.95
    assert np.isclose(res_shift["hd95_mm_on_orig_grid"], 4.0, atol=0.05)
