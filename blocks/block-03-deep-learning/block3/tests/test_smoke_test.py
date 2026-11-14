"""Smoke test for block3/src/smoke_test.py CLI, synthetic end-to-end pipeline, and checkpoint/memory helpers."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smoke_test import (  # noqa: E402
    compute_mask_side_and_extent,
    main,
    measure_downsampling_effect,
)


def test_measure_downsampling_effect_returns_valid_metrics():
    res = measure_downsampling_effect()
    assert "synthetic_medium_r16px_7.8mm" in res
    assert res["synthetic_medium_r16px_7.8mm"]["dice_on_orig_grid"] > 0.90
    assert res["synthetic_small_r4px_1.95mm"]["dice_on_orig_grid"] > 0.70


def test_smoke_test_cli_runs_end_to_end(tmp_path):
    out_json = tmp_path / "smoke_out.json"
    rc = main(["--max-slices", "4", "--out", str(out_json)])
    assert rc == 0 and out_json.exists()
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert "mobilenet_v3_small" in report["classification_2d_smoke"]
    assert "resnet18" in report["classification_2d_smoke"]
    assert report["process_memory_mb"]["peak_rss_mb"] > 0
    assert report["segmentation_2d_unet_smoke"]["orig_grid_eval_check"]["evaluated_shape_zyx"][1:] == [512, 512]


def test_compute_mask_side_and_extent_ras_coordinates():
    affine = np.array(
        [
            [0.5, 0.0, 0.0, -32.0],
            [0.0, 0.5, 0.0, -32.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    empty = np.zeros((4, 128, 128), dtype=np.uint8)
    assert compute_mask_side_and_extent(empty, affine)["status"] == "missing"

    right_mask = np.zeros((4, 128, 128), dtype=np.uint8)
    right_mask[1:3, 40:60, 80:100] = 1
    loc_r = compute_mask_side_and_extent(right_mask, affine)
    assert loc_r["status"] == "mask-derived"
    assert loc_r["side"] == "right"
    assert loc_r["extent_zyx"] == [1, 40, 80, 2, 59, 99]

    left_mask = np.zeros((4, 128, 128), dtype=np.uint8)
    left_mask[1:3, 40:60, 20:40] = 1
    assert compute_mask_side_and_extent(left_mask, affine)["side"] == "left"

    bilat_mask = right_mask | left_mask
    assert compute_mask_side_and_extent(bilat_mask, affine)["side"] == "bilateral"


def test_smoke_test_synthetic_pipeline_end_to_end(tmp_path):
    out_json = tmp_path / "smoke_synth.json"
    rc = main(["--synthetic", "--num-slices", "4", "--out", str(out_json)])
    assert rc == 0 and out_json.exists()
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["mode"] == "synthetic_end_to_end"
    assert report["requires_external_datasets"] is False
    assert report["npz_verified"]["shape_zyx"] == [4, 256, 256]
    assert report["segmentation_orig_grid_check"]["nifti_roundtrip_ok"] is True
    assert report["segmentation_orig_grid_check"]["nifti_shape_xyz"] == [512, 512, 4]
    assert report["gt_localization_check"]["side"] == "right"
    assert report["findings_schema_errors"] == []
    assert report["synthetic_splits_check"] == "passed"
