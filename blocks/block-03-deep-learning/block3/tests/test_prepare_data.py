"""Unit tests for prepare_data.py (synthetic volumes and storage checks)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_data import (  # noqa: E402
    check_storage_budget,
    resize_volume_2d,
    restore_mask_to_orig_grid,
    save_and_verify_npz,
    verify_preconverted_hu_array,
)


def test_resize_volume_preserves_dtype_and_nearest_for_mask():
    vol = np.random.randint(-1000, 1000, size=(10, 512, 512), dtype=np.int16)
    mask = np.zeros((10, 512, 512), dtype=np.uint8)
    mask[:, 100:200, 100:200] = 5  # label 5 = SDH in BHSD

    vol_256 = resize_volume_2d(vol, (256, 256), is_mask=False)
    mask_256 = resize_volume_2d(mask, (256, 256), is_mask=True)
    mask_512 = restore_mask_to_orig_grid(mask_256, (10, 512, 512))

    assert vol_256.shape == (10, 256, 256) and vol_256.dtype == np.int16
    assert mask_256.shape == (10, 256, 256) and mask_256.dtype == np.uint8
    assert mask_512.shape == (10, 512, 512) and mask_512.dtype == np.uint8
    assert set(np.unique(mask_256).tolist()).issubset({0, 5})
    assert np.array_equal(mask_512, mask)


def test_save_and_verify_npz_multiclass_and_binary_contracts(tmp_path):
    # 1. Multiclass 5-subtypes (BHSD, including 512x512 hu_orig_zyx_int16 for U2)
    npz_multi = tmp_path / "bhsd_case.npz"
    payload_multi = {
        "hu_int16": np.full((8, 256, 256), -1000, dtype=np.int16),
        "hu_orig_zyx_int16": np.full((8, 512, 512), -1000, dtype=np.int16),
        "orig_shape_zyx": np.array([8, 512, 512], dtype=np.int32),
        "mask_type": np.array("multiclass_5_subtypes"),
        "mask_ich_uint8": np.zeros((8, 256, 256), dtype=np.uint8),
        "mask_subtype_uint8": np.zeros((8, 256, 256), dtype=np.uint8),
        "mask_ich_orig_zyx_uint8": np.zeros((8, 512, 512), dtype=np.uint8),
        "mask_subtype_orig_zyx_uint8": np.zeros((8, 512, 512), dtype=np.uint8),
    }
    sha, size = save_and_verify_npz(npz_multi, payload_multi)
    assert sha.startswith("sha256:") and size > 0

    # 2. Binary ICH only (PhysioNet CT-ICH): must reject if mask_subtype_uint8 is present
    npz_bin = tmp_path / "physionet_case.npz"
    payload_bin = {
        "hu_int16": np.full((8, 256, 256), -1000, dtype=np.int16),
        "mask_type": np.array("binary_ich_only"),
        "mask_ich_uint8": np.zeros((8, 256, 256), dtype=np.uint8),
    }
    sha_b, _ = save_and_verify_npz(npz_bin, payload_bin)
    assert sha_b.startswith("sha256:")

    with pytest.raises(ValueError, match="MUST NOT contain mask_subtype"):
        save_and_verify_npz(
            tmp_path / "bad_bin.npz",
            {
                **payload_bin,
                "mask_subtype_uint8": np.zeros((8, 256, 256), dtype=np.uint8),
            },
        )


def test_verify_preconverted_hu_guards_against_double_minus_1024():
    good_hu = np.array([-1024, 0, 40, 1000], dtype=np.int16)
    verify_preconverted_hu_array(good_hu, "good")
    bad_double_offset = good_hu - 1024  # min becomes -2048, p1 around -2024
    with pytest.raises(ValueError, match="double -1024 offset"):
        verify_preconverted_hu_array(bad_double_offset, "bad")


def test_storage_budget_raises_when_min_free_exceeds_disk(tmp_path):
    with pytest.raises(RuntimeError, match="min_free_gb"):
        check_storage_budget(tmp_path, min_free_gb=999999.0, storage_budget_gb=30.0)


def test_load_bhsd_zip_map_remote_fallback(tmp_path):
    import io
    import zipfile
    from prepare_data import load_bhsd_zip_map

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("label_192/images/sample.nii.gz", b"ctbytes")
        zf.writestr("label_192/ground truths/sample.nii.gz", b"gtbytes")
    blob = buf.getvalue()
    missing_csv = tmp_path / "does_not_exist.csv"
    zmap = load_bhsd_zip_map(missing_csv, fetch=lambda s, e: blob[s:e + 1], size=len(blob))
    assert "label_192/images/sample.nii.gz" in zmap
    assert "label_192/ground truths/sample.nii.gz" in zmap
