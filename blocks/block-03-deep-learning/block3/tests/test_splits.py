"""Unit tests for patient-level split checker (synthetic data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_splits import validate_split_rows  # noqa: E402


def _row(case_id: str, pid: str, split: str) -> dict:
    return {
        "case_id": case_id,
        "patient_group_id": pid,
        "dataset_id": "synthetic",
        "dataset_release": "v1",
        "source_uri": f"synthetic://{case_id}",
        "split": split,
        "labels_available": "ICH,EDH,SDH,SAH,IPH,IVH",
        "label_scope": "slice_and_exam",
    }


def test_clean_patient_splits_pass():
    splits = {
        "train": [_row("c1", "pgrp_aaaa1111", "train"), _row("c2", "pgrp_aaaa1111", "train")],
        "validation": [_row("c3", "pgrp_bbbb2222", "validation")],
        "test": [_row("c4", "pgrp_cccc3333", "test")],
    }
    assert validate_split_rows(splits) == []


def test_patient_overlap_between_train_and_test_is_rejected():
    splits = {
        "train": [_row("c1", "pgrp_same0001", "train")],
        "validation": [_row("c2", "pgrp_bbbb2222", "validation")],
        "test": [_row("c3", "pgrp_same0001", "test")],
    }
    errors = validate_split_rows(splits)
    assert any("overlap between 'test' and 'train'" in e for e in errors)


def test_non_pseudonymized_patient_id_is_rejected():
    splits = {
        "train": [_row("c1", "ID_raw_patient", "train")],
        "validation": [_row("c2", "pgrp_bbbb2222", "validation")],
        "test": [_row("c3", "pgrp_cccc3333", "test")],
    }
    errors = validate_split_rows(splits)
    assert any("not pseudonymized" in e for e in errors)


def test_missing_required_column_is_rejected():
    bad = _row("c1", "pgrp_aaaa1111", "train")
    bad["source_uri"] = ""
    splits = {
        "train": [bad],
        "validation": [_row("c2", "pgrp_bbbb2222", "validation")],
        "test": [_row("c3", "pgrp_cccc3333", "test")],
    }
    errors = validate_split_rows(splits)
    assert any("missing columns" in e for e in errors)
