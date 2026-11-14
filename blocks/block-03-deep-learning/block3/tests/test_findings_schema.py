"""Tests for the findings.json validator. Data are synthetic, not model results."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from validate_output import validate_findings  # noqa: E402

VALID = {
    "schema_version": "1.0", "case_id": "synthetic-case", "status": "ok",
    "created_at_utc": "2026-01-01T00:00:00Z", "source_artifact_ids": [],
    "model_name": "synthetic", "model_version": "v0", "weights_hash": "sha256:0", "config_hash": "sha256:0",
    "input_space_id": "synthetic-case:block2:v1",
    "presence": {"label": "positive", "score": 0.9, "threshold": 0.5},
    "subtypes": {k: {"score": 0.1, "present": False} for k in ["EDH", "SDH", "SAH", "IPH", "IVH"]},
    "masks": {"ich": {"artifact": "mask_ich.nii.gz", "space_id": "synthetic-case:block2:v1"}, "subtype": None},
    "localization": {"status": "mask-derived", "side": "right", "regions": [], "coordinate_system": "RAS+"},
    "heatmaps": [], "warnings": [], "errors": [],
}


def make(**changes):
    doc = copy.deepcopy(VALID)
    doc.update(changes)
    return doc


def test_valid_document_passes():
    assert validate_findings(VALID) == []


def test_uncertain_label_allowed():
    assert validate_findings(make(presence={"label": "uncertain", "score": 0.5, "threshold": 0.5})) == []


def test_bad_presence_label_rejected():
    assert validate_findings(make(presence={"label": "maybe", "score": 0.5, "threshold": 0.5}))


def test_score_out_of_range_rejected():
    doc = make()
    doc["subtypes"]["SDH"]["score"] = 1.5
    assert validate_findings(doc)


def test_missing_subtype_rejected():
    doc = make()
    del doc["subtypes"]["IVH"]
    assert validate_findings(doc)


def test_unknown_subtype_code_rejected():
    doc = make()
    doc["subtypes"]["XYZ"] = {"score": 0.1, "present": False}
    assert validate_findings(doc)


def test_extra_top_level_field_rejected():
    assert validate_findings(make(extent_bbox=[0, 0, 0, 1, 1, 1]))


def test_partial_requires_warning():
    assert validate_findings(make(status="partial"))
    assert validate_findings(make(status="partial", warnings=["w"])) == []


def test_failed_requires_error():
    assert validate_findings(make(status="failed"))
    assert validate_findings(make(status="failed", errors=[{"code": "E1", "message": "x"}])) == []


def test_missing_subtype_mask_must_be_null_not_absent():
    doc = make()
    del doc["masks"]["subtype"]
    assert validate_findings(doc)


def test_cli_smoke(tmp_path):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(VALID), encoding="utf-8")
    bad.write_text(json.dumps(make(status="nope")), encoding="utf-8")
    run = lambda p: subprocess.run([sys.executable, str(SRC / "validate_output.py"), str(p)],
                                   capture_output=True, text=True)
    assert run(good).returncode == 0
    assert run(bad).returncode == 1
