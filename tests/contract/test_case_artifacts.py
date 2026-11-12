"""Cross-artifact consistency tests for the per-case contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import check_case_artifacts

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def copy_case(tmp_path: Path, name: str) -> Path:
    """Copy a fixture case into tmp_path and return the copy."""
    target = tmp_path / name
    shutil.copytree(FIXTURES_DIR / name, target)
    return target


def edit(case_dir: Path, artifact: str, mutate) -> None:
    """Apply a mutation to one artifact file inside a copied case."""
    path = case_dir / f"{artifact}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def test_ok_case_passes() -> None:
    assert check_case_artifacts.collect_failures(FIXTURES_DIR / "case-ok") == []


def test_partial_case_passes() -> None:
    assert check_case_artifacts.collect_failures(FIXTURES_DIR / "case-partial") == []


def test_space_id_mismatch_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")
    edit(case_dir, "volume", lambda doc: doc.update({"space_id": "case-synthetic-0001:block3:v1"}))

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("space_id mismatch" in failure for failure in failures)


def test_case_id_mismatch_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")
    edit(case_dir, "volume", lambda doc: doc.update({"case_id": "case-synthetic-9999"}))

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("case_id mismatch" in failure for failure in failures)


def test_subtype_volume_without_subtype_mask_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")
    edit(case_dir, "findings", lambda doc: doc["masks"].update({"subtype": None}))

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("by_subtype_ml must be null" in failure for failure in failures)


def test_status_downgrade_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-partial")
    edit(case_dir, "findings", lambda doc: doc.update({"status": "ok", "warnings": []}))

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("less severe than upstream" in failure for failure in failures)


def test_empty_warnings_on_partial_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-partial")
    edit(case_dir, "preprocessing", lambda doc: doc.update({"warnings": []}))

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("status partial requires non-empty warnings" in failure for failure in failures)


def test_failed_without_errors_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-partial")
    edit(case_dir, "volume", lambda doc: doc.update({"status": "failed", "errors": []}))

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("status failed requires non-empty errors" in failure for failure in failures)


def test_dangling_evidence_pointer_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")

    def break_pointer(doc: dict) -> None:
        doc["claims"].append(
            {"field": "voxel_count", "value": 1200, "evidence": ["volume.json#/missing_node"]}
        )

    edit(case_dir, "report", break_pointer)

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("dangling evidence pointer" in failure for failure in failures)


def test_evidence_to_missing_artifact_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")
    (case_dir / "volume.json").unlink()

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("references a missing artifact" in failure for failure in failures)


def test_source_mask_not_declared_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")
    edit(case_dir, "volume", lambda doc: doc.update({"source_mask": "mask_other.nii.gz"}))

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("is not declared in findings.masks" in failure for failure in failures)


def test_unsupported_claims_with_ok_status_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")
    edit(case_dir, "report", lambda doc: doc.update({"unsupported_claims": ["subtype"]}))

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("requires empty unsupported_claims" in failure for failure in failures)


def test_missing_provenance_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")
    edit(case_dir, "report", lambda doc: doc.pop("created_at_utc"))

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("missing created_at_utc" in failure for failure in failures)


def test_empty_case_dir_fails(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    failures = check_case_artifacts.collect_failures(empty)

    assert any("no contract artifacts found" in failure for failure in failures)


def test_mask_space_id_mismatch_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")
    edit(
        case_dir,
        "findings",
        lambda doc: doc["masks"]["ich"].update({"space_id": "case-synthetic-0001:other:v9"}),
    )

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("masks.ich.space_id" in failure for failure in failures)
    assert any("does not match input_space_id" in failure for failure in failures)


def test_claim_value_mismatch_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")

    def change_volume_claim(doc: dict) -> None:
        for claim in doc["claims"]:
            if claim["field"] == "volume_ml":
                claim["value"] = 99.0

    edit(case_dir, "report", change_volume_claim)

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("does not match volume.json#/volume_ml" in failure for failure in failures)


def test_volume_arithmetic_mismatch_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")
    edit(case_dir, "volume", lambda doc: doc.update({"volume_ml": 99.0}))

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("does not match voxel_count x voxel volume" in failure for failure in failures)


def test_subtype_mask_without_breakdown_passes(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")
    edit(case_dir, "volume", lambda doc: doc.update({"by_subtype_ml": None}))

    failures = check_case_artifacts.collect_failures(case_dir)

    assert not any("by_subtype_ml" in failure for failure in failures)


def test_malformed_referenced_artifact_reports_failure(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")
    (case_dir / "extra.json").write_text("{ broken", encoding="utf-8")

    def cite_extra(doc: dict) -> None:
        doc["claims"].append({"field": "x", "value": 1, "evidence": ["extra.json#/a"]})

    edit(case_dir, "report", cite_extra)

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("invalid JSON in extra.json" in failure for failure in failures)


def test_evidence_without_pointer_is_validated(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")

    def cite_missing_document(doc: dict) -> None:
        doc["claims"].append({"field": "qc", "value": 1, "evidence": ["preprocessing.json"]})

    edit(case_dir, "report", cite_missing_document)
    (case_dir / "preprocessing.json").unlink()

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("references a missing artifact" in failure for failure in failures)


def test_unsupported_evidence_reference_fails(tmp_path: Path) -> None:
    case_dir = copy_case(tmp_path, "case-ok")

    def cite_dicom(doc: dict) -> None:
        doc["claims"].append({"field": "x", "value": 1, "evidence": ["dicom://study/1"]})

    edit(case_dir, "report", cite_dicom)

    failures = check_case_artifacts.collect_failures(case_dir)

    assert any("unsupported evidence reference" in failure for failure in failures)


def test_cases_root_validates_every_case(tmp_path: Path) -> None:
    cases_root = tmp_path / "cases"
    cases_root.mkdir()
    shutil.copytree(FIXTURES_DIR / "case-ok", cases_root / "case-ok")
    shutil.copytree(FIXTURES_DIR / "case-partial", cases_root / "case-partial")
    (cases_root / "README.md").write_text("# cases\n", encoding="utf-8")

    count, failures = check_case_artifacts.collect_cases_failures(cases_root)

    assert count == 2
    assert failures == []


def test_cases_root_prefixes_failures(tmp_path: Path) -> None:
    cases_root = tmp_path / "cases"
    cases_root.mkdir()
    shutil.copytree(FIXTURES_DIR / "case-ok", cases_root / "broken")
    edit(cases_root / "broken", "volume", lambda doc: doc.update({"volume_ml": 99.0}))

    _, failures = check_case_artifacts.collect_cases_failures(cases_root)

    assert failures
    assert all(failure.startswith("broken/") for failure in failures)
