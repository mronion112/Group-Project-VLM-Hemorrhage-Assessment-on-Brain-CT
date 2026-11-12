"""Schema conformance tests for the shared contract artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import validate_contract
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ARTIFACTS = ("preprocessing", "findings", "volume", "report")


def load(path: Path) -> dict:
    """Load a JSON document from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_schema(name: str) -> dict:
    """Load the schema for an artifact name."""
    return load(REPO_ROOT / "schemas" / f"{name}.schema.json")


def schema_errors(document: dict, name: str) -> list[str]:
    """Return validator error messages for one document."""
    validator = Draft202012Validator(load_schema(name))
    return [error.message for error in validator.iter_errors(document)]


@pytest.mark.parametrize("case_name", ["case-ok", "case-partial"])
@pytest.mark.parametrize("name", ARTIFACTS)
def test_fixture_matches_schema(case_name: str, name: str) -> None:
    document = load(FIXTURES_DIR / case_name / f"{name}.json")

    assert schema_errors(document, name) == []


def test_documented_examples_validate() -> None:
    examples, failures = validate_contract.extract_examples(
        REPO_ROOT / "docs" / "INTERFACE_CONTRACT.md"
    )

    assert failures == []
    assert len(examples) == 5
    for file_name, document in examples.items():
        name = file_name[: -len(".json")]
        assert schema_errors(document, name) == []


def test_unknown_subtype_key_rejected() -> None:
    document = copy.deepcopy(load(FIXTURES_DIR / "case-ok" / "findings.json"))
    document["subtypes"]["EPH"] = document["subtypes"].pop("EDH")

    assert schema_errors(document, "findings")


def test_negative_volume_rejected() -> None:
    document = copy.deepcopy(load(FIXTURES_DIR / "case-ok" / "volume.json"))
    document["volume_ml"] = -1.0

    assert schema_errors(document, "volume")


def test_claim_without_evidence_rejected() -> None:
    document = copy.deepcopy(load(FIXTURES_DIR / "case-ok" / "report.json"))
    document["claims"][0].pop("evidence")

    assert schema_errors(document, "report")


def test_score_out_of_range_rejected() -> None:
    document = copy.deepcopy(load(FIXTURES_DIR / "case-ok" / "findings.json"))
    document["presence"]["score"] = 1.4

    assert schema_errors(document, "findings")


def test_unknown_status_rejected() -> None:
    document = copy.deepcopy(load(FIXTURES_DIR / "case-ok" / "preprocessing.json"))
    document["status"] = "unknown"

    assert schema_errors(document, "preprocessing")


def test_extra_field_rejected() -> None:
    document = copy.deepcopy(load(FIXTURES_DIR / "case-ok" / "volume.json"))
    document["unexpected_field"] = 1

    assert schema_errors(document, "volume")


def test_unknown_by_subtype_key_rejected() -> None:
    document = copy.deepcopy(load(FIXTURES_DIR / "case-ok" / "volume.json"))
    document["by_subtype_ml"]["EPH"] = 0.1

    assert schema_errors(document, "volume")
