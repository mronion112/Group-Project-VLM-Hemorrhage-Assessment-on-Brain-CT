#!/usr/bin/env python3
"""Check that the per-case contract artifacts agree with each other."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

ARTIFACT_ORDER = ("preprocessing", "findings", "volume", "report")
SEVERITY = {"ok": 0, "partial": 1, "failed": 2}
EVIDENCE_RE = re.compile(r"^(?P<file>[A-Za-z0-9_.-]+\.json)(?:#(?P<pointer>/.*)?)?$")
TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,9})?)?(Z|[+-]\d{2}:\d{2})$"
)
# Absolute floor and relative tolerance for the voxel-volume arithmetic.
VOLUME_TOLERANCE_ML = 0.01


def load_artifact(path: Path) -> object:
    """Load one artifact file, raising ValueError with a readable message."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path.name}: {error}") from error


def load_artifacts(case_dir: Path) -> dict[str, dict]:
    """Load every known artifact present in the case directory."""
    loaded: dict[str, dict] = {}
    for name in ARTIFACT_ORDER:
        path = case_dir / f"{name}.json"
        if path.is_file():
            loaded[name] = load_artifact(path)
    return loaded


def iter_cases(cases_root: Path) -> Iterator[Path]:
    """Yield the case directories below cases_root, skipping hidden entries."""
    for path in sorted(cases_root.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            yield path


def resolve_pointer(document: object, pointer: str) -> tuple[bool, object]:
    """Resolve a JSON pointer and return whether it exists plus the node."""
    if pointer == "":
        return True, document
    if not pointer.startswith("/"):
        return False, None
    node = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            if token not in node:
                return False, None
            node = node[token]
        elif isinstance(node, list):
            if not token.isdigit() or int(token) >= len(node):
                return False, None
            node = node[int(token)]
        else:
            return False, None
    return True, node


def compare_claim_value(claim_value: object, node_value: object) -> str | None:
    """Compare a claim value with its evidence node; return a mismatch detail."""
    # Structured evidence nodes carry more than the claim states, so an exact
    # comparison would report noise. Only scalar evidence is comparable.
    if isinstance(node_value, (dict, list)) or isinstance(claim_value, (dict, list)):
        return None
    if isinstance(node_value, bool) or isinstance(claim_value, bool):
        return None if claim_value == node_value else repr(claim_value)
    if isinstance(node_value, (int, float)) and isinstance(claim_value, (int, float)):
        tolerance = max(1e-6, 1e-6 * abs(float(node_value)))
        if abs(float(claim_value) - float(node_value)) <= tolerance:
            return None
        return f"{claim_value!r} (evidence: {node_value!r})"
    if isinstance(node_value, str) and isinstance(claim_value, str):
        return None if claim_value == node_value else f"{claim_value!r} (evidence: {node_value!r})"
    return f"{claim_value!r} (evidence: {node_value!r})"


def check_common(artifacts: dict[str, dict]) -> list[str]:
    """Check case_id agreement and non-empty status/warning rules."""
    failures: list[str] = []
    case_ids = {
        name: artifact.get("case_id") for name, artifact in artifacts.items()
    }
    distinct = {value for value in case_ids.values() if value is not None}
    if len(distinct) > 1:
        detail = ", ".join(f"{name}={value}" for name, value in case_ids.items())
        failures.append(f"case_id mismatch across artifacts: {detail}")

    for name, artifact in artifacts.items():
        status = artifact.get("status")
        if status not in SEVERITY:
            failures.append(f"{name}.json: unknown status {status!r}")
            continue
        if status == "partial" and not artifact.get("warnings"):
            failures.append(f"{name}.json: status partial requires non-empty warnings")
        if status == "failed" and not artifact.get("errors"):
            failures.append(f"{name}.json: status failed requires non-empty errors")
        failures.extend(check_provenance(name, artifact))

    return failures


def check_provenance(name: str, artifact: dict) -> list[str]:
    """Check the provenance fields required by the shared contract."""
    failures: list[str] = []
    created_at = artifact.get("created_at_utc")
    if not isinstance(created_at, str):
        failures.append(f"{name}.json: missing created_at_utc")
    elif not TIMESTAMP_RE.match(created_at):
        failures.append(f"{name}.json: created_at_utc is not an ISO-8601 UTC timestamp")
    else:
        try:
            datetime.fromisoformat(created_at)
        except ValueError:
            failures.append(f"{name}.json: created_at_utc is not a valid timestamp")
    if not isinstance(artifact.get("source_artifact_ids"), list):
        failures.append(f"{name}.json: missing source_artifact_ids array")
    return failures


def check_space_ids(artifacts: dict[str, dict]) -> list[str]:
    """Check that every artifact in the chain names one image space."""
    values = {
        "preprocessing.space_id": artifacts.get("preprocessing", {}).get("space_id"),
        "findings.input_space_id": artifacts.get("findings", {}).get("input_space_id"),
        "volume.space_id": artifacts.get("volume", {}).get("space_id"),
    }
    present = {label: value for label, value in values.items() if value is not None}
    if len(set(present.values())) > 1:
        detail = ", ".join(f"{label}={value}" for label, value in present.items())
        return [f"space_id mismatch: {detail}"]
    return []


def check_mask_spaces(artifacts: dict[str, dict]) -> list[str]:
    """Check that every mask in findings.json lives in the input image space."""
    findings = artifacts.get("findings")
    if not findings:
        return []

    expected = findings.get("input_space_id")
    if expected is None:
        return []

    failures: list[str] = []
    masks = findings.get("masks") or {}
    for label, mask in sorted(masks.items()):
        if not isinstance(mask, dict):
            continue
        space_id = mask.get("space_id")
        if space_id is not None and space_id != expected:
            failures.append(
                f"findings.masks.{label}.space_id {space_id!r} does not match "
                f"input_space_id {expected!r}"
            )
    return failures


def check_masks(artifacts: dict[str, dict]) -> list[str]:
    """Check source_mask and by_subtype_ml consistency with findings masks."""
    failures: list[str] = []
    findings = artifacts.get("findings")
    volume = artifacts.get("volume")
    if not findings or not volume:
        return failures

    masks = findings.get("masks") or {}
    declared = {
        mask.get("artifact")
        for mask in masks.values()
        if isinstance(mask, dict) and mask.get("artifact")
    }
    source_mask = volume.get("source_mask")
    if source_mask not in declared:
        failures.append(
            f"volume.source_mask {source_mask!r} is not declared in findings.masks"
        )

    # A subtype breakdown is only meaningful when the subtype mask exists, so a
    # breakdown without that mask is rejected. A subtype mask without a
    # breakdown stays valid: Block 4 may report only the total ICH volume.
    subtype_volume = volume.get("by_subtype_ml")
    if masks.get("subtype") is None and subtype_volume is not None:
        failures.append(
            "volume.by_subtype_ml must be null when findings.masks.subtype is null"
        )
    return failures


def check_volume_math(artifacts: dict[str, dict]) -> list[str]:
    """Check that volume_ml follows from voxel_count and physical spacing."""
    volume = artifacts.get("volume")
    if not volume:
        return []

    voxel_count = volume.get("voxel_count")
    spacing = volume.get("voxel_spacing_mm_xyz")
    volume_ml = volume.get("volume_ml")
    if not isinstance(voxel_count, int) or not isinstance(volume_ml, (int, float)):
        return []
    if not isinstance(spacing, list) or len(spacing) != 3:
        return []

    expected_ml = voxel_count * math.prod(float(value) for value in spacing) / 1000.0
    tolerance = max(VOLUME_TOLERANCE_ML, VOLUME_TOLERANCE_ML * expected_ml)
    if abs(float(volume_ml) - expected_ml) > tolerance:
        return [
            (
                f"volume.json: volume_ml {volume_ml!r} does not match "
                f"voxel_count x voxel volume ({expected_ml:.4f} mL)"
            )
        ]
    return []


def check_status_order(artifacts: dict[str, dict]) -> list[str]:
    """Check that severity never decreases downstream."""
    failures: list[str] = []
    previous_name: str | None = None
    previous_severity = 0
    for name in ARTIFACT_ORDER:
        artifact = artifacts.get(name)
        if not artifact or artifact.get("status") not in SEVERITY:
            continue
        severity = SEVERITY[artifact["status"]]
        if previous_name is not None and severity < previous_severity:
            failures.append(
                f"{name}.json: status {artifact['status']!r} is less severe than "
                f"upstream {previous_name}.json"
            )
        previous_name = name
        previous_severity = max(previous_severity, severity)
    return failures


def resolve_evidence(
    evidence: object, documents: dict[str, dict], case_dir: Path
) -> tuple[bool, object, str | None]:
    """Resolve one evidence reference to a node; return found, node and error."""
    match = EVIDENCE_RE.match(str(evidence))
    if not match:
        return False, None, f"unsupported evidence reference {evidence!r}"

    artifact_name = match.group("file")[: -len(".json")]
    if artifact_name not in documents:
        path = case_dir / f"{artifact_name}.json"
        if not path.is_file():
            return False, None, f"evidence {evidence!r} references a missing artifact"
        try:
            documents[artifact_name] = load_artifact(path)
        except ValueError as error:
            return False, None, str(error)

    found, node = resolve_pointer(documents[artifact_name], match.group("pointer") or "")
    if not found:
        return False, None, f"dangling evidence pointer {evidence!r}"
    return True, node, None


def check_report(artifacts: dict[str, dict], case_dir: Path) -> list[str]:
    """Check report claims, evidence pointers and unsupported claims."""
    report = artifacts.get("report")
    if not report:
        return []

    failures: list[str] = []
    documents = dict(artifacts)
    for claim in report.get("claims") or []:
        for evidence in claim.get("evidence") or []:
            _, node, error = resolve_evidence(evidence, documents, case_dir)
            if error:
                failures.append(f"report.json: {error}")
                continue
            mismatch = compare_claim_value(claim.get("value"), node)
            if mismatch:
                failures.append(
                    f"report.json: claim {claim.get('field')!r} value {mismatch} "
                    f"does not match {evidence}"
                )

    if report.get("status") == "ok" and report.get("unsupported_claims"):
        failures.append("report.json: status ok requires empty unsupported_claims")
    return failures


def collect_failures(case_dir: Path) -> list[str]:
    """Return every contract violation found in the case directory."""
    try:
        artifacts = load_artifacts(case_dir)
    except ValueError as error:
        return [str(error)]

    if not artifacts:
        return [f"no contract artifacts found in {case_dir}"]

    failures: list[str] = []
    failures.extend(check_common(artifacts))
    failures.extend(check_space_ids(artifacts))
    failures.extend(check_mask_spaces(artifacts))
    failures.extend(check_masks(artifacts))
    failures.extend(check_volume_math(artifacts))
    failures.extend(check_status_order(artifacts))
    failures.extend(check_report(artifacts, case_dir))
    return failures


def collect_cases_failures(cases_root: Path) -> tuple[int, list[str]]:
    """Validate every case directory below cases_root; return count and failures."""
    case_dirs = list(iter_cases(cases_root))
    failures: list[str] = []
    for case_dir in case_dirs:
        failures.extend(f"{case_dir.name}/{failure}" for failure in collect_failures(case_dir))
    return len(case_dirs), failures


def main(argv: list[str] | None = None) -> int:
    """Run the case artifact check and print one line per failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--case-dir",
        type=Path,
        help="directory containing any subset of the per-case artifacts",
    )
    target.add_argument(
        "--cases-root",
        type=Path,
        help="directory whose subdirectories are per-case artifact directories",
    )
    args = parser.parse_args(argv)

    if args.case_dir is not None:
        case_dir = args.case_dir.resolve()
        if not case_dir.is_dir():
            print(f"FAIL: case directory does not exist: {args.case_dir}")
            return 1
        failures = collect_failures(case_dir)
        success = "OK: case artifacts"
    else:
        cases_root = args.cases_root.resolve()
        if not cases_root.is_dir():
            print(f"FAIL: cases root does not exist: {args.cases_root}")
            return 1
        count, failures = collect_cases_failures(cases_root)
        success = f"OK: case artifacts ({count} cases)"

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print(success)
    return 0


if __name__ == "__main__":
    sys.exit(main())
