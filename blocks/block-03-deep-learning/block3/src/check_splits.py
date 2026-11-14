"""Validate patient-level split CSVs against Block 1/3 contract rules (zero patient overlap)."""
import csv
import sys
from pathlib import Path

REQUIRED_COLUMNS = (
    "case_id",
    "patient_group_id",
    "dataset_id",
    "dataset_release",
    "source_uri",
    "split",
    "labels_available",
    "label_scope",
)
VALID_SPLITS = {"train", "validation", "test", "external"}


def validate_split_rows(rows_by_split: dict[str, list[dict]]) -> list[str]:
    """Return a list of validation errors (empty list = clean splits)."""
    errors: list[str] = []
    patients_by_split: dict[str, set[str]] = {}
    cases_seen: dict[str, str] = {}

    for split_name, rows in rows_by_split.items():
        if split_name not in VALID_SPLITS:
            errors.append(f"invalid split name: {split_name}")
        patients_by_split[split_name] = set()
        if not rows:
            errors.append(f"{split_name}: empty split")
            continue
        for idx, row in enumerate(rows):
            missing = [c for c in REQUIRED_COLUMNS if not str(row.get(c, "")).strip()]
            if missing:
                errors.append(f"{split_name}[{idx}]: missing columns {missing}")
                continue
            if row["split"] != split_name:
                errors.append(f"{split_name}[{idx}]: row split='{row['split']}' != '{split_name}'")
            pid = row["patient_group_id"].strip()
            cid = row["case_id"].strip()
            if not pid.startswith("pgrp_"):
                errors.append(f"{split_name}[{idx}]: patient_group_id '{pid}' is not pseudonymized (must start with 'pgrp_')")
            patients_by_split[split_name].add(pid)
            if cid in cases_seen and cases_seen[cid] != split_name:
                errors.append(f"duplicate case_id '{cid}' across '{cases_seen[cid]}' and '{split_name}'")
            cases_seen[cid] = split_name

    split_names = sorted(patients_by_split.keys())
    for i, s1 in enumerate(split_names):
        for s2 in split_names[i + 1:]:
            overlap = patients_by_split[s1] & patients_by_split[s2]
            if overlap:
                sample = sorted(overlap)[:5]
                errors.append(f"patient_group_id overlap between '{s1}' and '{s2}': {len(overlap)} patients (e.g. {sample})")

    return errors


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_split_files(split_paths: dict[str, Path]) -> dict:
    rows_by_split = {name: load_csv(path) for name, path in split_paths.items()}
    errors = validate_split_rows(rows_by_split)
    counts = {
        name: {
            "cases": len(rows),
            "patients": len({r.get("patient_group_id", "") for r in rows}),
        }
        for name, rows in rows_by_split.items()
    }
    return {"status": "passed" if not errors else "failed", "counts": counts, "errors": errors}


def main(argv: list[str]) -> int:
    splits_dir = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "splits"
    paths = {s: splits_dir / f"{s}.csv" for s in ("train", "validation", "test") if (splits_dir / f"{s}.csv").exists()}
    if not paths:
        print(f"No split CSVs found in {splits_dir}")
        return 2
    report = check_split_files(paths)
    print(report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
