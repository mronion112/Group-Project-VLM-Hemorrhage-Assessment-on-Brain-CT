#!/usr/bin/env python3
"""Validate contract examples in the docs and single artifacts against schemas."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import jsonschema

HEADING_RE = re.compile(r"^###\s+(?P<name>[A-Za-z0-9_.-]+\.json)\b")
FENCE_OPEN_RE = re.compile(r"^(?P<fence>~~~|```)json\s*$")
DEFAULT_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_DOC = Path("docs/INTERFACE_CONTRACT.md")
SCHEMA_DIR = Path("schemas")
VALIDATOR_CLASS = jsonschema.Draft202012Validator


def schema_path(root: Path, artifact_name: str) -> Path:
    """Map an artifact file name such as findings.json to its schema file."""
    stem = artifact_name.removesuffix(".json").removesuffix(".schema")
    return root / SCHEMA_DIR / f"{stem}.schema.json"


def load_json(path: Path) -> object:
    """Load a JSON document, raising ValueError with the file name on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error


def validate_document(document: object, schema: object, label: str) -> list[str]:
    """Return readable failures for one document and one schema."""
    validator = VALIDATOR_CLASS(schema)
    failures: list[str] = []
    for error in sorted(validator.iter_errors(document), key=lambda item: list(item.path)):
        location = "/".join(str(part) for part in error.path) or "<root>"
        failures.append(f"{label}: {location}: {error.message}")
    return failures


def extract_examples(doc_path: Path) -> tuple[dict[str, object], list[str]]:
    """Return the json blocks that follow an artifact heading, keyed by file name."""
    examples: dict[str, object] = {}
    failures: list[str] = []
    current_name: str | None = None
    fence = ""
    buffer: list[str] = []

    for line in doc_path.read_text(encoding="utf-8").splitlines():
        if fence:
            if line.strip() == fence:
                try:
                    examples[current_name] = json.loads("\n".join(buffer))
                except json.JSONDecodeError as error:
                    failures.append(f"documented example {current_name} is not valid JSON: {error}")
                current_name = None
                buffer = []
                fence = ""
            else:
                buffer.append(line)
            continue

        heading = HEADING_RE.match(line)
        if heading:
            current_name = heading.group("name")
            continue

        # Any other heading closes the artifact section, so a later JSON block
        # is not attributed to the previous artifact.
        if line.startswith("#"):
            current_name = None
            continue

        opening = FENCE_OPEN_RE.match(line)
        if opening and current_name is not None:
            fence = opening.group("fence")

    return examples, failures


def run_docs(root: Path) -> tuple[int, list[str]]:
    """Validate every documented example and its schema coverage."""
    doc_path = root / CONTRACT_DOC
    if not doc_path.is_file():
        return 0, [f"missing contract document: {CONTRACT_DOC.as_posix()}"]

    examples, failures = extract_examples(doc_path)
    schemas = {
        path.name[: -len(".schema.json")]: path
        for path in sorted((root / SCHEMA_DIR).glob("*.schema.json"))
    }

    for name, document in examples.items():
        target = schema_path(root, name)
        if not target.is_file():
            failures.append(f"documented artifact without schema: {name}")
            continue
        schema = load_json(target)
        failures.extend(validate_document(document, schema, name))

    documented = {name[: -len(".json")] for name in examples}
    for stem in schemas:
        if stem not in documented:
            failures.append(f"schema without documented example: schemas/{stem}.schema.json")

    return len(examples), failures


def run_artifact(root: Path, artifact_name: str, path: Path) -> tuple[int, list[str]]:
    """Validate one artifact file against its schema."""
    target = schema_path(root, artifact_name)
    if not target.is_file():
        return 0, [f"missing schema: {target.relative_to(root).as_posix()}"]
    if not path.is_file():
        return 0, [f"missing artifact: {path}"]
    schema = load_json(target)
    document = load_json(path)
    return 1, validate_document(document, schema, path.as_posix())


def main(argv: list[str] | None = None) -> int:
    """Run the docs contract check or a single artifact check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="repository root (default: parent of this script's directory)",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("docs", help="validate documented examples (default)")
    artifact_parser = subparsers.add_parser("artifact", help="validate one artifact file")
    artifact_parser.add_argument("schema_name")
    artifact_parser.add_argument("path", type=Path)

    args = parser.parse_args(argv)
    root = args.root.resolve()

    try:
        if args.command == "artifact":
            checked, failures = run_artifact(root, args.schema_name, args.path)
            success = f"OK: artifact ({args.path})"
        else:
            checked, failures = run_docs(root)
            success = f"OK: contract docs ({checked} examples)"
    except ValueError as error:
        print(f"FAIL: {error}")
        return 1

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print(success)
    return 0


if __name__ == "__main__":
    sys.exit(main())
