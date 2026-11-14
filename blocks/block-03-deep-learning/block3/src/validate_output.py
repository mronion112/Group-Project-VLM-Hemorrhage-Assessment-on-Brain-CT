"""Validate a findings.json against the Block 3 schema (contract v1.0)."""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "findings.schema.json"


def validate_findings(findings: dict) -> list[str]:
    """Return a list of human-readable schema errors (empty list = valid)."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in sorted(validator.iter_errors(findings), key=lambda e: list(e.absolute_path))]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python validate_output.py <findings.json>")
        return 2
    errors = validate_findings(json.loads(Path(argv[1]).read_text(encoding="utf-8")))
    for line in errors:
        print("INVALID:", line)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
