#!/usr/bin/env python3
"""Check required documentation paths and reject imaging data in the repo."""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from collections.abc import Iterator
from pathlib import Path

# Hidden tooling directories stay out of the scan; .github is content and is kept.
VISIBLE_HIDDEN_DIRS = frozenset({".github"})
IGNORED_DIR_NAMES = frozenset({"node_modules", "venv", "__pycache__"})

REQUIRED_PATHS = (
    "README.md",
    "docs/INTERFACE_CONTRACT.md",
    "docs/OWNERSHIP.md",
    "docs/CI.md",
    "docs/INTEGRATION_STATUS.md",
    "contract-cases/README.md",
    "blocks/README.md",
    "blocks/block-01-dataset-split/Guide.md",
    "blocks/block-01-dataset-split/Report.md",
    "blocks/block-02-preprocessing/Guide.md",
    "blocks/block-02-preprocessing/Report.md",
    "blocks/block-03-deep-learning/Guide.md",
    "blocks/block-03-deep-learning/Report.md",
    "blocks/block-04-volume/Guide.md",
    "blocks/block-04-volume/Report.md",
    "blocks/block-05-vlm/Guide.md",
    "blocks/block-05-vlm/Report.md",
    "blocks/block-06-feedback-ledger/README.md",
)

FORBIDDEN_FILE_PATTERNS = (
    "*.nii",
    "*.nii.gz",
    "*.dcm",
    "*.dicom",
    "*.nrrd",
    "*.npy",
    "*.npz",
    "*.mha",
    "*.mhd",
)

FORBIDDEN_DIR_NAMES = frozenset({"data", "datasets", "raw"})

DATA_POLICY = "the repo stores references/manifests, not imaging data"


def is_ignored_dir(name: str) -> bool:
    """Report whether a directory name belongs to local tooling state."""
    if name in VISIBLE_HIDDEN_DIRS:
        return False
    return name.startswith(".") or name in IGNORED_DIR_NAMES


def iter_entries(root: Path) -> Iterator[Path]:
    """Yield every file and directory below root, pruning ignored trees."""
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not is_ignored_dir(name))
        base = Path(current)
        for name in dirs:
            yield base / name
        for name in sorted(files):
            yield base / name


def collect_failures(root: Path) -> list[str]:
    """Return one human-readable message per structure violation."""
    failures: list[str] = []

    for relative in REQUIRED_PATHS:
        if not (root / relative).is_file():
            failures.append(f"missing required path: {relative}")

    for path in iter_entries(root):
        relative = path.relative_to(root)
        name = path.name
        if path.is_dir():
            if name.lower() in FORBIDDEN_DIR_NAMES:
                failures.append(f"forbidden data directory {relative}/: {DATA_POLICY}")
            continue
        if any(fnmatch.fnmatch(name.lower(), pattern) for pattern in FORBIDDEN_FILE_PATTERNS):
            failures.append(f"forbidden imaging artifact {relative}: {DATA_POLICY}")

    return failures


def main(argv: list[str] | None = None) -> int:
    """Run the structure check and print one line per violation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (default: parent of this script's directory)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"FAIL: root directory does not exist: {args.root}")
        return 1

    failures = collect_failures(root)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print("OK: structure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
