"""Unit tests for the documentation structure and link checks."""

from __future__ import annotations

import check_links
import check_structure


def build_valid_tree(root):
    """Create the minimal documentation tree the structure check accepts."""
    (root / "docs").mkdir()
    (root / "blocks").mkdir()
    (root / "blocks" / "block-06-feedback-ledger").mkdir()
    for name in ("README.md", "docs/INTERFACE_CONTRACT.md", "docs/OWNERSHIP.md",
                 "docs/CI.md", "docs/INTEGRATION_STATUS.md", "blocks/README.md",
                 "blocks/block-06-feedback-ledger/README.md",
                 "contract-cases/README.md"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
    for name in (
        "block-01-dataset-split",
        "block-02-preprocessing",
        "block-03-deep-learning",
        "block-04-volume",
        "block-05-vlm",
    ):
        block_dir = root / "blocks" / name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "Guide.md").write_text(f"# Guide {name}\n", encoding="utf-8")
        (block_dir / "Report.md").write_text(f"# Report {name}\n", encoding="utf-8")


def test_valid_tree_passes(tmp_path):
    build_valid_tree(tmp_path)

    assert check_structure.collect_failures(tmp_path) == []


def test_missing_required_doc_fails(tmp_path):
    build_valid_tree(tmp_path)
    (tmp_path / "docs" / "OWNERSHIP.md").unlink()

    failures = check_structure.collect_failures(tmp_path)

    assert failures == ["missing required path: docs/OWNERSHIP.md"]


def test_forbidden_imaging_file_fails(tmp_path):
    build_valid_tree(tmp_path)
    (tmp_path / "blocks" / "volume_hu.nii.gz").write_bytes(b"\x1f\x8b")

    failures = check_structure.collect_failures(tmp_path)

    assert len(failures) == 1
    assert "forbidden imaging artifact" in failures[0]
    assert "not imaging data" in failures[0]


def test_forbidden_data_directory_fails(tmp_path):
    build_valid_tree(tmp_path)
    (tmp_path / "datasets").mkdir()

    failures = check_structure.collect_failures(tmp_path)

    assert failures == [
        (
            "forbidden data directory datasets/: "
            "the repo stores references/manifests, not imaging data"
        )
    ]


def test_link_to_missing_file_fails(tmp_path):
    (tmp_path / "README.md").write_text("[guide](docs/missing.md)\n", encoding="utf-8")

    _, failures = check_links.scan(tmp_path)

    assert failures == ["README.md:1: missing link target: docs/missing.md"]


def test_link_with_wrong_case_fails(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "Ownership.md").write_text("# Sở hữu\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("[docs](docs/ownership.md)\n", encoding="utf-8")

    _, failures = check_links.scan(tmp_path)

    assert len(failures) == 1
    assert "wrong case for link target: docs/ownership.md" in failures[0]


def test_link_to_existing_file_passes(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CI.md").write_text("# CI\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("[ci](docs/CI.md)\n", encoding="utf-8")

    count, failures = check_links.scan(tmp_path)

    assert failures == []
    assert count == 2


def test_anchor_with_diacritics_and_inline_code(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CI.md").write_text(
        "## Kiểm tra `check_links.py` tự động\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "[ci](docs/CI.md#kiểm-tra-check_linkspy-tự-động)\n", encoding="utf-8"
    )

    _, failures = check_links.scan(tmp_path)

    assert failures == []


def test_duplicate_heading_anchor_suffix(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CI.md").write_text(
        "## Ghi chú\n\n## Ghi chú\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "[a](docs/CI.md#ghi-chú)\n[b](docs/CI.md#ghi-chú-1)\n", encoding="utf-8"
    )

    _, failures = check_links.scan(tmp_path)

    assert failures == []


def test_missing_anchor_fails(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CI.md").write_text("## Ghi chú\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("[a](docs/CI.md#khac)\n", encoding="utf-8")

    _, failures = check_links.scan(tmp_path)

    assert failures == ["README.md:1: anchor not found in docs/CI.md: #khac"]


def test_link_in_fenced_block_ignored(tmp_path):
    (tmp_path / "README.md").write_text(
        "~~~text\n[broken](nowhere.md)\n~~~\n\n```text\n[broken](nowhere.md)\n```\n",
        encoding="utf-8",
    )

    _, failures = check_links.scan(tmp_path)

    assert failures == []


def test_link_in_inline_code_ignored(tmp_path):
    (tmp_path / "README.md").write_text(
        "Use `[broken](nowhere.md)` in prose.\n", encoding="utf-8"
    )

    _, failures = check_links.scan(tmp_path)

    assert failures == []


def test_external_link_ignored(tmp_path):
    (tmp_path / "README.md").write_text(
        "[web](https://example.com/a.md)\n[mail](mailto:user@example.com)\n",
        encoding="utf-8",
    )

    _, failures = check_links.scan(tmp_path)

    assert failures == []


def test_self_anchor_in_same_file(tmp_path):
    (tmp_path / "README.md").write_text(
        "# Tiêu đề\n\n[x](#tiêu-đề)\n[y](#khong-co)\n", encoding="utf-8"
    )

    _, failures = check_links.scan(tmp_path)

    assert failures == ["README.md:4: anchor not found: #khong-co"]


def test_missing_image_target_fails(tmp_path):
    (tmp_path / "README.md").write_text("![ct](img/missing.png)\n", encoding="utf-8")

    _, failures = check_links.scan(tmp_path)

    assert failures == ["README.md:1: missing link target: img/missing.png"]


def test_present_image_target_passes(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "present.png").write_bytes(b"\x89PNG")
    (tmp_path / "README.md").write_text("![ct](img/present.png)\n", encoding="utf-8")

    _, failures = check_links.scan(tmp_path)

    assert failures == []


def test_link_with_title_is_checked(tmp_path):
    (tmp_path / "README.md").write_text(
        '[a](nowhere.md "title text")\n', encoding="utf-8"
    )

    _, failures = check_links.scan(tmp_path)

    assert failures == ["README.md:1: missing link target: nowhere.md"]


def test_external_image_ignored(tmp_path):
    (tmp_path / "README.md").write_text(
        "![web](https://example.com/a.png)\n", encoding="utf-8"
    )

    _, failures = check_links.scan(tmp_path)

    assert failures == []
