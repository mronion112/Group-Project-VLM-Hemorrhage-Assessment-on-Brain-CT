#!/usr/bin/env python3
"""Validate relative Markdown links and heading anchors."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

# Hidden tooling directories stay out of the scan; .github is content and is kept.
VISIBLE_HIDDEN_DIRS = frozenset({".github"})
IGNORED_DIR_NAMES = frozenset({"node_modules", "venv", "__pycache__"})

EXTERNAL_PREFIXES = ("http://", "https://", "mailto:")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
LINK_RE = re.compile(
    r"(?P<image>!?)\[(?P<text>[^\]]*)\]\(\s*(?P<target>[^()\s]*)"
    r'(?:\s+"[^"]*")?\s*\)'
)
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def strip_inline_code(text: str) -> str:
    """Remove inline code spans, including their backtick delimiters."""
    parts: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text[index] != "`":
            parts.append(text[index])
            index += 1
            continue
        end = index
        while end < length and text[end] == "`":
            end += 1
        ticks = text[index:end]
        closing = text.find(ticks, end)
        if closing == -1:
            parts.append(ticks)
            index = end
            continue
        index = closing + len(ticks)
    return "".join(parts)


def slugify(heading: str) -> str:
    """Convert heading text into a GitHub-style anchor slug."""
    # Heading anchors keep inline code content and only drop the backticks.
    text = heading.replace("`", "")
    text = LINK_RE.sub(lambda match: match.group("text"), text)
    text = text.strip().lower()
    kept = [char for char in text if char.isalnum() or char in "_- "]
    return "".join(kept).replace(" ", "-")


def read_lines(path: Path) -> list[str]:
    """Read a file as UTF-8 lines, tolerating undecodable bytes."""
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def iter_content_lines(lines: list[str]):
    """Yield (line_number, text) pairs outside fenced code blocks."""
    fence = ""
    for number, line in enumerate(lines, start=1):
        match = FENCE_RE.match(line)
        if fence:
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
                fence = ""
            continue
        if match:
            fence = match.group(1)
            continue
        yield number, line


def collect_anchors(path: Path) -> set[str]:
    """Return the anchor slugs declared by a Markdown file's headings."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for _, line in iter_content_lines(read_lines(path)):
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = slugify(match.group(2))
        count = seen.get(base, 0)
        anchors.add(base if count == 0 else f"{base}-{count}")
        seen[base] = count + 1
    return anchors


def resolve_case_insensitive(source_dir: Path, target: str) -> Path | None:
    """Find a path that matches target ignoring case, or return None."""
    current = source_dir
    for part in Path(target).parts:
        if part in ("", ".", ".."):
            current = (current / part)
            continue
        if not current.is_dir():
            return None
        lowered = part.lower()
        match = next((child for child in current.iterdir() if child.name.lower() == lowered), None)
        if match is None:
            return None
        current = match
    return current


def check_link(
    source: Path,
    source_dir: Path,
    target: str,
    anchors: dict[Path, set[str]],
    is_image: bool = False,
) -> str | None:
    """Return a failure reason for one link target, or None when it resolves."""
    path_part, _, anchor = target.partition("#")
    if is_image:
        # Image targets point at files, never at document anchors.
        anchor = ""
    if not path_part:
        if not anchor:
            return None
        if resolve_anchor(source, anchor, anchors):
            return None
        return f"anchor not found: #{anchor}"

    resolved = (source_dir / path_part).resolve()
    if resolved.exists():
        if anchor and not resolve_anchor(resolved, anchor, anchors):
            return f"anchor not found in {path_part}: #{anchor}"
        return None

    insensitive = resolve_case_insensitive(source_dir, path_part)
    if insensitive is not None:
        return f"wrong case for link target: {path_part} (actual: {insensitive})"
    return f"missing link target: {path_part}"


def resolve_anchor(path: Path, anchor: str, anchors: dict[Path, set[str]]) -> bool:
    """Report whether an anchor exists in a Markdown file; non-Markdown targets always pass."""
    if path.suffix.lower() != ".md" or not path.is_file():
        return True
    if path not in anchors:
        anchors[path] = collect_anchors(path)
    return anchor.lower() in anchors[path]


def is_ignored_dir(name: str) -> bool:
    """Report whether a directory name belongs to local tooling state."""
    if name in VISIBLE_HIDDEN_DIRS:
        return False
    return name.startswith(".") or name in IGNORED_DIR_NAMES


def iter_markdown_files(root: Path) -> Iterator[Path]:
    """Yield every Markdown file below root, pruning ignored trees."""
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not is_ignored_dir(name))
        base = Path(current)
        for name in sorted(files):
            if name.lower().endswith(".md"):
                yield base / name


def scan(root: Path) -> tuple[int, list[str]]:
    """Validate every Markdown file under root; return file count and failures."""
    markdown_files = sorted(iter_markdown_files(root))
    anchors: dict[Path, set[str]] = {}
    failures: list[str] = []

    for path in markdown_files:
        relative = path.relative_to(root).as_posix()
        for number, line in iter_content_lines(read_lines(path)):
            for match in LINK_RE.finditer(strip_inline_code(line)):
                target = match.group("target").strip()
                if not target or target.startswith(EXTERNAL_PREFIXES):
                    continue
                reason = check_link(
                    path,
                    path.parent,
                    target,
                    anchors,
                    is_image=bool(match.group("image")),
                )
                if reason:
                    failures.append(f"{relative}:{number}: {reason}")

    return len(markdown_files), failures


def main(argv: list[str] | None = None) -> int:
    """Run the link check and print one line per failure."""
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

    count, failures = scan(root)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print(f"OK: links ({count} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
