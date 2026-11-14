"""Tests for HttpRangeFile using an in-memory ZIP served through a fake fetch (synthetic data)."""
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from remote_zip import HttpRangeFile  # noqa: E402


def make_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("a/one.txt", b"hello" * 1000)
        archive.writestr("a/two.txt", b"world" * 2000)
        archive.writestr("labels.csv", b"id,label\n1,0\n")
    return buffer.getvalue()


def make_remote(blob: bytes, block_size=64):
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return blob[start:end + 1]
    return HttpRangeFile(fetch, len(blob), block_size=block_size), calls


def test_lists_members_and_extracts_one_without_reading_everything():
    blob = make_zip_bytes()
    remote, calls = make_remote(blob)
    with zipfile.ZipFile(remote) as archive:
        assert sorted(archive.namelist()) == ["a/one.txt", "a/two.txt", "labels.csv"]
        assert archive.read("labels.csv") == b"id,label\n1,0\n"
    assert remote.bytes_fetched < len(blob)
    assert all(0 <= s <= e < len(blob) for s, e in calls)


def test_extracted_content_matches_original():
    blob = make_zip_bytes()
    remote, _ = make_remote(blob)
    with zipfile.ZipFile(remote) as archive:
        assert archive.read("a/two.txt") == b"world" * 2000


def make_deflate_zip(payloads: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_fetch_member_roundtrip_without_central_directory():
    from remote_zip import fetch_member
    import random
    rng = random.Random(0)
    payloads = {f"d/f{i}.bin": bytes(rng.randrange(256) for _ in range(3000)) * 3 for i in range(4)}
    blob = make_deflate_zip(payloads)
    fetch = lambda s, e: blob[s:e + 1]
    for info in zipfile.ZipFile(io.BytesIO(blob)).infolist():
        got = fetch_member(fetch, info.header_offset, info.compress_size, len(info.filename))
        assert got == payloads[info.filename]


def test_fetch_member_head_is_prefix_of_full_data():
    from remote_zip import fetch_member
    data = b"HEADER" + bytes(range(256)) * 500
    blob = make_deflate_zip({"x.bin": data})
    info = zipfile.ZipFile(io.BytesIO(blob)).infolist()[0]
    fetch = lambda s, e: blob[s:e + 1]
    head = fetch_member(fetch, info.header_offset, info.compress_size, len(info.filename), head_only_bytes=200)
    assert head and data.startswith(head)


def test_seek_end_and_read_past_end():
    blob = make_zip_bytes()
    remote, _ = make_remote(blob)
    remote.seek(-4, 2)
    assert remote.read(100) == blob[-4:]
    assert remote.read(10) == b""
