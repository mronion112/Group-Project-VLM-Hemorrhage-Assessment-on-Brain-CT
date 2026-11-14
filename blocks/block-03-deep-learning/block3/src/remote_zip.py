"""Read a remote ZIP over HTTP range requests (no full download).

HttpRangeFile is a read-only seekable file object: zipfile.ZipFile can list the
central directory and extract single members from it. All network access goes
through an injectable `fetch(start, end_inclusive) -> bytes`, so it is unit-testable.
"""
import io
import struct
import urllib.request
import zlib
from typing import Callable

Fetch = Callable[[int, int], bytes]


def http_fetch(url: str, timeout: int = 60) -> Fetch:
    def fetch(start: int, end: int) -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    return fetch


def http_size(url: str, timeout: int = 60) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.headers["Content-Length"])


_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")  # signature .. name_len, extra_len
_SLACK = 1024


def fetch_member(fetch: Fetch, header_offset: int, compress_size: int, name_len: int,
                 head_only_bytes: int | None = None) -> bytes:
    """Fetch and inflate ONE deflate member given its listing entry (no central directory needed).

    head_only_bytes: if set, only fetch/inflate that many compressed bytes (a prefix of the
    uncompressed data; enough for a DICOM header read with stop_before_pixels).
    """
    wanted = compress_size if head_only_bytes is None else min(compress_size, head_only_bytes)
    blob = fetch(header_offset, header_offset + _LOCAL_HEADER.size + name_len + wanted + _SLACK - 1)
    fields = _LOCAL_HEADER.unpack(blob[:_LOCAL_HEADER.size])
    if fields[0] != 0x04034B50:
        raise ValueError("bad local header signature")
    data_start = _LOCAL_HEADER.size + fields[9] + fields[10]
    if len(blob) < data_start + wanted:
        blob += fetch(header_offset + len(blob), header_offset + data_start + wanted - 1)
    payload = blob[data_start:data_start + wanted]
    inflater = zlib.decompressobj(-15)
    data = inflater.decompress(payload)
    if head_only_bytes is None:
        data += inflater.flush()
    return data


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file. Counts bytes fetched."""

    def __init__(self, fetch: Fetch, size: int, block_size: int = 1 << 20):
        self._fetch, self._size, self._block = fetch, size, block_size
        self._pos = 0
        self._cache_start, self._cache = 0, b""
        self.bytes_fetched = 0
        self.requests = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._size - self._pos
        end = min(self._pos + size, self._size)
        if end <= self._pos:
            return b""
        cache_end = self._cache_start + len(self._cache)
        if not (self._cache_start <= self._pos and end <= cache_end):
            fetch_end = min(max(end, self._pos + self._block), self._size)
            self._cache = self._fetch(self._pos, fetch_end - 1)
            self._cache_start = self._pos
            self.bytes_fetched += len(self._cache)
            self.requests += 1
        data = self._cache[self._pos - self._cache_start: end - self._cache_start]
        self._pos += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)
