"""List RSNA remote zip members (reads only the central directory) -> data/inventory/rsna_zip_listing.csv."""
import csv
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block3" / "src"))
from remote_zip import HttpRangeFile, http_fetch, http_size  # noqa: E402

URL = "https://intracranial-hemorrhage.s3.us-west-2.amazonaws.com/rsna-ihd-dataset.zip"
OUT = ROOT / "block3" / "data" / "inventory" / "rsna_zip_listing.csv"


def main() -> None:
    started = time.time()
    remote = HttpRangeFile(http_fetch(URL, timeout=300), http_size(URL), block_size=8 << 20)
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "compress_size", "file_size", "compress_type", "header_offset"])
        for info in infos:
            writer.writerow([info.filename, info.compress_size, info.file_size, info.compress_type, info.header_offset])
    print(f"entries={len(infos)} bytes_fetched={remote.bytes_fetched} requests={remote.requests} "
          f"seconds={time.time() - started:.1f} out={OUT}")


if __name__ == "__main__":
    main()
