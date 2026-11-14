"""Measure hardware/storage facts and write block3/configs/hardware.json.

Only measures; never invents values. Fields that cannot be measured are null.
"""
import ctypes
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "block3" / "configs" / "hardware.json"
GB = 1024 ** 3


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory_gb() -> dict:
    if os.name == "nt":
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"total_gb": None, "available_gb": None, "load_percent": None}
        return {"total_gb": round(status.ullTotalPhys / GB, 2),
                "available_gb": round(status.ullAvailPhys / GB, 2),
                "load_percent": status.dwMemoryLoad}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        kv = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            parts = line.split(":")
            if len(parts) == 2 and parts[1].strip().split()[0].isdigit():
                kv[parts[0]] = int(parts[1].strip().split()[0])
        total_gb = round(kv.get("MemTotal", 0) / (1024 ** 2), 2)
        avail_gb = round(kv.get("MemAvailable", kv.get("MemFree", 0)) / (1024 ** 2), 2)
        load_pct = int(round(100.0 * (total_gb - avail_gb) / total_gb)) if total_gb > 0 else None
        return {"total_gb": total_gb, "available_gb": avail_gb, "load_percent": load_pct}
    return {"total_gb": None, "available_gb": None, "load_percent": None}


def folder_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0
    return round(total / GB, 3)


def collect() -> dict:
    disk = shutil.disk_usage(PROJECT_ROOT)
    info = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": PROJECT_ROOT.name,
        "os": platform.platform(),
        "cpu_name": platform.processor(),
        "cpu_logical_threads": os.cpu_count(),
        "ram": memory_gb(),
        "disk": {"drive": PROJECT_ROOT.drive or "/", "total_gb": round(disk.total / GB, 2),
                 "free_gb": round(disk.free / GB, 2)},
        "venv_size_gb": folder_size_gb(PROJECT_ROOT / ".venv"),
        "python": sys.version.split()[0],
        "inside_onedrive": "onedrive" in str(PROJECT_ROOT).lower(),
    }
    try:
        import torch
        info["torch"] = {"version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                         "num_threads": torch.get_num_threads()}
    except ImportError:
        info["torch"] = None
    return info


if __name__ == "__main__":
    data = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))
