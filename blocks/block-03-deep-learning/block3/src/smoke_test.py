"""Phase 2 CPU Smoke Test for Block 3: RAM check, preprocess E1 windowing, resolution roundtrip, and CPU epoch timing.

Supports two modes:
- Default: uses local processed `.npz` files in `block3/data/processed/` (or minimal fallback).
- `--synthetic`: self-contained end-to-end pipeline test (synthetic 3D HU CT + lesion -> .npz check -> E1 windowing
  -> classifier + TinyUNet2D train/checkpoint -> 512x512 restore -> NIfTI mask export -> findings.json schema check).
  Requires ZERO downloaded datasets so teammates can verify the pipeline immediately after cloning.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, resnet18

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block3" / "src"))
from check_splits import check_split_files, validate_split_rows  # noqa: E402
from preprocess import (  # noqa: E402
    apply_window_mode,
    evaluate_mask_on_orig_grid,
)
from prepare_data import resize_volume_2d, save_and_verify_npz  # noqa: E402
from validate_output import validate_findings  # noqa: E402

if os.name == "nt":
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
else:
    import resource


def get_system_ram_status() -> dict[str, float | int | str]:
    if os.name == "nt":
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        total_gb = round(m.ullTotalPhys / (1024 ** 3), 2)
        avail_gb = round(m.ullAvailPhys / (1024 ** 3), 2)
        used_gb = round((m.ullTotalPhys - m.ullAvailPhys) / (1024 ** 3), 2)
        load_pct = int(m.dwMemoryLoad)
    else:
        total_kb = 0
        avail_kb = 0
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            kv: dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                parts = line.split(":")
                if len(parts) == 2:
                    vals = parts[1].strip().split()
                    if vals and vals[0].isdigit():
                        kv[parts[0]] = int(vals[0])
            total_kb = kv.get("MemTotal", 0)
            avail_kb = kv.get("MemAvailable", kv.get("MemFree", 0))
        elif hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_sz = os.sysconf("SC_PAGE_SIZE")
            total_kb = int((pages * page_sz) // 1024)
            avail_kb = 0
        total_gb = round(total_kb / (1024 ** 2), 2)
        avail_gb = round(avail_kb / (1024 ** 2), 2)
        used_gb = round(max(0.0, total_gb - avail_gb), 2)
        load_pct = int(round(100.0 * (used_gb / total_gb))) if total_gb > 0 else 0

    warning = ""
    if avail_gb > 0 and (avail_gb < 4.0 or load_pct >= 80):
        warning = (
            f"WARNING: System RAM is {load_pct}% used ({avail_gb} GB free of {total_gb} GB). "
            "Please close background applications (browsers, unused Electron windows) before running multi-epoch training!"
        )
    return {
        "total_gb": total_gb,
        "available_gb": avail_gb,
        "used_gb": used_gb,
        "memory_load_percent": load_pct,
        "warning": warning,
    }


def get_process_rss_mb() -> dict[str, float]:
    if os.name == "nt":
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = k32.GetCurrentProcess()
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
        if not ok:
            k32.K32GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
            k32.K32GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
        return {
            "rss_mb": round(pmc.WorkingSetSize / (1024 ** 2), 2),
            "peak_rss_mb": round(pmc.PeakWorkingSetSize / (1024 ** 2), 2),
            "pagefile_mb": round(pmc.PagefileUsage / (1024 ** 2), 2),
            "peak_pagefile_mb": round(pmc.PeakPagefileUsage / (1024 ** 2), 2),
        }

    # POSIX fallback (Linux /proc/self/status + resource.getrusage; macOS resource.getrusage in bytes)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = ru / (1024 ** 2) if sys.platform == "darwin" else ru / 1024.0
    rss_mb = peak_mb
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                vals = line.split()
                if len(vals) >= 2 and vals[1].isdigit():
                    rss_mb = int(vals[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                vals = line.split()
                if len(vals) >= 2 and vals[1].isdigit():
                    peak_mb = int(vals[1]) / 1024.0
    return {
        "rss_mb": round(rss_mb, 2),
        "peak_rss_mb": round(peak_mb, 2),
        "pagefile_mb": 0.0,
        "peak_pagefile_mb": 0.0,
    }


class TinyUNet2D(nn.Module):
    """Lightweight 2D U-Net for CPU segmentation (input: Bx3x256x256 -> BxCx256x256)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 6, base: int = 16) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_classifier(arch: str, num_classes: int = 6) -> nn.Module:
    if arch == "mobilenet_v3_small":
        m = mobilenet_v3_small(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if arch == "resnet18":
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unsupported classifier arch: {arch}")


def save_and_verify_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    extra: dict,
) -> str:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "extra": extra,
    }
    torch.save(payload, ckpt_path)
    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if loaded["epoch"] != epoch or loaded["step"] != step:
        raise RuntimeError("Checkpoint resume verification failed!")
    return f"sha256:{sha}"


def measure_downsampling_effect() -> dict:
    """Measure exact Dice & HD95 (mm) loss from 512x512 -> 256x256 -> 512x512 nearest-neighbor roundtrip."""
    spacing = (0.4883, 0.4883, 5.0)
    shape = (16, 512, 512)
    yy, xx = np.ogrid[:512, :512]
    results = {}

    # 1. Medium spherical/cylindrical parenchymal lesion (radius=16 px ~ 7.81 mm across 4 slices)
    gt_med = np.zeros(shape, dtype=np.uint8)
    gt_med[6:10, (yy - 256) ** 2 + (xx - 256) ** 2 <= 16 ** 2] = 1
    sub_med = resize_volume_2d(gt_med, (256, 256), is_mask=True)
    results["synthetic_medium_r16px_7.8mm"] = {
        "orig_positive_voxels": int(gt_med.sum()),
        **evaluate_mask_on_orig_grid(sub_med, gt_med, spacing),
    }

    # 2. Small lesion (radius=4 px ~ 1.95 mm across 2 slices)
    gt_small = np.zeros(shape, dtype=np.uint8)
    gt_small[7:9, (yy - 201) ** 2 + (xx - 303) ** 2 <= 4 ** 2] = 1
    sub_small = resize_volume_2d(gt_small, (256, 256), is_mask=True)
    results["synthetic_small_r4px_1.95mm"] = {
        "orig_positive_voxels": int(gt_small.sum()),
        **evaluate_mask_on_orig_grid(sub_small, gt_small, spacing),
    }

    # 3. Thin subdural crescent (thickness=2 px ~ 0.98 mm along arc)
    gt_thin = np.zeros(shape, dtype=np.uint8)
    dist = np.sqrt((yy - 256) ** 2 + (xx - 256) ** 2)
    gt_thin[5:9, (dist >= 180.0) & (dist <= 182.0) & (xx > 256)] = 1
    sub_thin = resize_volume_2d(gt_thin, (256, 256), is_mask=True)
    results["synthetic_thin_crescent_2px_0.98mm"] = {
        "orig_positive_voxels": int(gt_thin.sum()),
        **evaluate_mask_on_orig_grid(sub_thin, gt_thin, spacing),
    }

    # 4. Real BHSD local smoke case if present
    bhsd_npz = ROOT / "block3" / "data" / "processed" / "bhsd" / "bhsd_009adb6527f7500c.npz"
    if bhsd_npz.exists():
        with np.load(bhsd_npz, allow_pickle=False) as d:
            gt_orig = d["mask_ich_orig_zyx_uint8"]
            sub_mask = d["mask_ich_uint8"]
            sp = d["orig_spacing_xyz_mm"]
            results["real_bhsd_009adb6527f7500c"] = {
                "orig_positive_voxels": int(gt_orig.sum()),
                **evaluate_mask_on_orig_grid(sub_mask, gt_orig, sp),
            }
    return results


def load_local_smoke_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Load local processed .npz slices (or fallback synthetic if none exist)."""
    proc_dir = ROOT / "block3" / "data" / "processed"
    rsna_npz = proc_dir / "rsna_ich" / "rsna_0007c6017e8b082b.npz"
    bhsd_npz = proc_dir / "bhsd" / "bhsd_009adb6527f7500c.npz"

    cls_slices = []
    cls_labels = []
    if bhsd_npz.exists():
        with np.load(bhsd_npz, allow_pickle=False) as d:
            hu_b = d["hu_int16"]
            lbl_b = d["slice_labels_uint8"]
            mask_b = d["mask_subtype_uint8"]
            mask_orig_b = d["mask_ich_orig_zyx_uint8"]
            sp_b = d["orig_spacing_xyz_mm"]
        win_b = apply_window_mode(hu_b, mode="brain_subdural_bone", expand_to_3ch=True)
        cls_slices.append(win_b)
        cls_labels.append(lbl_b)
    else:
        hu_b = np.random.randint(-1000, 1000, size=(16, 256, 256), dtype=np.int16)
        win_b = apply_window_mode(hu_b, mode="brain_subdural_bone", expand_to_3ch=True)
        lbl_b = np.zeros((16, 6), dtype=np.uint8)
        mask_b = np.zeros((16, 256, 256), dtype=np.uint8)
        mask_orig_b = np.zeros((16, 512, 512), dtype=np.uint8)
        sp_b = np.array([0.4883, 0.4883, 5.0], dtype=np.float32)
        cls_slices.append(win_b)
        cls_labels.append(lbl_b)

    if rsna_npz.exists():
        with np.load(rsna_npz, allow_pickle=False) as d:
            hu_r = d["hu_int16"]
            lbl_r = d["slice_labels_uint8"]
        win_r = apply_window_mode(hu_r, mode="brain_subdural_bone", expand_to_3ch=True)
        cls_slices.append(win_r)
        cls_labels.append(lbl_r)

    x_cls = torch.from_numpy(np.concatenate(cls_slices, axis=0))
    y_cls = torch.from_numpy(np.concatenate(cls_labels, axis=0).astype(np.float32))
    x_seg = torch.from_numpy(win_b)
    y_seg = torch.from_numpy(mask_b.astype(np.int64))
    return x_cls, y_cls, x_seg, y_seg, mask_orig_b, sp_b


def run_smoke_benchmark(max_cls_slices: int | None = None, max_seg_slices: int | None = None) -> dict:
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    ram_before = get_system_ram_status()
    if ram_before["warning"]:
        print(ram_before["warning"])

    rss_start = get_process_rss_mb()
    x_cls, y_cls, x_seg, y_seg, mask_orig_b, sp_b = load_local_smoke_tensors()
    if max_cls_slices is not None:
        x_cls, y_cls = x_cls[:max_cls_slices], y_cls[:max_cls_slices]
    if max_seg_slices is not None:
        x_seg, y_seg = x_seg[:max_seg_slices], y_seg[:max_seg_slices]
        mask_orig_b = mask_orig_b[:max_seg_slices]

    ckpt_dir = ROOT / "block3" / "checkpoints" / "smoke"
    cls_results = {}
    batch_size_cls = 8
    criterion_cls = nn.BCEWithLogitsLoss()

    for arch in ("mobilenet_v3_small", "resnet18"):
        model = build_classifier(arch, num_classes=6)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        t0 = time.perf_counter()
        n_slices = int(x_cls.shape[0])
        steps = 0
        last_loss = 0.0
        for i in range(0, n_slices, batch_size_cls):
            bx = x_cls[i : i + batch_size_cls]
            by = y_cls[i : i + batch_size_cls]
            opt.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = criterion_cls(logits, by)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            steps += 1
        elapsed = time.perf_counter() - t0
        sec_per_slice = elapsed / max(1, n_slices)
        ckpt_path = ckpt_dir / f"smoke_{arch}.pt"
        ckpt_sha = save_and_verify_checkpoint(ckpt_path, model, opt, epoch=1, step=steps, extra={"arch": arch})
        rss_now = get_process_rss_mb()
        cls_results[arch] = {
            "num_slices_measured": n_slices,
            "batch_size": batch_size_cls,
            "steps": steps,
            "epoch_seconds": round(elapsed, 3),
            "ms_per_slice": round(sec_per_slice * 1000.0, 2),
            "slices_per_sec": round(1.0 / max(1e-9, sec_per_slice), 2),
            "extrapolated_epoch_min_3000_slices": round((sec_per_slice * 3000.0) / 60.0, 2),
            "extrapolated_epoch_min_10017_slices": round((sec_per_slice * 10017.0) / 60.0, 2),
            "last_loss": round(last_loss, 5),
            "checkpoint_verified": ckpt_sha,
            "rss_mb": rss_now["rss_mb"],
            "peak_rss_mb": rss_now["peak_rss_mb"],
        }

    # Segmentation 2D U-Net smoke test
    seg_model = TinyUNet2D(in_channels=3, out_channels=6, base=16)
    seg_model.train()
    seg_opt = torch.optim.AdamW(seg_model.parameters(), lr=1e-3)
    seg_crit = nn.CrossEntropyLoss()
    batch_size_seg = 4
    n_seg = int(x_seg.shape[0])
    t0_seg = time.perf_counter()
    seg_steps = 0
    seg_loss_val = 0.0
    for i in range(0, n_seg, batch_size_seg):
        bx = x_seg[i : i + batch_size_seg]
        by = y_seg[i : i + batch_size_seg]
        seg_opt.zero_grad(set_to_none=True)
        out = seg_model(bx)
        loss = seg_crit(out, by)
        loss.backward()
        seg_opt.step()
        seg_loss_val = float(loss.item())
        seg_steps += 1
    seg_elapsed = time.perf_counter() - t0_seg
    seg_sec_per_slice = seg_elapsed / max(1, n_seg)
    seg_ckpt_sha = save_and_verify_checkpoint(
        ckpt_dir / "smoke_tiny_unet2d.pt", seg_model, seg_opt, epoch=1, step=seg_steps, extra={"base": 16}
    )

    # Inference + restoration to original 512x512 grid
    seg_model.eval()
    t0_inf = time.perf_counter()
    with torch.no_grad():
        pred_logits = seg_model(x_seg)
        pred_sub_zyx = torch.argmax(pred_logits, dim=1).numpy().astype(np.uint8)
    orig_eval = evaluate_mask_on_orig_grid(pred_sub_zyx > 0, mask_orig_b > 0, sp_b)
    inf_elapsed = time.perf_counter() - t0_inf

    downsampling_study = measure_downsampling_effect()
    rss_end = get_process_rss_mb()
    ram_after = get_system_ram_status()

    # Clean up smoke test temporary checkpoints so we don't pollute checkpoints/
    for p in ckpt_dir.glob("*.pt"):
        p.unlink(missing_ok=True)

    return {
        "system_ram_before": ram_before,
        "system_ram_after": ram_after,
        "process_memory_mb": {
            "start_rss_mb": rss_start["rss_mb"],
            "end_rss_mb": rss_end["rss_mb"],
            "peak_rss_mb": rss_end["peak_rss_mb"],
            "pagefile_mb": rss_end["pagefile_mb"],
            "peak_pagefile_mb": rss_end["peak_pagefile_mb"],
        },
        "cpu_threads_used": torch.get_num_threads(),
        "classification_2d_smoke": cls_results,
        "segmentation_2d_unet_smoke": {
            "num_slices_measured": n_seg,
            "batch_size": batch_size_seg,
            "steps": seg_steps,
            "train_epoch_seconds": round(seg_elapsed, 3),
            "ms_per_slice": round(seg_sec_per_slice * 1000.0, 2),
            "extrapolated_epoch_min_4400_bhsd_slices": round((seg_sec_per_slice * 4400.0) / 60.0, 2),
            "inference_and_orig_grid_restore_seconds_per_volume": round(inf_elapsed, 3),
            "orig_grid_eval_check": orig_eval,
            "last_loss": round(seg_loss_val, 5),
            "checkpoint_verified": seg_ckpt_sha,
        },
        "downsampling_512_to_256_roundtrip_effect": downsampling_study,
    }


def generate_synthetic_ct_case(
    num_slices: int = 6,
    hw: int = 512,
    seed: int = 2026,
) -> dict[str, np.ndarray]:
    """Generate a synthetic 3D brain CT (int16 HU) + 5-subtype ICH mask on an exact (Z, H, W) grid."""
    rng = np.random.default_rng(seed)
    spacing = np.array([0.4883, 0.4883, 5.0], dtype=np.float32)
    # Centered RAS+ diagonal affine so center voxel maps near physical X=0
    affine = np.array(
        [
            [spacing[0], 0.0, 0.0, -(hw / 2.0) * spacing[0]],
            [0.0, spacing[1], 0.0, -(hw / 2.0) * spacing[1]],
            [0.0, 0.0, spacing[2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    yy, xx = np.ogrid[:hw, :hw]
    r2 = (yy - hw // 2) ** 2 + (xx - hw // 2) ** 2
    hu = np.full((num_slices, hw, hw), -1000, dtype=np.int16)
    mask_sub = np.zeros((num_slices, hw, hw), dtype=np.uint8)

    # Skull (+900 HU), brain parenchyma (+35 HU), ventricles (+10 HU)
    skull = (r2 <= (hw * 0.40) ** 2) & (r2 >= (hw * 0.36) ** 2)
    brain = r2 < (hw * 0.36) ** 2
    ventricles = ((yy - hw // 2) ** 2 / (22.0 ** 2) + (xx - hw // 2) ** 2 / (14.0 ** 2)) <= 1.0
    for z in range(num_slices):
        hu[z, skull] = 900
        hu[z, brain] = 35
        hu[z, ventricles] = 10

    # Right-hemisphere parenchymal bleed (BHSD code 2=IPH) + intraventricular component (code 3=IVH)
    lesion_slices = range(1, max(2, num_slices - 1))
    iph_mask = ((yy - int(hw * 0.48)) ** 2 + (xx - int(hw * 0.64)) ** 2) <= 22 ** 2
    ivh_mask = ((yy - int(hw * 0.50)) ** 2 + (xx - int(hw * 0.54)) ** 2) <= 6 ** 2
    for z in lesion_slices:
        mask_sub[z, iph_mask] = 2
        mask_sub[z, ivh_mask] = 3
        hu[z, iph_mask | ivh_mask] = 68

    noise = rng.integers(-4, 5, size=hu.shape, dtype=np.int16)
    hu = (hu + noise).clip(-1024, 3071).astype(np.int16)
    mask_ich = (mask_sub > 0).astype(np.uint8)

    # Canonical slice/exam labels: [ICH, EDH, SDH, SAH, IPH, IVH]
    bhsd_to_canonical = {1: 1, 5: 2, 4: 3, 2: 4, 3: 5}
    slice_labels = np.zeros((num_slices, 6), dtype=np.uint8)
    for z in range(num_slices):
        codes = set(np.unique(mask_sub[z]).tolist()) - {0}
        if codes:
            slice_labels[z, 0] = 1
            for code in codes:
                slice_labels[z, bhsd_to_canonical[code]] = 1
    exam_labels = (slice_labels.max(axis=0) > 0).astype(np.uint8)

    return {
        "hu_orig_zyx": hu,
        "mask_sub_orig_zyx": mask_sub,
        "mask_ich_orig_zyx": mask_ich,
        "slice_labels_uint8": slice_labels,
        "exam_labels_uint8": exam_labels,
        "orig_spacing_xyz_mm": spacing,
        "orig_affine": affine,
    }


def compute_mask_side_and_extent(
    mask_zyx: np.ndarray,
    orig_affine: np.ndarray,
) -> dict[str, object]:
    """Derive 3D voxel bounding box (extent_zyx) and RAS+ hemisphere side from a (Z, Y, X) mask."""
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        return {"status": "missing", "side": None, "extent_zyx": None, "positive_voxels": 0}

    z_min, y_min, x_min = coords.min(axis=0).tolist()
    z_max, y_max, x_max = coords.max(axis=0).tolist()
    zs, ys, xs = coords[:, 0].astype(np.float64), coords[:, 1].astype(np.float64), coords[:, 2].astype(np.float64)

    z_dim, h_dim, w_dim = mask_zyx.shape
    center_vox = np.array([w_dim / 2.0, h_dim / 2.0, z_dim / 2.0, 1.0], dtype=np.float64)
    center_ras_x = float((orig_affine @ center_vox)[0])
    vox_ras_x = (
        float(orig_affine[0, 0]) * xs
        + float(orig_affine[0, 1]) * ys
        + float(orig_affine[0, 2]) * zs
        + float(orig_affine[0, 3])
    )
    dx_ras = vox_ras_x - center_ras_x
    n_total = len(dx_ras)
    n_right = int(np.count_nonzero(dx_ras > 1e-6))
    n_left = int(np.count_nonzero(dx_ras < -1e-6))

    if n_right > 0 and n_left > 0 and min(n_right, n_left) >= 0.1 * n_total:
        side = "bilateral"
    elif n_right >= n_left and n_right > 0:
        side = "right"
    elif n_left > 0:
        side = "left"
    else:
        side = "midline"

    return {
        "status": "mask-derived",
        "side": side,
        "extent_zyx": [int(z_min), int(y_min), int(x_min), int(z_max), int(y_max), int(x_max)],
        "positive_voxels": int(n_total),
    }


def run_synthetic_pipeline_smoke(
    num_slices: int = 6,
    artifacts_dir: Path | None = None,
    seed: int = 2026,
) -> dict:
    """Run full Block 3 synthetic pipeline end-to-end without any downloaded external datasets."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    ram_before = get_system_ram_status()
    rss_start = get_process_rss_mb()
    t_total_0 = time.perf_counter()

    ctx = tempfile.TemporaryDirectory(prefix="block3_synth_") if artifacts_dir is None else None
    work_dir = Path(ctx.name) if ctx is not None else Path(artifacts_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        # 1. Synthetic CT volume + .npz prepare/verify contract
        case = generate_synthetic_ct_case(num_slices=num_slices, hw=512, seed=seed)
        hu_256 = resize_volume_2d(case["hu_orig_zyx"], (256, 256), is_mask=False)
        mask_sub_256 = resize_volume_2d(case["mask_sub_orig_zyx"], (256, 256), is_mask=True)
        mask_ich_256 = (mask_sub_256 > 0).astype(np.uint8)

        npz_path = work_dir / "synthetic_case.npz"
        npz_sha, npz_bytes = save_and_verify_npz(
            npz_path,
            {
                "hu_int16": hu_256,
                "mask_type": np.array("multiclass_5_subtypes"),
                "mask_ich_uint8": mask_ich_256,
                "mask_subtype_uint8": mask_sub_256,
                "slice_labels_uint8": case["slice_labels_uint8"],
                "exam_labels_uint8": case["exam_labels_uint8"],
                "orig_shape_zyx": np.array(case["hu_orig_zyx"].shape, dtype=np.int32),
                "orig_spacing_xyz_mm": case["orig_spacing_xyz_mm"],
                "orig_affine": case["orig_affine"],
            },
        )

        # 2. Preprocess windowing (E1 brain_subdural_bone)
        with np.load(npz_path, allow_pickle=False) as loaded:
            x_np = apply_window_mode(loaded["hu_int16"], mode="brain_subdural_bone", expand_to_3ch=True)
            y_cls_np = loaded["slice_labels_uint8"].astype(np.float32)
            y_seg_np = loaded["mask_subtype_uint8"].astype(np.int64)
            orig_spacing = loaded["orig_spacing_xyz_mm"]
            orig_affine = loaded["orig_affine"]

        x_t = torch.from_numpy(x_np)
        y_cls_t = torch.from_numpy(y_cls_np)
        y_seg_t = torch.from_numpy(y_seg_np)

        # 3. Classifier + TinyUNet2D train steps & checkpoint verification
        cls_model = build_classifier("mobilenet_v3_small", num_classes=6)
        cls_model.train()
        cls_opt = torch.optim.AdamW(cls_model.parameters(), lr=1e-3)
        cls_crit = nn.BCEWithLogitsLoss()
        cls_loss_val = 0.0
        for _ in range(2):
            cls_opt.zero_grad(set_to_none=True)
            loss = cls_crit(cls_model(x_t), y_cls_t)
            loss.backward()
            cls_opt.step()
            cls_loss_val = float(loss.item())
        cls_sha = save_and_verify_checkpoint(
            work_dir / "cls_mobilenet_v3_small.pt",
            cls_model,
            cls_opt,
            epoch=1,
            step=2,
            extra={"synthetic": True},
        )

        seg_model = TinyUNet2D(in_channels=3, out_channels=6, base=16)
        seg_model.train()
        seg_opt = torch.optim.AdamW(seg_model.parameters(), lr=1e-3)
        seg_crit = nn.CrossEntropyLoss()
        seg_loss_val = 0.0
        for _ in range(2):
            seg_opt.zero_grad(set_to_none=True)
            loss = seg_crit(seg_model(x_t), y_seg_t)
            loss.backward()
            seg_opt.step()
            seg_loss_val = float(loss.item())
        seg_sha = save_and_verify_checkpoint(
            work_dir / "seg_tiny_unet2d.pt",
            seg_model,
            seg_opt,
            epoch=1,
            step=2,
            extra={"synthetic": True},
        )

        # 4. Inference + 512x512 grid restoration & metrics
        cls_model.eval()
        seg_model.eval()
        with torch.no_grad():
            cls_probs = torch.sigmoid(cls_model(x_t)).numpy()  # (Z, 6)
            case_probs = cls_probs.max(axis=0)  # exam-level max aggregation
            pred_sub_256 = torch.argmax(seg_model(x_t), dim=1).numpy().astype(np.uint8)
        pred_ich_256 = (pred_sub_256 > 0).astype(np.uint8)

        pred_eval = evaluate_mask_on_orig_grid(pred_ich_256, case["mask_ich_orig_zyx"], orig_spacing)
        roundtrip_ceiling = evaluate_mask_on_orig_grid(mask_ich_256, case["mask_ich_orig_zyx"], orig_spacing)
        pred_ich_512 = resize_volume_2d(pred_ich_256, (512, 512), is_mask=True)
        pred_sub_512 = resize_volume_2d(pred_sub_256, (512, 512), is_mask=True)

        # 5. Write NIfTI masks in canonical (X, Y, Z) layout and verify roundtrip
        ich_nii_path = work_dir / "mask_ich.nii.gz"
        sub_nii_path = work_dir / "mask_subtype.nii.gz"
        nib.save(nib.Nifti1Image(np.transpose(pred_ich_512, (2, 1, 0)), orig_affine), str(ich_nii_path))
        nib.save(nib.Nifti1Image(np.transpose(pred_sub_512, (2, 1, 0)), orig_affine), str(sub_nii_path))

        loaded_ich = nib.load(str(ich_nii_path))
        loaded_sub = nib.load(str(sub_nii_path))
        nifti_roundtrip_ok = (
            tuple(loaded_ich.shape) == (512, 512, num_slices)
            and tuple(loaded_sub.shape) == (512, 512, num_slices)
            and bool(np.allclose(loaded_ich.affine, orig_affine, atol=1e-4))
            and bool(np.allclose(loaded_ich.header.get_zooms()[:3], orig_spacing, atol=1e-4))
        )
        if not nifti_roundtrip_ok:
            raise RuntimeError("NIfTI roundtrip geometry verification failed in synthetic smoke test.")

        # 6. Localization (GT check + pred check) and schema-validated findings.json
        gt_loc = compute_mask_side_and_extent(case["mask_ich_orig_zyx"], orig_affine)
        pred_loc = compute_mask_side_and_extent(pred_ich_512, orig_affine)
        threshold = 0.5
        ich_score = float(case_probs[0])
        sub_codes = ("EDH", "SDH", "SAH", "IPH", "IVH")
        subtypes_dict = {
            code: {
                "score": round(float(case_probs[idx + 1]), 6),
                "present": bool(case_probs[idx + 1] >= threshold),
            }
            for idx, code in enumerate(sub_codes)
        }
        findings = {
            "schema_version": "1.0",
            "case_id": "synthetic-smoke-001",
            "status": "ok",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "source_artifact_ids": [npz_sha],
            "model_name": "mobilenet_v3_small+tiny_unet2d",
            "model_version": "synthetic-smoke-v0",
            "weights_hash": cls_sha,
            "config_hash": f"sha256:{hashlib.sha256(b'synthetic-smoke-v0').hexdigest()}",
            "input_space_id": "synthetic-smoke-001:block2:v1",
            "presence": {
                "label": "positive" if ich_score >= threshold else "negative",
                "score": round(ich_score, 6),
                "threshold": threshold,
            },
            "subtypes": subtypes_dict,
            "masks": {
                "ich": {"artifact": "mask_ich.nii.gz", "space_id": "synthetic-smoke-001:block2:v1"},
                "subtype": {"artifact": "mask_subtype.nii.gz", "space_id": "synthetic-smoke-001:block2:v1"},
            },
            "localization": {
                "status": str(pred_loc["status"]),
                "side": pred_loc["side"],
                "regions": [],
                "coordinate_system": "RAS+",
            },
            "heatmaps": [],
            "warnings": ["Synthetic CT smoke test output (not calibrated clinical predictions)."],
            "errors": [],
        }
        findings_path = work_dir / "findings.json"
        findings_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")
        schema_errors = validate_findings(json.loads(findings_path.read_text(encoding="utf-8")))
        if schema_errors:
            raise RuntimeError(f"Synthetic findings.json failed schema validation: {schema_errors}")

        aux_loc = {
            "case_id": "synthetic-smoke-001",
            "source_mask": "mask_ich.nii.gz",
            "predicted_mask_localization": pred_loc,
            "ground_truth_mask_localization": gt_loc,
            "coordinate_system": "RAS+",
        }
        (work_dir / "block3_aux_localization.json").write_text(json.dumps(aux_loc, indent=2), encoding="utf-8")

        # 7. Split validator check (synthetic + repo splits if present)
        synth_split_errors = validate_split_rows({
            "train": [{
                "case_id": "syn_01", "patient_group_id": "pgrp_syn_a", "dataset_id": "synthetic",
                "dataset_release": "v0", "source_uri": "synthetic://01", "split": "train",
                "labels_available": "ICH", "label_scope": "voxel_3d_multiclass",
            }],
            "validation": [{
                "case_id": "syn_02", "patient_group_id": "pgrp_syn_b", "dataset_id": "synthetic",
                "dataset_release": "v0", "source_uri": "synthetic://02", "split": "validation",
                "labels_available": "ICH", "label_scope": "voxel_3d_multiclass",
            }],
            "test": [{
                "case_id": "syn_03", "patient_group_id": "pgrp_syn_c", "dataset_id": "synthetic",
                "dataset_release": "v0", "source_uri": "synthetic://03", "split": "test",
                "labels_available": "ICH", "label_scope": "voxel_3d_multiclass",
            }],
        })
        if synth_split_errors:
            raise RuntimeError(f"Synthetic split check failed: {synth_split_errors}")

        splits_dir = ROOT / "block3" / "data" / "splits"
        split_paths = {s: splits_dir / f"{s}.csv" for s in ("train", "validation", "test") if (splits_dir / f"{s}.csv").exists()}
        repo_splits_status = check_split_files(split_paths) if len(split_paths) == 3 else {"status": "skipped_missing"}

        rss_end = get_process_rss_mb()
        ram_after = get_system_ram_status()
        elapsed = time.perf_counter() - t_total_0

        return {
            "mode": "synthetic_end_to_end",
            "requires_external_datasets": False,
            "elapsed_seconds": round(elapsed, 3),
            "system_ram_before": ram_before,
            "system_ram_after": ram_after,
            "process_memory_mb": {
                "start_rss_mb": rss_start["rss_mb"],
                "end_rss_mb": rss_end["rss_mb"],
                "peak_rss_mb": rss_end["peak_rss_mb"],
            },
            "npz_verified": {"sha256": npz_sha, "size_bytes": npz_bytes, "shape_zyx": list(hu_256.shape)},
            "training_steps_verified": {
                "classifier_arch": "mobilenet_v3_small",
                "classifier_last_loss": round(cls_loss_val, 5),
                "classifier_checkpoint": cls_sha,
                "unet_last_loss": round(seg_loss_val, 5),
                "unet_checkpoint": seg_sha,
            },
            "segmentation_orig_grid_check": {
                "pred_eval": pred_eval,
                "gt_downsampling_ceiling": roundtrip_ceiling,
                "nifti_roundtrip_ok": nifti_roundtrip_ok,
                "nifti_shape_xyz": list(loaded_ich.shape),
            },
            "gt_localization_check": gt_loc,
            "aux_localization": aux_loc,
            "findings_schema_errors": schema_errors,
            "synthetic_splits_check": "passed",
            "repo_splits_check": repo_splits_status,
        }
    finally:
        if ctx is not None:
            ctx.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Block 3 CPU smoke test (Phase 2).")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run self-contained synthetic CT end-to-end smoke test (zero external datasets required).",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=6,
        help="Number of axial slices for --synthetic volume (default: 6).",
    )
    parser.add_argument("--max-slices", type=int, default=None, help="Optional cap on slices for quick unit test.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path (defaults to block3/results/latency/smoke_test_cpu.json only in non-synthetic mode).",
    )
    args = parser.parse_args(argv)

    if args.synthetic:
        report = run_synthetic_pipeline_smoke(num_slices=args.num_slices)
        out_path = Path(args.out) if args.out else None
    else:
        report = run_smoke_benchmark(max_cls_slices=args.max_slices, max_seg_slices=args.max_slices)
        out_path = Path(args.out) if args.out else (ROOT / "block3" / "results" / "latency" / "smoke_test_cpu.json")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
