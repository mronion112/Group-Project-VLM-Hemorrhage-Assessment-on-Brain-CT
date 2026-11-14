"""Batch data preparation for Block 3 (RSNA DICOM series & BHSD NIfTI volumes via HTTP Range).

Workflow per batch (Plan Section 2b):
1. Check disk free_gb >= min_free_gb (50 GB) and block3/data usage <= storage_budget_gb (30 GB).
2. Download batch into `block3/data/raw_staging/<dataset>/<case_id>/` (resumable).
3. Process into compressed `.npz` in `block3/data/processed/<dataset>/<case_id>.npz`:
   - Verify HU metadata (DICOM RescaleSlope/Intercept or verified preconverted HU in BHSD).
   - Order slices strictly by physical axial coordinate.
   - Downsample in-plane to 256x256 (`hu_int16` via bilinear interpolation; `mask_subtype_uint8` via nearest-neighbor).
   - Store original geometry (`orig_shape`, `orig_spacing_mm`, `orig_affine`) so masks can be restored to exact input grid.
4. Append verified record with SHA-256 to `block3/data/processed/manifest.jsonl`.
5. Read back the `.npz` file and verify shape, dtype, finite values, and hash.
6. NEVER deletes raw staging files unless explicitly approved by user (`delete_raw=False` locked by default).
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block3" / "src"))
from remote_zip import fetch_member, http_fetch  # noqa: E402

CONFIG_PATH = ROOT / "block3" / "configs" / "config_v1.json"
SPLITS_DIR = ROOT / "block3" / "data" / "splits"
INV_DIR = ROOT / "block3" / "data" / "inventory"
RAW_DIR = ROOT / "block3" / "data" / "raw_staging"
PROC_DIR = ROOT / "block3" / "data" / "processed"
MANIFEST_PATH = PROC_DIR / "manifest.jsonl"

RSNA_URL = "https://intracranial-hemorrhage.s3.us-west-2.amazonaws.com/rsna-ihd-dataset.zip"
BHSD_URL = "https://huggingface.co/datasets/WuBiao/BHSD/resolve/main/label_192.zip"
BHSD_EXPECTED_ZIP_BYTES = 1446135083
BHSD_EXPECTED_ZIP_SHA256 = "582bf184af993541a4958a4d209a6a44e3bbe702a5daefaf9fb1733a4e7a6e39"
TARGET_HW = (256, 256)
GB = 1024 ** 3


def check_storage_budget(project_root: Path, min_free_gb: float = 50.0, storage_budget_gb: float = 30.0) -> dict:
    disk = shutil.disk_usage(project_root)
    free_gb = disk.free / GB
    data_dir = project_root / "block3" / "data"
    used_gb = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file()) / GB if data_dir.exists() else 0.0
    if free_gb < min_free_gb:
        raise RuntimeError(f"Free disk {free_gb:.2f} GB < min_free_gb ({min_free_gb} GB). Stopping.")
    if used_gb > storage_budget_gb:
        raise RuntimeError(f"Data usage {used_gb:.2f} GB > storage_budget_gb ({storage_budget_gb} GB). Stopping.")
    return {"free_gb": round(free_gb, 2), "data_used_gb": round(used_gb, 3)}


def resize_volume_2d(volume_zyx: np.ndarray, target_hw: tuple[int, int] = TARGET_HW, is_mask: bool = False) -> np.ndarray:
    """Resize a (Z, H, W) array in-plane to target_hw. Order=1 for HU, Order=0 (nearest) for masks."""
    z, h, w = volume_zyx.shape
    if (h, w) == target_hw:
        return volume_zyx
    factors = (1.0, target_hw[0] / h, target_hw[1] / w)
    out = zoom(volume_zyx, factors, order=0 if is_mask else 1)
    return out.astype(volume_zyx.dtype)


def dicom_bytes_to_hu_and_meta(dcm_bytes: bytes) -> tuple[np.ndarray, dict]:
    """Convert a single DICOM slice to HU (int16) ONLY when RescaleSlope & RescaleIntercept exist."""
    ds = pydicom.dcmread(io.BytesIO(dcm_bytes))
    slope = getattr(ds, "RescaleSlope", None)
    intercept = getattr(ds, "RescaleIntercept", None)
    if slope is None or intercept is None:
        raise ValueError("Missing RescaleSlope or RescaleIntercept in DICOM header; refusing to assume HU.")
    raw = ds.pixel_array.astype(np.float32)
    hu = np.rint(raw * float(slope) + float(intercept)).clip(-32768, 32767).astype(np.int16)
    ipp = [float(x) for x in getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])]
    iop = [float(x) for x in getattr(ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])]
    ps = [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])]
    row_cos = np.array(iop[:3], dtype=np.float64)
    col_cos = np.array(iop[3:6], dtype=np.float64)
    normal = np.cross(row_cos, col_cos)
    z_coord = float(np.dot(np.array(ipp, dtype=np.float64), normal))
    return hu, {
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "patient_id": str(getattr(ds, "PatientID", "")),
        "study_id": str(getattr(ds, "StudyInstanceUID", "")),
        "series_id": str(getattr(ds, "SeriesInstanceUID", "")),
        "pixel_spacing_xy": [ps[1], ps[0]],
        "ipp": ipp,
        "iop": iop,
        "z_coord": z_coord,
        "rescale_slope": float(slope),
        "rescale_intercept": float(intercept),
    }


def restore_mask_to_orig_grid(mask_zyx: np.ndarray, orig_shape_zyx: tuple[int, int, int] | list[int] | np.ndarray) -> np.ndarray:
    """Restore a (Z, H_sub, W_sub) predicted or downsampled mask back to (Z_orig, H_orig, W_orig) via nearest-neighbor."""
    target_z, target_h, target_w = (int(orig_shape_zyx[0]), int(orig_shape_zyx[1]), int(orig_shape_zyx[2]))
    if mask_zyx.shape[0] != target_z:
        raise ValueError(f"Axial slice count mismatch: mask has Z={mask_zyx.shape[0]}, orig_shape_zyx has Z={target_z}")
    return resize_volume_2d(mask_zyx, (target_h, target_w), is_mask=True)


def verify_preconverted_hu_array(arr: np.ndarray, source_name: str) -> None:
    """Verify NIfTI voxel values are already in plausible CT HU range and guard against double -1024 offset."""
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    p1, p99 = [float(x) for x in np.percentile(arr, [1, 99])]
    # Out-of-FOV scanner padding can be -2048 or -3024 HU while brain/skull inside FOV has p99 > 100 HU.
    # A double -1024 offset shifts both background (p1 < -1500) and tissue/bone down by 1024 (p99 < 100).
    if p1 < -1500.0 and p99 < 100.0:
        raise ValueError(
            f"{source_name}: voxel min={vmin:.1f}, p1={p1:.1f}, p99={p99:.1f} indicates a double -1024 offset or corrupt HU!"
        )
    if p99 > 20000.0:
        raise ValueError(f"{source_name}: voxel p99={p99:.1f} (max={vmax:.1f}) exceeds plausible HU range!")


def save_and_verify_npz(npz_path: Path, payload: dict) -> tuple[str, int]:
    """Save compressed .npz, compute SHA-256, and read back to verify integrity and mask_type contract."""
    if "mask_type" not in payload:
        raise ValueError("payload must include explicit 'mask_type' ('none' | 'binary_ich_only' | 'multiclass_5_subtypes')")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **payload)
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    size_bytes = npz_path.stat().st_size

    with np.load(npz_path, allow_pickle=False) as loaded:
        hu = loaded["hu_int16"]
        if hu.dtype != np.int16 or hu.ndim != 3 or hu.shape[1:] != TARGET_HW:
            raise ValueError(f"Read-back verification failed for {npz_path}: shape={hu.shape}, dtype={hu.dtype}")
        verify_preconverted_hu_array(hu, str(npz_path.name))
        mask_type = str(loaded["mask_type"].item()) if "mask_type" in loaded else ""
        if mask_type not in {"none", "binary_ich_only", "multiclass_5_subtypes"}:
            raise ValueError(f"Invalid mask_type='{mask_type}' in {npz_path}")
        if mask_type == "none":
            if "mask_ich_uint8" in loaded or "mask_subtype_uint8" in loaded:
                raise ValueError(f"mask_type='none' must not contain mask arrays in {npz_path}")
        elif mask_type == "binary_ich_only":
            if "mask_subtype_uint8" in loaded or "mask_subtype_orig_zyx_uint8" in loaded:
                raise ValueError(f"mask_type='binary_ich_only' MUST NOT contain mask_subtype_* in {npz_path}")
            m = loaded["mask_ich_uint8"]
            if m.dtype != np.uint8 or m.shape != hu.shape or not set(np.unique(m).tolist()).issubset({0, 1}):
                raise ValueError(f"Binary mask verification failed for {npz_path}: shape={m.shape}, dtype={m.dtype}")
        elif mask_type == "multiclass_5_subtypes":
            m_ich = loaded["mask_ich_uint8"]
            m_sub = loaded["mask_subtype_uint8"]
            if m_ich.dtype != np.uint8 or m_ich.shape != hu.shape or not set(np.unique(m_ich).tolist()).issubset({0, 1}):
                raise ValueError(f"mask_ich_uint8 verification failed for {npz_path}")
            if m_sub.dtype != np.uint8 or m_sub.shape != hu.shape or not set(np.unique(m_sub).tolist()).issubset({0, 1, 2, 3, 4, 5}):
                raise ValueError(f"mask_subtype_uint8 verification failed for {npz_path}")
        if "hu_orig_zyx_int16" in loaded:
            hu_orig = loaded["hu_orig_zyx_int16"]
            expected_orig = tuple(int(x) for x in loaded["orig_shape_zyx"])
            if hu_orig.dtype != np.int16 or hu_orig.shape != expected_orig:
                raise ValueError(
                    f"hu_orig_zyx_int16 verification failed for {npz_path}: shape={hu_orig.shape}, expected={expected_orig}, dtype={hu_orig.dtype}"
                )
            verify_preconverted_hu_array(hu_orig, f"{npz_path.name}:hu_orig_zyx_int16")
    return f"sha256:{digest}", size_bytes


def load_done_cases(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    qc_csv = INV_DIR / "bhsd_per_volume_qc.csv"
    qc_done: set[str] = set()
    if qc_csv.exists():
        with qc_csv.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "img_gt_affine_max_diff" in r:
                    qc_done.add(r["case_id"])
    done = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec["case_id"]
        if rec.get("dataset_id") == "bhsd":
            if cid not in qc_done:
                continue
            npz_file = ROOT / rec.get("processed_path", "")
            if not npz_file.exists():
                continue
            try:
                with np.load(npz_file, allow_pickle=False) as d:
                    if "hu_orig_zyx_int16" not in d.files:
                        continue
            except Exception:
                continue
        done.add(cid)
    return done


def append_manifest(manifest_path: Path, entry: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                records[r["case_id"]] = r
    records[entry["case_id"]] = entry
    with manifest_path.open("w", encoding="utf-8") as f:
        for k in sorted(records.keys()):
            f.write(json.dumps(records[k]) + "\n")


def process_rsna_case(case_id: str, slice_rows: list[dict], fetch, allow_network: bool = True) -> dict:
    raw_case_dir = RAW_DIR / "rsna_ich" / case_id
    raw_case_dir.mkdir(parents=True, exist_ok=True)

    slices_data = []
    for r in slice_rows:
        sop = r["sop_instance_uid"]
        dcm_file = raw_case_dir / f"{sop}.dcm"
        if not dcm_file.exists():
            if not allow_network or fetch is None:
                raise RuntimeError(
                    f"RSNA S3 download is paused (rsna_s3_download_paused=True) and {dcm_file} is not in raw_staging."
                )
            blob = fetch_member(fetch, int(r["header_offset"]), int(r["compress_size"]), len(r["zip_member"]))
            dcm_file.write_bytes(blob)
        hu_2d, meta = dicom_bytes_to_hu_and_meta(dcm_file.read_bytes())
        labels = np.array([int(r[k]) for k in ("ICH", "EDH", "SDH", "SAH", "IPH", "IVH")], dtype=np.uint8)
        slices_data.append((meta["z_coord"], int(r["slice_index"]), hu_2d, labels, meta))

    # Sort strictly by physical axial coordinate (fallback to slice_index if identical)
    slices_data.sort(key=lambda x: (x[0], x[1]))
    vol_orig = np.stack([x[2] for x in slices_data], axis=0)  # (Z, H, W)
    slice_labels = np.stack([x[3] for x in slices_data], axis=0)  # (Z, 6)
    exam_labels = (slice_labels.max(axis=0) > 0).astype(np.uint8)  # (6,)

    z_coords = np.array([x[0] for x in slices_data], dtype=np.float32)
    dz = float(np.median(np.abs(np.diff(z_coords)))) if len(z_coords) > 1 else 5.0
    ps_xy = slices_data[0][4]["pixel_spacing_xy"]
    orig_spacing_mm = np.array([ps_xy[0], ps_xy[1], dz], dtype=np.float32)
    orig_shape = np.array(vol_orig.shape, dtype=np.int32)

    vol_256 = resize_volume_2d(vol_orig, TARGET_HW, is_mask=False)
    out_npz = PROC_DIR / "rsna_ich" / f"{case_id}.npz"
    payload = {
        "hu_int16": vol_256,
        "mask_type": np.array("none"),
        "slice_labels_uint8": slice_labels,
        "exam_labels_uint8": exam_labels,
        "slice_z_mm": z_coords,
        "orig_shape_zyx": orig_shape,
        "orig_spacing_xyz_mm": orig_spacing_mm,
        "orig_affine": np.eye(4, dtype=np.float32),
    }
    sha256, size_bytes = save_and_verify_npz(out_npz, payload)
    return {
        "case_id": case_id,
        "dataset_id": "rsna_ich",
        "split": slice_rows[0]["split"],
        "patient_group_id": slice_rows[0]["patient_group_id"],
        "processed_path": str(out_npz.relative_to(ROOT)),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "num_slices": int(vol_256.shape[0]),
        "orig_shape_zyx": orig_shape.tolist(),
        "stored_shape_zyx": list(vol_256.shape),
        "orig_spacing_xyz_mm": [round(float(x), 4) for x in orig_spacing_mm],
        "mask_type": "none",
        "hu_status": "verified_dicom_rescale_slope_intercept",
    }


def inspect_bhsd_volume_qc(
    row: dict,
    stem: str,
    img_nii: nib.Nifti1Image,
    gt_nii: nib.Nifti1Image,
    img_xyz: np.ndarray,
    mask_zyx: np.ndarray,
    slice_labels: np.ndarray,
) -> dict:
    """Compute per-volume HU distribution, geometry, orientation, and anomaly flags for BHSD QC inventory."""
    hu_min = round(float(np.min(img_xyz)), 2)
    hu_max = round(float(np.max(img_xyz)), 2)
    p1, p50, p99 = [round(float(x), 2) for x in np.percentile(img_xyz, [1, 50, 99])]
    fov_voxels = img_xyz[img_xyz > -1500.0]
    fov_p1 = round(float(np.percentile(fov_voxels, 1)), 2) if fov_voxels.size > 0 else p1
    out_of_fov_frac = round(float(np.mean(img_xyz <= -1500.0)), 4)
    zooms_xyz = [round(float(x), 6) for x in img_nii.header.get_zooms()[:3]]
    axcodes = "".join(nib.aff2axcodes(img_nii.affine))
    aff = img_nii.affine.astype(np.float64)
    aff_det = round(float(np.linalg.det(aff[:3, :3])), 6)
    off_diag_max = round(float(np.max(np.abs(aff[:3, :3] - np.diag(np.diag(aff[:3, :3]))))), 6)
    img_gt_aff_diff = round(float(np.max(np.abs(img_nii.affine - gt_nii.affine))), 6)
    slope, inter = img_nii.header.get_slope_inter()
    z_len = int(mask_zyx.shape[0])
    pos_slices = int(slice_labels[:, 0].sum())
    pos_voxels = int(np.count_nonzero(mask_zyx))
    subtype_names = ("EDH", "SDH", "SAH", "IPH", "IVH")
    present_subtypes = [subtype_names[i] for i in range(5) if int(slice_labels[:, i + 1].max()) > 0]

    anomalies: list[str] = []
    n_lt_neg3500 = int(np.count_nonzero(img_xyz < -3500.0))
    n_gt_5000 = int(np.count_nonzero(img_xyz > 5000.0))
    if img_xyz.shape[:2] != (512, 512):
        anomalies.append(f"non_512_inplane:{img_xyz.shape[:2]}")
    if z_len < 20 or z_len > 60:
        anomalies.append(f"unusual_slice_count:{z_len}")
    if zooms_xyz[2] < 2.5 or zooms_xyz[2] > 6.5:
        anomalies.append(f"unusual_z_spacing_mm:{zooms_xyz[2]:.4f}")
    if zooms_xyz[0] < 0.30 or zooms_xyz[0] > 0.65 or abs(zooms_xyz[0] - zooms_xyz[1]) > 1e-3:
        anomalies.append(f"unusual_xy_spacing_mm:{zooms_xyz[0]:.4f}x{zooms_xyz[1]:.4f}")
    if hu_min >= -100.0 or p1 > -200.0:
        anomalies.append(f"non_negative_or_uint12_wrapped_hu:min={hu_min},p1={p1},p50={p50},p99={p99},max={hu_max}")
    elif fov_p1 < -1200.0 or fov_p1 > -500.0 or p99 < 100.0 or p99 > 4000.0:
        anomalies.append(f"hu_percentile_anomaly:fov_p1={fov_p1},p99={p99}")
    if n_lt_neg3500 > 0 or n_gt_5000 > 0:
        anomalies.append(
            f"hu_extreme_outlier_voxels:min={hu_min}(n<-3500={n_lt_neg3500}),max={hu_max}(n>5000={n_gt_5000})"
        )
    if off_diag_max > 0.05:
        anomalies.append(f"tilted_or_oblique_affine:max_off_diag={off_diag_max:.4f}")
    if img_gt_aff_diff > 1e-3:
        anomalies.append(f"img_gt_affine_header_diff:max_diff={img_gt_aff_diff:.4f}")
    if pos_voxels == 0:
        anomalies.append("empty_gt_mask")

    return {
        "case_id": row["case_id"],
        "patient_group_id": row["patient_group_id"],
        "raw_stem": stem,
        "split": row["split"],
        "shape_xyz": list(img_xyz.shape),
        "shape_zyx": [z_len, int(img_xyz.shape[1]), int(img_xyz.shape[0])],
        "num_slices": z_len,
        "dx_mm": zooms_xyz[0],
        "dy_mm": zooms_xyz[1],
        "dz_mm": zooms_xyz[2],
        "orientation_axcodes": axcodes,
        "affine_det": aff_det,
        "affine_max_off_diag": off_diag_max,
        "img_gt_affine_max_diff": img_gt_aff_diff,
        "raw_dtype": str(img_nii.get_data_dtype()),
        "scl_slope": None if slope is None or np.isnan(slope) else float(slope),
        "scl_inter": None if inter is None or np.isnan(inter) else float(inter),
        "hu_min": hu_min,
        "hu_p1": p1,
        "fov_hu_p1": fov_p1,
        "out_of_fov_frac": out_of_fov_frac,
        "hu_p50": p50,
        "hu_p99": p99,
        "hu_max": hu_max,
        "voxels_lt_neg3500": n_lt_neg3500,
        "voxels_gt_5000": n_gt_5000,
        "pos_slices": pos_slices,
        "neg_slices": z_len - pos_slices,
        "pos_voxels": pos_voxels,
        "subtypes": ",".join(present_subtypes),
        "anomaly_count": len(anomalies),
        "anomalies": ";".join(anomalies),
    }


def process_bhsd_case(
    row: dict,
    bhsd_zip_map: dict[str, dict],
    fetch,
    local_zf: zipfile.ZipFile | None = None,
) -> tuple[dict, dict]:
    case_id = row["case_id"]
    stem = row["source_uri"].split("/")[-1].replace(".nii.gz", "")
    img_member = f"label_192/images/{stem}.nii.gz"
    gt_member = f"label_192/ground truths/{stem}.nii.gz"

    if local_zf is not None:
        img_gz_bytes = local_zf.read(img_member)
        gt_gz_bytes = local_zf.read(gt_member)
    else:
        raw_case_dir = RAW_DIR / "bhsd" / case_id
        raw_case_dir.mkdir(parents=True, exist_ok=True)
        img_path = raw_case_dir / f"{stem}_img.nii.gz"
        gt_path = raw_case_dir / f"{stem}_gt.nii.gz"
        if not img_path.exists():
            m = bhsd_zip_map[img_member]
            img_path.write_bytes(fetch_member(fetch, int(m["header_offset"]), int(m["compress_size"]), len(img_member)))
        if not gt_path.exists():
            m = bhsd_zip_map[gt_member]
            gt_path.write_bytes(fetch_member(fetch, int(m["header_offset"]), int(m["compress_size"]), len(gt_member)))
        img_gz_bytes = img_path.read_bytes()
        gt_gz_bytes = gt_path.read_bytes()

    img_nii = nib.Nifti1Image.from_bytes(gzip.decompress(img_gz_bytes))
    gt_nii = nib.Nifti1Image.from_bytes(gzip.decompress(gt_gz_bytes))

    # Convert (X, Y, Z) -> (Z, Y, X) for uniform axial slice ordering; DO NOT apply -1024 offset again
    img_xyz = np.asarray(img_nii.dataobj, dtype=np.float32)
    verify_preconverted_hu_array(img_xyz, f"BHSD:{case_id}")
    gt_xyz = np.asarray(gt_nii.dataobj, dtype=np.uint8)
    if img_xyz.shape != gt_xyz.shape:
        raise ValueError(f"BHSD voxel shape mismatch for {case_id}: img={img_xyz.shape}, gt={gt_xyz.shape}")

    vol_zyx = np.transpose(np.rint(img_xyz).clip(-32768, 32767).astype(np.int16), (2, 1, 0))
    mask_zyx = np.transpose(gt_xyz, (2, 1, 0))
    mask_ich_orig_zyx = (mask_zyx > 0).astype(np.uint8)

    # Map BHSD voxel labels (1=EDH, 2=IPH, 3=IVH, 4=SAH, 5=SDH) to canonical slice/exam labels
    # Canonical order: [ICH, EDH, SDH, SAH, IPH, IVH]
    bhsd_to_canonical = {1: 1, 5: 2, 4: 3, 2: 4, 3: 5}
    z_len = mask_zyx.shape[0]
    slice_labels = np.zeros((z_len, 6), dtype=np.uint8)
    for z in range(z_len):
        u = set(np.unique(mask_zyx[z]).tolist()) - {0}
        if u:
            slice_labels[z, 0] = 1
            for b_code in u:
                if b_code in bhsd_to_canonical:
                    slice_labels[z, bhsd_to_canonical[b_code]] = 1
    exam_labels = (slice_labels.max(axis=0) > 0).astype(np.uint8)

    qc_rec = inspect_bhsd_volume_qc(row, stem, img_nii, gt_nii, img_xyz, mask_zyx, slice_labels)

    vol_256 = resize_volume_2d(vol_zyx, TARGET_HW, is_mask=False)
    mask_256 = resize_volume_2d(mask_zyx, TARGET_HW, is_mask=True)
    mask_ich_256 = (mask_256 > 0).astype(np.uint8)
    zooms_xyz = np.array(img_nii.header.get_zooms()[:3], dtype=np.float32)
    orig_shape_zyx = np.array(vol_zyx.shape, dtype=np.int32)

    out_npz = PROC_DIR / "bhsd" / f"{case_id}.npz"
    payload = {
        "hu_int16": vol_256,
        "hu_orig_zyx_int16": vol_zyx,
        "mask_type": np.array("multiclass_5_subtypes"),
        "mask_ich_uint8": mask_ich_256,
        "mask_subtype_uint8": mask_256,
        "mask_ich_orig_zyx_uint8": mask_ich_orig_zyx,
        "mask_subtype_orig_zyx_uint8": mask_zyx,
        "slice_labels_uint8": slice_labels,
        "exam_labels_uint8": exam_labels,
        "orig_shape_zyx": orig_shape_zyx,
        "orig_spacing_xyz_mm": zooms_xyz,
        "orig_affine": img_nii.affine.astype(np.float32),
    }
    sha256, size_bytes = save_and_verify_npz(out_npz, payload)
    manifest_entry = {
        "case_id": case_id,
        "dataset_id": "bhsd",
        "split": row["split"],
        "patient_group_id": row["patient_group_id"],
        "processed_path": str(out_npz.relative_to(ROOT)),
        "sha256": sha256,
        "size_bytes": size_bytes,
        "num_slices": int(vol_256.shape[0]),
        "orig_shape_zyx": orig_shape_zyx.tolist(),
        "stored_shape_zyx": list(vol_256.shape),
        "has_hu_orig_512": True,
        "orig_spacing_xyz_mm": [round(float(x), 4) for x in zooms_xyz],
        "orientation_axcodes": qc_rec["orientation_axcodes"],
        "hu_min_max_p1_p99": [qc_rec["hu_min"], qc_rec["hu_max"], qc_rec["hu_p1"], qc_rec["hu_p99"]],
        "mask_type": "multiclass_5_subtypes",
        "hu_status": "verified_preconverted_hu_nifti",
    }
    return manifest_entry, qc_rec


def save_bhsd_qc_inventory(qc_rows: list[dict], zip_verification: dict | None = None) -> dict:
    """Save per-volume QC CSV and aggregate QC + anomaly summary JSON to block3/data/inventory/."""
    INV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = INV_DIR / "bhsd_per_volume_qc.csv"
    existing: dict[str, dict] = {}
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[r["case_id"]] = r
    for r in qc_rows:
        existing[r["case_id"]] = r

    ordered = [existing[k] for k in sorted(existing.keys())]
    if ordered:
        fieldnames = list(ordered[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(ordered)

    # Compute summary across all recorded volumes
    n_vols = len(ordered)
    hu_mins = [float(r["hu_min"]) for r in ordered]
    hu_maxs = [float(r["hu_max"]) for r in ordered]
    hu_p1s = [float(r["hu_p1"]) for r in ordered]
    hu_p99s = [float(r["hu_p99"]) for r in ordered]
    dxs = [float(r["dx_mm"]) for r in ordered]
    dys = [float(r["dy_mm"]) for r in ordered]
    dzs = [float(r["dz_mm"]) for r in ordered]
    slices = [int(r["num_slices"]) for r in ordered]
    ax_counts: dict[str, int] = {}
    shape_xy_counts: dict[str, int] = {}
    flagged = []
    for r in ordered:
        ax = str(r["orientation_axcodes"])
        ax_counts[ax] = ax_counts.get(ax, 0) + 1
        s_zyx = r["shape_zyx"] if isinstance(r["shape_zyx"], list) else json.loads(str(r["shape_zyx"]))
        hw_key = f"{s_zyx[1]}x{s_zyx[2]}"
        shape_xy_counts[hw_key] = shape_xy_counts.get(hw_key, 0) + 1
        if str(r.get("anomalies", "")).strip():
            flagged.append({
                "case_id": r["case_id"],
                "raw_stem": r["raw_stem"],
                "split": r["split"],
                "num_slices": int(r["num_slices"]),
                "dx_mm": float(r["dx_mm"]),
                "dy_mm": float(r["dy_mm"]),
                "dz_mm": float(r["dz_mm"]),
                "orientation_axcodes": ax,
                "hu_min": float(r["hu_min"]),
                "hu_p1": float(r["hu_p1"]),
                "hu_p99": float(r["hu_p99"]),
                "hu_max": float(r["hu_max"]),
                "anomalies": str(r["anomalies"]),
            })

    summary = {
        "dataset_id": "bhsd",
        "zip_verification": zip_verification,
        "total_volumes_audited": n_vols,
        "inplane_shape_counts": shape_xy_counts,
        "orientation_axcodes_counts": ax_counts,
        "slice_count_range": {
            "min": int(min(slices)) if slices else 0,
            "median": float(np.median(slices)) if slices else 0.0,
            "max": int(max(slices)) if slices else 0,
            "total_slices": int(sum(slices)),
        },
        "spacing_mm_range": {
            "dx_min_max": [round(min(dxs), 6), round(max(dxs), 6)] if dxs else [],
            "dy_min_max": [round(min(dys), 6), round(max(dys), 6)] if dys else [],
            "dz_min_max": [round(min(dzs), 6), round(max(dzs), 6)] if dzs else [],
            "dz_median": round(float(np.median(dzs)), 6) if dzs else 0.0,
        },
        "hu_range_across_volumes": {
            "global_min_hu": round(min(hu_mins), 2) if hu_mins else 0.0,
            "global_max_hu": round(max(hu_maxs), 2) if hu_maxs else 0.0,
            "p1_range_hu": [round(min(hu_p1s), 2), round(max(hu_p1s), 2)] if hu_p1s else [],
            "p99_range_hu": [round(min(hu_p99s), 2), round(max(hu_p99s), 2)] if hu_p99s else [],
            "all_verified_preconverted_hu": True,
        },
        "anomalous_volumes_count": len(flagged),
        "anomalous_volumes": flagged,
    }
    (INV_DIR / "bhsd_volumes_qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_bhsd_zip_map(listing_path: Path, fetch=None, size: int | None = None, url: str = BHSD_URL) -> dict[str, dict]:
    """Load BHSD ZIP entry map from CSV if present, or fetch central directory via HTTP Range."""
    if listing_path.exists():
        with listing_path.open(encoding="utf-8") as f:
            return {r["name"]: r for r in csv.DictReader(f)}
    from remote_zip import HttpRangeFile, http_size
    total_size = size if size is not None else http_size(url, timeout=60)
    f_fetch = fetch or http_fetch(url, timeout=120)
    with zipfile.ZipFile(HttpRangeFile(f_fetch, total_size)) as zf:
        return {
            info.filename: {
                "name": info.filename,
                "compress_size": str(info.compress_size),
                "file_size": str(info.file_size),
                "compress_type": str(info.compress_type),
                "header_offset": str(info.header_offset),
            }
            for info in zf.infolist()
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Block 3 datasets in controlled batches.")
    parser.add_argument("--dataset", choices=["rsna_ich", "bhsd"], required=True)
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=2, help="Max cases to process in this batch run.")
    args = parser.parse_args(argv)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    min_free = float(cfg["storage"]["min_free_gb"] or 50.0)
    budget = float(cfg["storage"]["storage_budget_gb"] or 30.0)
    budget_status = check_storage_budget(ROOT, min_free_gb=min_free, storage_budget_gb=budget)
    print("Storage check before batch:", budget_status)

    done_cases = load_done_cases(MANIFEST_PATH)

    if args.dataset == "rsna_ich":
        rsna_paused = bool(cfg["storage"].get("rsna_s3_download_paused", False))
        by_case: dict[str, list[dict]] = {}
        with (SPLITS_DIR / "rsna_selected_slices.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if args.split != "all" and r["split"] != args.split:
                    continue
                by_case.setdefault(r["case_id"], []).append(r)
        pending = [cid for cid in sorted(by_case.keys()) if cid not in done_cases][:args.limit]
        fetch = None if rsna_paused else http_fetch(RSNA_URL, timeout=120)
        for cid in pending:
            check_storage_budget(ROOT, min_free_gb=min_free, storage_budget_gb=budget)
            entry = process_rsna_case(cid, by_case[cid], fetch, allow_network=not rsna_paused)
            append_manifest(MANIFEST_PATH, entry)
            print("Processed RSNA case:", entry["case_id"], "slices:", entry["num_slices"], "bytes:", entry["size_bytes"])
    else:
        local_zip_path = RAW_DIR / "bhsd" / "label_192.zip"
        zip_ver = None
        local_zf = None
        fetch = None
        bhsd_zip_map: dict[str, dict] = {}
        if local_zip_path.exists():
            actual_sz = local_zip_path.stat().st_size
            if actual_sz != BHSD_EXPECTED_ZIP_BYTES:
                raise RuntimeError(
                    f"Local label_192.zip size mismatch: got {actual_sz}, expected {BHSD_EXPECTED_ZIP_BYTES}"
                )
            h = hashlib.sha256()
            with local_zip_path.open("rb") as zf_in:
                while chunk := zf_in.read(8 * 1024 * 1024):
                    h.update(chunk)
            actual_sha = h.hexdigest()
            if actual_sha != BHSD_EXPECTED_ZIP_SHA256:
                raise RuntimeError(
                    f"Local label_192.zip SHA-256 mismatch: got {actual_sha}, expected {BHSD_EXPECTED_ZIP_SHA256}"
                )
            zip_ver = {
                "path": str(local_zip_path.relative_to(ROOT)),
                "size_bytes": actual_sz,
                "expected_size_bytes": BHSD_EXPECTED_ZIP_BYTES,
                "sha256": f"sha256:{actual_sha}",
                "expected_sha256": f"sha256:{BHSD_EXPECTED_ZIP_SHA256}",
                "verified_match": True,
                "preserved_on_disk": True,
            }
            print("Verified local BHSD zip:", json.dumps(zip_ver))
            local_zf = zipfile.ZipFile(local_zip_path, "r")
        else:
            fetch = http_fetch(BHSD_URL, timeout=120)
            bhsd_zip_map = load_bhsd_zip_map(INV_DIR / "bhsd_zip_listing.csv", fetch=fetch)

        qc_rows: list[dict] = []
        try:
            bhsd_rows = []
            for sp in ("train", "validation", "test"):
                if args.split != "all" and sp != args.split:
                    continue
                with (SPLITS_DIR / f"{sp}.csv").open(encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        if r["dataset_id"] == "bhsd" and r["case_id"] not in done_cases:
                            bhsd_rows.append(r)
            for idx, r in enumerate(bhsd_rows[:args.limit], start=1):
                check_storage_budget(ROOT, min_free_gb=min_free, storage_budget_gb=budget)
                entry, qc_rec = process_bhsd_case(r, bhsd_zip_map, fetch, local_zf=local_zf)
                append_manifest(MANIFEST_PATH, entry)
                qc_rows.append(qc_rec)
                if idx == 1 or idx % 20 == 0 or idx == len(bhsd_rows[:args.limit]):
                    print(
                        f"[{idx}/{len(bhsd_rows[:args.limit])}] Processed BHSD case: {entry['case_id']} "
                        f"slices={entry['num_slices']} bytes={entry['size_bytes']} anomalies={qc_rec['anomaly_count']}"
                    )
        finally:
            if qc_rows or zip_ver is not None:
                qc_summary = save_bhsd_qc_inventory(qc_rows, zip_verification=zip_ver)
                print(
                    "BHSD QC summary:",
                    json.dumps(
                        {
                            "total_volumes_audited": qc_summary["total_volumes_audited"],
                            "anomalous_volumes_count": qc_summary["anomalous_volumes_count"],
                            "inplane_shape_counts": qc_summary["inplane_shape_counts"],
                            "orientation_axcodes_counts": qc_summary["orientation_axcodes_counts"],
                        }
                    ),
                )
            if local_zf is not None:
                local_zf.close()

    print("Storage check after batch:", check_storage_budget(ROOT, min_free_gb=min_free, storage_budget_gb=budget))
    return 0


if __name__ == "__main__":
    sys.exit(main())
