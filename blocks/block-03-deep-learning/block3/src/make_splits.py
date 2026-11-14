"""Deterministically build patient-level splits for RSNA (controlled subset) and BHSD (192 volumes).

Follows docs/INTERFACE_CONTRACT.md and blocks/block-01-dataset-split/Guide.md:
- Groups strictly by pseudonymized `patient_group_id` BEFORE any slice extraction.
- Prevents cross-dataset patient leakage between RSNA and BHSD (since BHSD volumes originate from RSNA IDs).
- Derives RSNA exam-level labels from slice labels (exam positive iff >=1 slice positive).
"""
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block3" / "src"))
from check_splits import REQUIRED_COLUMNS, check_split_files  # noqa: E402

RAW_DIR = ROOT / "block3" / "data" / "raw_staging"
INV_DIR = ROOT / "block3" / "data" / "inventory"
SPLITS_DIR = ROOT / "block3" / "data" / "splits"

SALT = "hoopstats-vlm-ich-block3-2026:"
SUBTYPES = ("EDH", "SDH", "SAH", "IPH", "IVH")
RAW_TO_SUBTYPE = {
    "epidural": "EDH",
    "subdural": "SDH",
    "subarachnoid": "SAH",
    "intraparenchymal": "IPH",
    "intraventricular": "IVH",
}


def pseudonymize_patient(raw_patient_id: str) -> str:
    digest = hashlib.sha256((SALT + raw_patient_id.strip()).encode("utf-8")).hexdigest()[:16]
    return f"pgrp_{digest}"


def pseudonymize_case(dataset_prefix: str, raw_case_id: str) -> str:
    digest = hashlib.sha256((SALT + dataset_prefix + ":" + raw_case_id.strip()).encode("utf-8")).hexdigest()[:16]
    return f"{dataset_prefix}_{digest}"


def stratify_patients(patient_ids: list[str], patient_strata: dict[str, str], seed: int,
                      ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> dict[str, str]:
    """Assign each patient_group_id to train/validation/test stratified by patient_strata."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for pid in sorted(set(patient_ids)):
        by_stratum[patient_strata.get(pid, "other")].append(pid)

    assignment: dict[str, str] = {}
    for stratum in sorted(by_stratum.keys()):
        group = by_stratum[stratum][:]
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_val = min(max(1, n_val), n - n_train - 1)
        for p in group[:n_train]:
            assignment[p] = "train"
        for p in group[n_train:n_train + n_val]:
            assignment[p] = "validation"
        for p in group[n_train + n_val:]:
            assignment[p] = "test"
    return assignment


def build_bhsd_records(bhsd_listing_csv: Path) -> tuple[list[dict], set[str]]:
    """Return BHSD case records (enriched with verified ground-truth subtype flags) and raw RSNA patient IDs."""
    imgs = {}
    with bhsd_listing_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            if name.startswith("label_192/images/") and name.endswith(".nii.gz"):
                stem = Path(name).name.replace(".nii.gz", "")
                imgs[stem] = row

    gt_summary_path = INV_DIR / "bhsd_ground_truth_summary.json"
    vol_meta = {}
    if gt_summary_path.exists():
        vol_meta = json.loads(gt_summary_path.read_text(encoding="utf-8")).get("volumes", {})

    rarity_order = ("EDH", "SDH", "IVH", "SAH", "IPH")
    patient_subtypes: dict[str, set[str]] = defaultdict(set)
    for stem in sorted(imgs.keys()):
        parts = stem.split("_")
        raw_pid = f"{parts[0]}_{parts[1]}"
        pgrp = pseudonymize_patient(raw_pid)
        subs = set(vol_meta.get(stem, {}).get("subtypes", []))
        patient_subtypes[pgrp].update(subs)

    records = []
    raw_pids = set()
    for stem in sorted(imgs.keys()):
        parts = stem.split("_")
        raw_pid = f"{parts[0]}_{parts[1]}"
        raw_pids.add(raw_pid)
        pgrp = pseudonymize_patient(raw_pid)
        case_id = pseudonymize_case("bhsd", stem)
        vm = vol_meta.get(stem, {})
        subs = set(vm.get("subtypes", []))
        p_subs = patient_subtypes[pgrp]
        chosen_stratum = "bhsd_other"
        for st in rarity_order:
            if st in p_subs:
                chosen_stratum = f"bhsd_{st}"
                break
        records.append({
            "case_id": case_id,
            "raw_case_id": stem,
            "patient_group_id": pgrp,
            "dataset_id": "bhsd",
            "dataset_release": "WuBiao-BHSD-label192-2023",
            "source_uri": f"hf://WuBiao/BHSD/label_192.zip#label_192/images/{stem}.nii.gz",
            "labels_available": "ICH,EDH,SDH,SAH,IPH,IVH,MASK_ICH,MASK_SUBTYPE",
            "label_scope": "voxel_3d_multiclass",
            "stratum": chosen_stratum,
            "num_slices": int(vm.get("slices", 0)),
            "ICH": 1 if subs else 0,
            "EDH": 1 if "EDH" in subs else 0,
            "SDH": 1 if "SDH" in subs else 0,
            "SAH": 1 if "SAH" in subs else 0,
            "IPH": 1 if "IPH" in subs else 0,
            "IVH": 1 if "IVH" in subs else 0,
        })
    return records, raw_pids


def build_rsna_controlled_subset(
    slice_labels_csv: Path,
    exclude_raw_pids: set[str],
    seed: int = 2026,
    target_patients_per_stratum: int = 50,
) -> tuple[list[dict], list[dict]]:
    """Select a balanced, controlled patient-level subset of RSNA (300 patients, 6 strata x 50)."""
    # Group slices by study_ID and map study_ID -> patient_ID
    study_meta: dict[str, dict] = {}
    study_slices: dict[str, list[dict]] = defaultdict(list)
    patient_studies: dict[str, list[str]] = defaultdict(list)

    with slice_labels_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw_pid = r["patient_ID"]
            if raw_pid in exclude_raw_pids:
                continue
            sid = r["study_ID"]
            if sid not in study_meta:
                study_meta[sid] = {
                    "raw_pid": raw_pid,
                    "series_ID": r["series_ID"],
                    "any": 0,
                    "EDH": 0,
                    "SDH": 0,
                    "SAH": 0,
                    "IPH": 0,
                    "IVH": 0,
                    "num_slices": 0,
                }
                patient_studies[raw_pid].append(sid)
            m = study_meta[sid]
            m["num_slices"] += 1
            if int(r["any"]) == 1:
                m["any"] = 1
            for raw_k, code in RAW_TO_SUBTYPE.items():
                if int(r[raw_k]) == 1:
                    m[code] = 1
            study_slices[sid].append(r)

    # Filter to patients with single study of typical axial slice count (24..48 slices) for clean 3D reconstruction
    # and assign each patient to rarest positive subtype (rarity order: EDH, IVH, SAH, IPH, SDH, negative)
    rarity_order = ["EDH", "IVH", "SAH", "IPH", "SDH", "negative"]
    candidates_by_stratum: dict[str, list[str]] = defaultdict(list)

    for raw_pid in sorted(patient_studies.keys()):
        sids = patient_studies[raw_pid]
        if len(sids) != 1:
            continue
        sid = sids[0]
        m = study_meta[sid]
        if not (24 <= m["num_slices"] <= 48):
            continue
        chosen_stratum = "negative"
        if m["any"] == 1:
            for st in ("EDH", "IVH", "SAH", "IPH", "SDH"):
                if m[st] == 1:
                    chosen_stratum = st
                    break
        candidates_by_stratum[chosen_stratum].append(raw_pid)

    rng = random.Random(seed)
    selected_pids: list[tuple[str, str]] = []
    for stratum in rarity_order:
        pool = candidates_by_stratum[stratum][:]
        rng.shuffle(pool)
        for pid in pool[:target_patients_per_stratum]:
            selected_pids.append((pid, stratum))

    case_records = []
    selected_slice_rows = []
    for raw_pid, stratum in selected_pids:
        sid = patient_studies[raw_pid][0]
        m = study_meta[sid]
        pgrp = pseudonymize_patient(raw_pid)
        case_id = pseudonymize_case("rsna", sid)
        case_records.append({
            "case_id": case_id,
            "raw_case_id": sid,
            "patient_group_id": pgrp,
            "dataset_id": "rsna_ich",
            "dataset_release": "RSNA-ICH-stage2-2019",
            "source_uri": f"s3://intracranial-hemorrhage/rsna-ihd-dataset.zip#study={sid}",
            "labels_available": "ICH,EDH,SDH,SAH,IPH,IVH",
            "label_scope": "slice_2d_and_exam_derived",
            "stratum": f"rsna_{stratum}",
            "num_slices": m["num_slices"],
            "ICH": m["any"],
            "EDH": m["EDH"],
            "SDH": m["SDH"],
            "SAH": m["SAH"],
            "IPH": m["IPH"],
            "IVH": m["IVH"],
        })
        for s_row in study_slices[sid]:
            fname = s_row["filename"]
            # IM000001_ID_45785016b.png
            slice_idx = int(fname.split("_")[0].replace("IM", "")) - 1
            sop_id = "ID_" + fname.split("_ID_")[1].replace(".png", "")
            selected_slice_rows.append({
                "case_id": case_id,
                "patient_group_id": pgrp,
                "study_id": sid,
                "series_id": s_row["series_ID"],
                "slice_index": slice_idx,
                "sop_instance_uid": sop_id,
                "zip_member": f"rsna-intracranial-hemorrhage-detection/stage_2_train/{sop_id}.dcm",
                "ICH": int(s_row["any"]),
                "EDH": int(s_row["epidural"]),
                "SDH": int(s_row["subdural"]),
                "SAH": int(s_row["subarachnoid"]),
                "IPH": int(s_row["intraparenchymal"]),
                "IVH": int(s_row["intraventricular"]),
            })

    return case_records, selected_slice_rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    seed = 2026
    cfg_path = ROOT / "block3" / "configs" / "config_v1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    rsna_enabled = bool(cfg.get("rsna_enabled", False))

    bhsd_listing = INV_DIR / "bhsd_zip_listing.csv"
    rsna_slice_csv = RAW_DIR / "rsna_slice_labels.csv"

    bhsd_records, bhsd_raw_pids = build_bhsd_records(bhsd_listing)
    if rsna_enabled:
        rsna_records, rsna_slice_rows = build_rsna_controlled_subset(
            rsna_slice_csv, exclude_raw_pids=bhsd_raw_pids, seed=seed, target_patients_per_stratum=50
        )
    else:
        rsna_records, rsna_slice_rows = [], []

    all_records = rsna_records + bhsd_records
    patient_strata = {r["patient_group_id"]: r["stratum"] for r in all_records}
    assignment = stratify_patients(list(patient_strata.keys()), patient_strata, seed=seed)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    extra_cols = ("num_slices", "ICH", "EDH", "SDH", "SAH", "IPH", "IVH", "stratum")

    for rec in all_records:
        sp = assignment[rec["patient_group_id"]]
        rec["split"] = sp
        by_split[sp].append(rec)

    fieldnames = list(REQUIRED_COLUMNS) + [c for c in extra_cols if c not in REQUIRED_COLUMNS]
    split_paths = {}
    for sp, rows in by_split.items():
        out_path = SPLITS_DIR / f"{sp}.csv"
        split_paths[sp] = out_path
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in sorted(rows, key=lambda x: x["case_id"]):
                writer.writerow(r)

    slice_index_path = SPLITS_DIR / "rsna_selected_slices.csv"
    if rsna_enabled and rsna_slice_rows:
        needed_members = {r["zip_member"] for r in rsna_slice_rows}
        member_zip_info = {}
        with (INV_DIR / "rsna_zip_listing.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["name"] in needed_members:
                    member_zip_info[row["name"]] = (int(row["header_offset"]), int(row["compress_size"]), int(row["file_size"]))

        slice_fields = [
            "case_id", "patient_group_id", "split", "slice_index", "sop_instance_uid",
            "zip_member", "header_offset", "compress_size", "file_size",
            "ICH", "EDH", "SDH", "SAH", "IPH", "IVH",
        ]
        with slice_index_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=slice_fields)
            w.writeheader()
            for r in sorted(rsna_slice_rows, key=lambda x: (x["case_id"], x["slice_index"])):
                off, csz, fsz = member_zip_info[r["zip_member"]]
                r["split"] = assignment[r["patient_group_id"]]
                r["header_offset"] = off
                r["compress_size"] = csz
                r["file_size"] = fsz
                w.writerow({k: r[k] for k in slice_fields})
    elif slice_index_path.exists():
        archived = INV_DIR / "rsna_surveyed_not_used_slices.csv"
        if not archived.exists():
            slice_index_path.replace(archived)
        else:
            slice_index_path.unlink()

    # Verify zero patient overlap
    check_report = check_split_files(split_paths)
    if check_report["status"] != "passed":
        print("SPLIT CHECK FAILED:", check_report)
        return 1

    checksums = {sp: f"sha256:{sha256_file(p)}" for sp, p in split_paths.items()}
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    split_config = {
        "schema_version": "1.0",
        "created_at_utc": now_utc,
        "seed": seed,
        "rsna_enabled": rsna_enabled,
        "group_key": "patient_group_id",
        "stratification": "rarest_positive_subtype_per_patient",
        "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "exclusions": [
            "RSNA excluded from splits, manifest, and experiments per Phase 2 decision (rsna_enabled=false; surveyed_not_used).",
            "BHSD 192 volumes mapped to 191 unique patients (patient ID_2db7ee14 has 2 volumes grouped together).",
        ],
        "script": "block3/src/make_splits.py",
        "split_checksums": checksums,
        "counts": check_report["counts"],
    }
    (SPLITS_DIR / "split_config.json").write_text(json.dumps(split_config, indent=2), encoding="utf-8")

    dataset_manifest = {
        "schema_version": "1.0",
        "created_at_utc": now_utc,
        "source_artifact_ids": ["block3/data/splits/split_config.json"],
        "dataset_id": "bhsd" if not rsna_enabled else "rsna_ich+bhsd",
        "dataset_release": "BHSD-2023-label192" if not rsna_enabled else "RSNA-2019_BHSD-2023",
        "status": "ok",
        "license_status": "verified-for-project-use",
        "cohort_config": "block3-bhsd192-patient-split-v2",
        "split_config": {
            "seed": seed,
            "group_key": "patient_group_id",
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        },
        "split_manifests": {
            "train": "splits/train.csv",
            "validation": "splits/validation.csv",
            "test": "splits/test.csv",
        },
        "record_count": len(all_records),
        "patient_overlap_check": {"status": "passed", "overlap_count": 0},
        "warnings": [],
        "errors": [],
    }
    (INV_DIR / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")

    print(f"Splits created successfully: total_cases={len(all_records)} (RSNA={len(rsna_records)}, BHSD={len(bhsd_records)})")
    print("Split check report:", json.dumps(check_report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
