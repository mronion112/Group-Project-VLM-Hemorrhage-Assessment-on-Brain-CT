"""Generate complete dataset inventory (CSV + JSON) from verified facts and headers."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INV_DIR = ROOT / "block3" / "data" / "inventory"


def build_inventory() -> list[dict]:
    now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return [
        {
            "dataset_id": "rsna_ich",
            "name": "RSNA Intracranial Hemorrhage Detection (Stage 2)",
            "source_url": "s3://intracranial-hemorrhage/rsna-ihd-dataset.zip (us-west-2, public HTTP Range)",
            "release_version": "Stage 2 (2019-11 / S3 zip 2024-03-27)",
            "audit_date": now_date,
            "license": "RSNA Competition Rules / Non-Commercial Research & Education (citation required: Flanders et al. 2020, DOI:10.1148/ryai.2020190211)",
            "access_status": "surveyed_not_used_rsna_enabled_false",
            "total_cases_or_volumes": "21744 train studies (18938 unique patients) + test studies",
            "total_slices": "752803 train DICOMs + 121232 test DICOMs (874035 total)",
            "voxel_spacing_mm": "In-plane ~0.45-0.50 mm (PixelSpacing verified in DICOM header); z-spacing ~3.0-5.0 mm derived from ImagePositionPatient",
            "hu_representation": "VERIFIED_RAW_STORED_WITH_RESCALE: 16-bit signed stored pixels with explicit RescaleSlope (1.0) and RescaleIntercept (-1024 or -33792) in DICOM header -> converts directly to raw HU",
            "patient_identifier": "DICOM tag (0010,0020) PatientID (e.g. ID_f15c0eee), StudyInstanceUID, SeriesInstanceUID; pseudonymized to pgrp_<sha256>",
            "label_type": "Slice-level multi-label binary (any, epidural, subdural, subarachnoid, intraparenchymal, intraventricular) in stage_2_train.csv; exam-level derived (positive iff >=1 positive slice); NO pixel mask",
            "official_split": "Surveyed only; excluded from active Block 3 splits and manifest (rsna_enabled=false)",
            "size_compressed_gb": 194.67,
            "size_uncompressed_gb": 458.83,
            "selected_subset_size_gb": "0.275 GiB on disk (zip listing + CSVs + 1 surveyed case rsna_0007c6017e8b082b; separated and not used)",
            "role_in_block3": "Surveyed in Phase 1 (1 sample case downloaded), NOT used in splits, manifest, or training/evaluation per Phase 2 decision (rsna_enabled=false)",
        },
        {
            "dataset_id": "bhsd",
            "name": "Brain Hemorrhage Segmentation Dataset (BHSD)",
            "source_url": "https://huggingface.co/datasets/WuBiao/BHSD/resolve/main/label_192.zip",
            "release_version": "MICCAI MLMI 2023 (label_192.zip, 1,446,135,083 bytes, SHA-256: 582bf184af993541a4958a4d209a6a44e3bbe702a5daefaf9fb1733a4e7a6e39, preserved on disk)",
            "audit_date": now_date,
            "license": "LICENSE DISCREPANCY DOCUMENTED: Hugging Face repo card (cardData.license) and README.md state 'mit' (MIT License), whereas dataset.json inside the release states 'AIML'. Furthermore, BHSD CT scans originate from RSNA 2019 ICH Challenge, so BOTH BHSD (Wu et al. 2023) and RSNA (Flanders et al. 2020) must be cited, and usage is strictly restricted to non-commercial academic research. Note: 'rsna_enabled=false' means excluding the original RSNA 2D-labeled dataset on S3; BHSD still contains underlying CT volumes sourced from RSNA.",
            "citation": "Wu B., et al. 'BHSD: A 3D Multi-class Brain Hemorrhage Segmentation Dataset', MICCAI MLMI 2023, LNCS 14348, pp. 147-156 (DOI: 10.1007/978-3-031-45673-2_15); and Flanders A.E., et al., Radiology: AI 2020 (DOI: 10.1148/ryai.2020190211).",
            "access_status": "verified_local_zip_sha256_and_192_of_192_volumes_processed",
            "total_cases_or_volumes": "192 3D NIfTI volumes + 192 3D multi-class ground-truth masks belonging to 191 unique patients (1 patient ID_2db7ee14 has 2 volumes: ID_2db7ee14_ID_2591d00dfc and ID_2db7ee14_ID_f27e38fdfd; 190 patients have 1 volume). Volume subtype counts: IPH=127, SAH=109, IVH=104, SDH=70, EDH=23 (0 negative volumes).",
            "total_slices": "6404 axial slices across 192 volumes (24-60 slices/vol, median 32.0; 192/192 in-plane 512x512; 2368 positive slices = 36.98%, 4036 negative slices = 63.02%; slice subtype counts: SAH=976, IPH=888, SDH=765, IVH=713, EDH=181)",
            "voxel_spacing_mm": "Audited across all 192/192 volumes (bhsd_per_volume_qc.csv): dx=dy in [0.3711, 0.6836] mm (191/192 in [0.3711, 0.5859] mm; 1 vol bhsd_8ea0e9dd0b45873e has 0.6836 mm), dz in [0.5000, 6.5182] mm (median 5.0350 mm; 2 vols outside [2.5, 6.5] mm: bhsd_8cbfc3e3b610d89c dz=0.50 mm, bhsd_f6714e52ac8a58eb dz=6.5182 mm). Orientation: 192/192 LAS (120 with clinical gantry tilt in affine[:3,:3]; 4 with sheared sform vs orthogonal qform header diff between img and gt).",
            "hu_representation": "AUDITED 192/192 VOLUMES (scl_slope=nan, scl_inter=nan, float64; do NOT apply -1024 offset again): 191/192 volumes are standard signed HU (in-FOV p1 in [-1024, -999] HU, p99 in [736, 1614] HU; out-of-FOV padding -1024/-1000 in 118 vols, -2048 in 42 vols, -3024 in 31 vols; 23 vols have isolated 16-bit spike pixels <-3500 or >5000). 1 ANOMALOUS VOLUME (bhsd_59ccfacbf94609f8 / ID_a429fd9a_ID_2eabc43f3d in train) has unsigned 12-bit wrap-around on negative HU (min=0, p1=10, p50=3096=[-1000+4096], max=4095=[-1+4096], while positive lesion HU is [31.9, 45.0]).",
            "patient_identifier": "Filename prefix ID_<patient>_ID_<study>.nii.gz (matches RSNA PatientID); 191 unique patients across 192 volumes; split strictly by patient_group_id (70/15/15, overlap_count=0) so both volumes of ID_2db7ee14 stay in train; splits frozen.",
            "label_type": "Voxel-level 3D multi-class mask (0=background, 1=EDH, 2=IPH, 3=IVH, 4=SAH, 5=SDH) + derived slice-level and exam-level 5-subtype labels",
            "official_split": "dataset.json lists 96 training (95 unique patients) + 96 test (96 unique patients); Block 3 re-splits all 191 patients 70/15/15 by patient_group_id stratified by rarest positive subtype (train=134 vols/133 patients, val=28/28, test=30/30, overlap_count=0)",
            "size_compressed_gb": 1.45,
            "size_uncompressed_gb": 4.20,
            "selected_subset_size_gb": "1.45 GB raw zip preserved + 1.69 GB processed .npz (192 volumes with both 256x256 hu_int16 and 512x512 hu_orig_zyx_int16 for U2; total block3/data usage = 6.082 GB <= 30 GB budget)",
            "role_in_block3": "Primary internal dataset (train/validation/test split strictly by patient_group_id, frozen) for BOTH Presence/5-Subtype Classification AND Binary/5-Subtype Voxel Segmentation (U2: 256x256 patch train on 512x512 grid, 512x512 full-slice inference)",
        },
        {
            "dataset_id": "physionet_ct_ich",
            "name": "PhysioNet Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation",
            "source_url": "https://physionet.org/content/ct-ich/1.3.1/ (local verified dir: block3/data/raw_external/physionet_ct_ich)",
            "release_version": "1.3.1 (2020-03-10)",
            "audit_date": now_date,
            "license": "PhysioNet Restricted Health Data License 1.5.0 (Restricted Access; account + signed DUA required; NO redistribution or sharing between users)",
            "citation": "Dataset: Hssayeni, M. (2020), PhysioNet v1.3.1, DOI: 10.13026/4nae-zg36. Paper: Hssayeni et al., Data 2020, 5(1), 14 (https://www.mdpi.com/2306-5729/5/1/14).",
            "access_status": "verified_local_sha256_156_of_156_passed",
            "total_cases_or_volumes": "75 NIfTI CT+mask pairs (subjects #49-#130 excluding missing #59-#65; Patient_demographics.csv lists all 82 subjects #49-#130)",
            "total_slices": "2814 axial slices (31-58 slices per scan, 512x512 in-plane); 36 ICH positive subjects (EDH:21, IPH:16, SAH:7, IVH:5, SDH:4), 39 negative, 22 fracture; slice subtype counts: EDH:173, IPH:73, SDH:56, IVH:24, SAH:18 (fracture:196)",
            "voxel_spacing_mm": "Verified across all 75 NIfTIs: dx=dy in [0.3145, 0.4883] mm, dz=5.0 mm (100% uniform 5 mm slice thickness)",
            "hu_representation": "VERIFIED_PRECONVERTED_HU_INT16_NIFTI: 75/75 uncompressed .nii int16 with raw NIfTI header slope=1.0, inter=0.0 (nibabel header scl_slope=nan, scl_inter=nan); voxel values already in HU range [-1024.0, 3071.0] (do NOT apply -1024 intercept again). Faces blurred/noised for de-identification.",
            "mask_representation": "VERIFIED_BINARY_MASK_0_254: 75/75 uncompressed .nii (36 float64 positive scans, 39 int16 negative scans), unique values [0, 254] (must binarize >0); 100% shape/affine match with CT scans and 100% slice alignment with hemorrhage_diagnosis_raw_ct.csv.",
            "patient_identifier": "Integer subject ID #49..#130 (1 volume per subject; pseudonymized to pgrp_<sha256> when split; raw_external excluded from git/Drive)",
            "label_type": "Binary ICH voxel mask (0/254 in masks/*.nii) + slice-level 5-subtype & fracture CSV (hemorrhage_diagnosis_raw_ct.csv, 2814 rows)",
            "official_split": "None (75 volumes; candidate for external/secondary segmentation evaluation)",
            "size_compressed_gb": 0.6,
            "size_uncompressed_gb": 2.95,
            "selected_subset_size_gb": "2.95 GB (2.748 GiB) extracted in gitignored raw_external/physionet_ct_ich/ (+ 0.601 GB / 0.560 GiB source zip)",
            "role_in_block3": "Secondary / external segmentation & slice-subtype evaluation (Restricted Access, kept locally only)",
        },
        {
            "dataset_id": "cq500",
            "name": "Qure.ai CQ500 Head CT Dataset",
            "source_url": "https://www.kaggle.com/datasets/crawford/qureai-headct",
            "release_version": "1.0 (2018)",
            "audit_date": now_date,
            "license": "CC BY-NC-SA 4.0 (Non-Commercial ShareAlike)",
            "access_status": "user_has_kaggle_account_not_yet_downloaded",
            "total_cases_or_volumes": "491 examinations (multiple DICOM series per exam: thin + thick reconstructions)",
            "total_slices": "Varies by series (20-50 thick slices or 150-300 thin slices per exam)",
            "voxel_spacing_mm": "Varies by reconstruction series (DICOM header)",
            "hu_representation": "DICOM with RescaleSlope/RescaleIntercept (to be verified per series upon download)",
            "patient_identifier": "CQ500-CT-<id> (1 patient per exam)",
            "label_type": "Exam-level binary labels from 3 radiologists (reads.csv: ICH, EDH, SDH, SAH, IPH, IVH); NO pixel masks",
            "official_split": "External evaluation cohort only (100% external test)",
            "size_compressed_gb": 28.66,
            "size_uncompressed_gb": 62.00,
            "selected_subset_size_gb": "Full download is 28.66 GB (nearly exhausts the 30 GB budget); recommend downloading either a small subset via Kaggle per-patient zips or omitting full download",
            "role_in_block3": "External evaluation (exam-level presence/subtype) if subset downloaded",
        },
        {
            "dataset_id": "instance_2022",
            "name": "INSTANCE 2022 Challenge (Intracranial Hemorrhage Segmentation)",
            "source_url": "https://instance.grand-challenge.org/",
            "release_version": "MICCAI 2022",
            "audit_date": now_date,
            "license": "CC BY-NC-ND (requires Grand Challenge account + signed challenge agreement approved by organizers)",
            "access_status": "restricted_requires_manual_organizer_approval",
            "total_cases_or_volumes": "100 labeled train + 30 val + 70 test volumes",
            "total_slices": "20-40 slices per volume",
            "voxel_spacing_mm": "~0.45 x 0.45 x 5.0 mm",
            "hu_representation": "NIfTI HU (not downloaded)",
            "patient_identifier": "Case index 001..100",
            "label_type": "Binary ICH voxel mask only (no 5-subtype voxel labels)",
            "official_split": "100 train / 30 val / 70 test",
            "size_compressed_gb": 1.20,
            "size_uncompressed_gb": 3.00,
            "selected_subset_size_gb": "0 GB (omitted unless user obtains manual approval; BHSD already provides 192 public 5-subtype volumes)",
            "role_in_block3": "Not used (requires manual email approval; superseded by public BHSD 192-volume dataset)",
        },
        {
            "dataset_id": "m3d_seg_cap",
            "name": "M3D-Seg / M3D-Cap",
            "source_url": "https://github.com/BAAI-DCAI/M3D",
            "release_version": "2024",
            "audit_date": now_date,
            "license": "Apache-2.0 / dataset-specific terms",
            "access_status": "excluded_out_of_scope_and_size",
            "total_cases_or_volumes": "5772 3D volumes (multi-organ)",
            "total_slices": "N/A",
            "voxel_spacing_mm": "Multi-organ varied",
            "hu_representation": "Normalized/varied",
            "patient_identifier": "Multi-source",
            "label_type": "M3D-Cap is text-image (Block 5); M3D-Seg is multi-organ",
            "official_split": "N/A",
            "size_compressed_gb": 150.0,
            "size_uncompressed_gb": 400.0,
            "selected_subset_size_gb": "0 GB (excluded per Plan Section 4 & A5b)",
            "role_in_block3": "Excluded (M3D-Cap belongs to Block 5; M3D-Seg multi-organ exceeds budget and BHSD already covers 3D ICH masks)",
        },
    ]


def main() -> None:
    INV_DIR.mkdir(parents=True, exist_ok=True)
    records = build_inventory()
    json_path = INV_DIR / "datasets_inventory.json"
    csv_path = INV_DIR / "datasets_inventory.csv"

    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    all_keys = list(dict.fromkeys(k for r in records for k in r.keys()))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} dataset inventory records to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
