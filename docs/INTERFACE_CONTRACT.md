# Hợp đồng input/output liên block

Tài liệu này là nguồn chuẩn cho các artifact trao đổi giữa các block. Tên field có thể được map vào code theo ngôn ngữ triển khai, nhưng ý nghĩa, đơn vị, không gian tọa độ và trạng thái không được thay đổi ngầm. Thay đổi contract phải tăng schema_version và cập nhật Guide của block liên quan.

## Quy ước chung

- Mỗi ca có một case_id và patient_group_id giả danh ổn định trong toàn pipeline; không đưa tên, ngày sinh, MRN hoặc thông tin định danh trực tiếp vào artifact.
- Artifact theo-ca có schema_version, case_id, status (ok, partial, failed), created_at_utc, source_artifact_ids; artifact cấp dataset có dataset_id/release và danh sách record tương ứng. Block có mô hình ghi thêm model_name, model_version, weights_hash, config_hash.
- status=partial phải nêu warnings[]; status=failed phải nêu errors[] với code ổn định và thông điệp đọc được. Không điền giá trị đoán để giả lập input bị thiếu.
- space_id định danh một không gian ảnh cụ thể. NIfTI affine, shape, orientation và spacing phải được giữ kèm hoặc có thể truy xuất không mơ hồ.
- Đơn vị spacing là mm; thể tích là mL. Chỉ số voxel và slice dùng index 0-based. Subtype dùng đúng mã EDH, SDH, SAH, IPH, IVH.
- Mask là NIfTI cùng shape/affine với volume được chỉ ra. Nếu đổi/resample không gian, cần artifact ghi phép biến đổi và phải đưa mask về đúng không gian tham chiếu trước khi tính thể tích.
- CT 3D là dữ liệu chuẩn. Lát 2D dẫn xuất phải giữ slice_index, source_volume_id, orientation và window profile.

## Artifact và luồng sở hữu

| Artifact | Sinh bởi | Đọc bởi | Bắt buộc |
|---|---|---|---|
| dataset_manifest.json | Block 1 | Block 2, 3 | Dataset/release, license/DUA status, cohort, label scope, source references và provenance |
| splits/{train,validation,test}.csv | Block 1 | Block 2, 3 | case_id, patient_group_id, dataset/release, source_uri, split và labels_available; patient không được overlap |
| split_config.json | Block 1 | Block 2, 3 | seed, grouping/stratification rules, target ratios, exclusions, script/version và manifest checksum |
| volume_hu.nii.gz | Block 2 | Block 3 | Volume 3D có intensity HU đã xác minh, hoặc status thể hiện rõ hạn chế nguồn |
| volume_diffusion.nii.gz | Block 2 | Block 3 | Chỉ có khi diffusion branch được tạo; phải giữ raw baseline riêng và ghi không gian/transform |
| slices.csv | Block 2 | Block 3 | slice_index, vị trí vật lý, source volume, space_id và window profile; ảnh/tensor 2D dẫn xuất tham chiếu dòng tương ứng |
| preprocessing.json | Block 2 | Block 3, 5 | biến đổi, cửa sổ, spacing, QC và space_id |
| findings.json | Block 3 | Block 4, 5 | presence, điểm subtype, ngưỡng, model version, tham chiếu mask |
| mask_ich.nii.gz | Block 3 | Block 4, 5 | Mask nhị phân ICH, cùng không gian với volume đầu vào DL |
| mask_subtype.nii.gz | Block 3 | Block 4, 5 | Chỉ có nếu mô hình/ground truth thực sự cung cấp mask theo subtype |
| volume.json | Block 4 | Block 5 | voxel count, mL, mask/spacing nguồn và quality flags |
| report.json | Block 5 | Ứng dụng/đánh giá | Text báo cáo, claims chuẩn hóa và bằng chứng nguồn của từng claim |

## Nội dung tối thiểu của các JSON

Các mẫu dưới đây rút gọn để minh họa field theo block. Artifact JSON thực tế vẫn phải có created_at_utc và source_artifact_ids theo quy ước chung, ngoài các field block-specific.

### dataset_manifest.json — Block 1

~~~json
{
  "schema_version": "1.0",
  "dataset_id": "dataset-id",
  "dataset_release": "release-version",
  "status": "ok",
  "license_status": "verified-for-project-use",
  "cohort_config": "cohort-v1",
  "split_config": {"seed": 2026, "group_key": "patient_group_id", "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15}},
  "split_manifests": {"train": "splits/train.csv", "validation": "splits/validation.csv", "test": "splits/test.csv"},
  "record_count": 100,
  "patient_overlap_check": {"status": "passed", "overlap_count": 0},
  "warnings": [],
  "errors": []
}
~~~

Tên dataset, seed, tỷ lệ và số record trong ví dụ chỉ minh họa cấu trúc. Split manifest giữ source_uri đến DICOM/NIfTI gốc; Block 1 không tạo volume ảnh.

### preprocessing.json — Block 2

~~~json
{
  "schema_version": "1.0",
  "case_id": "case-pseudonymous-id",
  "status": "ok",
  "source_uri": "controlled-data://dataset-id/case-pseudonymous-id/series-id",
  "split_manifest_id": "train-split-v1",
  "partition": "train",
  "output_artifact": "volume_hu.nii.gz",
  "space_id": "case-pseudonymous-id:block2:v1",
  "hu_conversion": {"status": "applied", "source": "DICOM rescale slope/intercept"},
  "resampling": {"applied": false, "target_spacing_mm_xyz": null},
  "windows": [{"name": "brain", "wl": 40, "ww": 80}],
  "diffusion_correction": {"enabled": true, "artifact": "volume_diffusion.nii.gz", "model_version": "diffusion-version", "branch_status": "candidate"},
  "qc": {"geometry_valid": true, "slice_order_valid": true, "intensity_status": "verified"},
  "warnings": [],
  "errors": []
}
~~~

Thông số WL/WW chỉ là ví dụ profile; cấu hình thật phải có version. Không ghi chung “đã đổi HU” nếu nguồn intensity không xác minh được.

### findings.json — Block 3

~~~json
{
  "schema_version": "1.0",
  "case_id": "case-pseudonymous-id",
  "status": "ok",
  "model_name": "model-id",
  "model_version": "version-id",
  "input_space_id": "case-pseudonymous-id:block2:v1",
  "presence": {"label": "positive", "score": 0.97, "threshold": 0.5},
  "subtypes": {
    "EDH": {"score": 0.02, "present": false},
    "SDH": {"score": 0.73, "present": true},
    "SAH": {"score": 0.08, "present": false},
    "IPH": {"score": 0.04, "present": false},
    "IVH": {"score": 0.01, "present": false}
  },
  "masks": {
    "ich": {"artifact": "mask_ich.nii.gz", "space_id": "case-pseudonymous-id:block2:v1"},
    "subtype": null
  },
  "localization": {"status": "mask-derived", "side": "right", "regions": [], "coordinate_system": "RAS+"},
  "heatmaps": [],
  "warnings": [],
  "errors": []
}
~~~

Điểm và nhãn trong ví dụ là dữ liệu minh họa, không phải kết quả mô hình. Cho phép nhiều subtype cùng present. Nếu phân loại chưa hiệu chỉnh, ghi rõ score không phải xác suất đã calibration.

### volume.json — Block 4

~~~json
{
  "schema_version": "1.0",
  "case_id": "case-pseudonymous-id",
  "status": "ok",
  "method": "voxel_count_times_physical_voxel_volume",
  "source_mask": "mask_ich.nii.gz",
  "space_id": "case-pseudonymous-id:block2:v1",
  "voxel_spacing_mm_xyz": [0.45, 0.45, 5.0],
  "voxel_count": 1200,
  "volume_ml": 6.075,
  "by_subtype_ml": null,
  "quality_flags": [],
  "warnings": [],
  "errors": []
}
~~~

Các giá trị số ví dụ không phải benchmark. by_subtype_ml phải là null nếu không có mask subtype đáng tin cậy; không suy ra từ heatmap.

### report.json — Block 5

~~~json
{
  "schema_version": "1.0",
  "case_id": "case-pseudonymous-id",
  "status": "ok",
  "language": "vi",
  "report_text": "Bản giải thích được sinh từ các bằng chứng có cấu trúc.",
  "claims": [
    {"field": "presence", "value": "positive", "evidence": ["findings.json#/presence"]},
    {"field": "subtype", "value": ["SDH"], "evidence": ["findings.json#/subtypes/SDH"]},
    {"field": "volume_ml", "value": 6.075, "evidence": ["volume.json#/volume_ml"]}
  ],
  "unsupported_claims": [],
  "missing_evidence": [],
  "model_name": "vlm-id",
  "model_version": "version-id"
}
~~~

Report text phải khớp claims; nếu evidence thiếu, biểu đạt unknown/not provided thay vì tự suy đoán. Không đưa claim subtype-volume nếu mask chỉ là mask ICH tổng.

## Quy tắc tương thích

1. Block chỉ đọc input đã liệt kê và không phụ thuộc tên file nội bộ khác của block trước.
2. Mỗi output có artifact ID/hash để phát hiện trộn nhầm ca hoặc dùng output cũ.
3. case_id, space_id, shape, affine, spacing và model version được truyền xuyên suốt.
4. Nếu status không phải ok, block sau vẫn có thể chạy khi trường bắt buộc hợp lệ; phải giữ nguyên warning/error và hạ trạng thái output tương ứng.
5. Mỗi block có thể thay framework/model miễn là giữ contract và báo model/config version.

