# Block 1 — Dataset curation và chia tập

**Phụ trách:** Chưa chỉ định trong danh sách hiện tại.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Mục tiêu

Block 1 chỉ kiểm kê/chọn dataset, quản lý metadata và chia tập chuẩn theo patient. Block này không convert, preprocess, resample, window hay chỉnh sửa ảnh. Dữ liệu gốc giữ tại kho được cấp quyền; repo chỉ chứa manifest và công cụ cấu hình nếu cần.

## Input

- Dataset/release đã được cấp quyền truy cập: ví dụ RSNA, CQ500, PhysioNet CT-ICH, INSTANCE, BHSD hoặc M3D sau khi audit đúng task.
- Metadata của từng ca/series: patient/study/series ID (đã giả danh), modality, label scope, subtype labels, mask availability, source path/URI, release/version, license/DUA.
- Quy tắc chọn cohort và split; seed; danh sách dataset external test nếu có.

## Output bắt buộc

- dataset_manifest.json: version/release, nguồn, license/DUA status, cohort selection, số ca/series, label schema, source reference, checksum và ghi chú metadata còn thiếu.
- splits/train.csv, splits/validation.csv, splits/test.csv (hoặc cùng nội dung ở định dạng JSONL). Mỗi dòng tối thiểu có case_id, patient_group_id giả danh, dataset_id/release, source_uri, series_id nếu cần, split, labels_available và label_scope.
- splits/external.csv riêng khi dùng dataset/site độc lập để external evaluation. Không trộn external test vào train hoặc dùng để chọn threshold.
- split_config.json: seed, grouping key, stratification labels, tỷ lệ mục tiêu, exclusions, phiên bản script và checksum của manifest.
- Artifact không chứa ảnh y tế và không commit credential/local path tuyệt đối. source_uri có thể là relative path hoặc URI trong kho dữ liệu được kiểm soát.

## Quy trình chia tập chuẩn

1. Audit dataset/release và xác nhận quyền sử dụng, định nghĩa ca, patient ID, label availability, modality, series/reconstruction và khả năng định danh patient xuyên các nguồn.
2. Chốt cohort trước khi chia: quy tắc inclusion/exclusion và label task; ghi lý do mỗi ca bị loại. Không suy mask từ nhãn classification.
3. Tạo patient_group_id giả danh ổn định. Gộp mọi exam, series, reconstruction và lát của cùng patient vào cùng một partition.
4. Mặc định đề xuất 70/15/15 train/validation/test theo patient nếu cohort đủ lớn. Stratify ở mức patient theo presence/subtype khi nhãn cho phép; lưu seed và số ca từng lớp. Với cohort nhỏ/mất cân bằng, giữ split chính thức của dataset hoặc dùng patient-grouped cross-validation thay vì ép tỷ lệ thiếu lớp.
5. Giữ dataset external (ví dụ CQ500/PhysioNet khi không dùng training) làm đánh giá độc lập. Nếu một patient xuất hiện ở nhiều nguồn, group chung hoặc loại khỏi external comparison theo protocol đã ghi.
6. Kiểm tra giao nhau patient_group_id giữa splits bằng script/report; kiểm tra duplicate checksum và các alias ID đã biết. Chỉ khóa split sau khi không còn overlap.
7. Xuất manifest/split lists có checksum/version để Block 2 lấy đúng source_uri; thay cohort hoặc split phải tạo version mới.

## Hợp đồng với Block 2

Block 2 đọc source_uri từ split manifest và mở DICOM/NIfTI nguồn theo format được khai báo. Block 1 không xuất volume.nii.gz chuẩn hóa. Block 2 sở hữu loading/geometry/HU conversion/preprocessing và ghi provenance. Không phụ thuộc đường dẫn máy cá nhân; sử dụng config/mount theo môi trường.

## Chất lượng và lỗi

- Không chia ngẫu nhiên theo slice. Mọi lát/series/exam cùng patient ở một split.
- Nếu thiếu patient ID hoặc không thể xác định trùng bệnh nhân, đánh dấu cohort risk và không tuyên bố split leakage-proof.
- Nếu label granularity không rõ (case, series, slice, voxel), ghi unknown và không dùng cho task cần label chi tiết hơn.
- Không dùng test để chọn model, threshold, preprocessing hoặc prompt.
- Không đưa dataset có DUA/giới hạn chia sẻ vào Git LFS/GitHub công khai.

## Phân công và liên quan

Người phụ trách Block 1 chưa được chỉ định. CI/CD do Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát. Liên hệ owner các Block 2–5 để khóa format split trước khi chạy pipeline.

- [Contract input/output](../../docs/INTERFACE_CONTRACT.md)
- [Phân công tổng thể](../../docs/OWNERSHIP.md)
- [Report Block 1](Report.md)

