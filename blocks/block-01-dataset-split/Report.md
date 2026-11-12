# Report — Block 1: Dataset curation và chia tập

**Phụ trách:** Chưa chỉ định trong danh sách hiện tại.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Trạng thái

Phạm vi: Block 1 chỉ kiểm kê dataset và tạo split manifests theo patient. Chưa có dataset nào được audit/download/chia tập trong repo; chưa có kết quả benchmark.

## Phương pháp

- Tạo inventory theo dataset release, license/DUA, định dạng, số patient/exam/series, label granularity và vị trí annotation.
- Chốt cohort bằng quy tắc inclusion/exclusion có phiên bản.
- Chia theo patient_group_id; mọi series/reconstruction/slice của một patient ở cùng split.
- Tỷ lệ tham chiếu 70/15/15 chỉ dùng khi cohort đủ lớn. Với cohort nhỏ hoặc subtype hiếm, giữ split official hoặc patient-grouped cross-validation; báo số patient từng lớp.
- Lưu seed, script/config version, manifest checksum và exclusions để có thể tái lập.
- Không convert hay sửa ảnh tại block này; dữ liệu nguồn chuyển qua source_uri cho Block 2.

## Benchmark cần báo cáo

| Hạng mục | Cách đo | Kết quả dự án |
|---|---|---|
| Dataset audit | Dataset/release, số patient/exam/series, label scope và mask availability | Chưa đo |
| Split integrity | Số patient overlap giữa train/validation/test; duplicate/hash overlap | Chưa đo |
| Coverage | Patient/case counts và prevalence theo từng subtype, partition, dataset/site | Chưa đo |
| Reproducibility | Kết quả sinh split với cùng seed/config; manifest/checksum version | Chưa đo |
| Thời gian | Thời gian audit/index/chia tập; không tính thời gian đọc/preprocess ảnh của Block 2 | Chưa đo |

## Quality gate và giới hạn

- Split chỉ đạt khi patient-level overlap bằng 0 sau khi áp dụng mapping/duplicate audit.
- Trường hợp thiếu patient identifiers phải được đánh dấu; không thể tuyên bố leakage-free nếu không kiểm tra được identity overlap.
- Stratification không bảo đảm đại diện nếu số positive/subtype ít. Báo cả số patient, không chỉ số slice.
- External test phải độc lập; không dùng kết quả external để tuning.
- Các bộ dữ liệu có định nghĩa label khác nhau không được gộp thành một ground truth duy nhất khi chưa thống nhất mapping.
- Tỷ lệ 70/15/15 là cấu hình gợi ý, không phải kết quả hoặc bắt buộc cho dataset nhỏ.

## Bàn giao

Bàn giao dataset_manifest.json, các split manifests, split_config.json, exclusions và report overlap cho Block 2. Block 3 dùng cùng split version khi training/evaluation. Thay đổi bất cứ cohort membership nào phải tăng dataset/split version và chạy lại kiểm tra overlap.

