# Phân công và bàn giao

Đây là bảng phân công hiện tại của dự án. Block 1 chưa có cá nhân được chỉ định trong danh sách; cần chốt người nhận việc trước khi tạo split chính thức. Mỗi người phụ trách cập nhật Guide/Report của block mình và bàn giao đúng contract tại [INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md).

| Phần | Người phụ trách | Trách nhiệm chính | Output bàn giao |
|---|---|---|---|
| Block 1 — Dataset và chia tập | Chưa chỉ định | Kiểm kê dataset/release/license; xác nhận nhãn, patient ID và nguồn; tạo patient-level train/validation/test manifests; ghi seed, split và checksum; kiểm tra không rò rỉ giữa các tập | dataset_manifest.json và split manifests; không sửa hay commit dữ liệu ảnh |
| Block 2 — Preprocessing + Diffusion | Nguyễn Hoàng Anh | HU conversion, geometry, CT windows, QC, map lát 2D; phát triển diffusion denoising/motion-artifact branch; giữ baseline gốc và chứng minh ảnh hưởng downstream trước khi chọn branch | volume_hu.nii.gz, slices.csv, nhánh diffusion có version nếu khả dụng, preprocessing.json và benchmark |
| Block 3 — Deep Learning | Nguyễn Đình Hữu Minh, Vũ Minh Hoàng | Presence, multi-label subtype, detection/localization khi nhãn hỗ trợ, segmentation; calibration và benchmark theo patient split | findings.json, mask(s), heatmap tùy chọn, weights/config version và Report |
| Block 4 — Tính thể tích | Nguyễn Văn Đức, Nguyễn Hoàng Anh | Tính thể tích từ mask/physical spacing; xử lý QC conflict; so sánh với reference và đo sai số | volume.json, công thức/geometry provenance và benchmark MAE/RMSE/agreement |
| Block 5 — VLM | Nguyễn Hoàng Minh | Nhận evidence có cấu trúc, sinh text có claim/evidence pointers, đo factuality/completeness/repeatability | report.json, prompt/model version và ablation/benchmark |
| Block 6 — Feedback ledger | Nguyễn Tùng Sơn | Chuẩn bị thiết kế lưu feedback đã duyệt, model/config version và lineage; chưa xây service/blockchain trong placeholder hiện tại | Phạm vi/schema đề xuất ở giai đoạn sau; không nằm trong runtime của Block 1–5 |
| CI/CD — phụ trách, cố vấn và giám sát | Nguyễn Quang Vũ Hoàng | Điều phối CI/CD toàn dự án: quy ước branch/PR, lint/docs checks, pipeline status và phát hành artifact có version; phối hợp owner từng block để giữ workflow dùng chung | Kế hoạch CI/CD và trạng thái tích hợp; không sở hữu model/domain output của từng block |

## Bàn giao giữa các owner

- Block 1 khóa danh sách case và patient-level splits trước khi Block 2/3 huấn luyện. Mọi case mới hoặc thay đổi split phải tạo manifest/version mới.
- Block 2 ghi rõ preprocessing profile và nhánh ảnh (raw HU hoặc diffusion) mà Block 3 dùng. Cùng một case không được trộn output giữa hai nhánh mà thiếu provenance.
- Block 3 bàn giao mask cùng geometry/space_id cho Block 4; heatmap không thay mask.
- Block 4 bàn giao thể tích và quality flags cho Block 5; Block 5 không tính lại volume.
- Block 6 là placeholder, không chặn các block 1–5.
- Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát CI/CD xuyên suốt; từng owner chịu trách nhiệm tính đúng của artifact và benchmark do block đó tạo.
