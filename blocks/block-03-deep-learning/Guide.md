# Block 3 — Deep Learning: presence, subtype, localization, mask

**Phụ trách:** Nguyễn Đình Hữu Minh và Vũ Minh Hoàng.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Mục tiêu

Từ volume đã QC, tạo kết quả ảnh học được có cấu trúc: có/không/không chắc ICH, điểm cho 5 subtype, mask và vị trí khi mô hình/nhãn cho phép. Block này không tạo thể tích mL và không viết report bằng ngôn ngữ tự nhiên.

## Input

- volume_hu.nii.gz, preprocessing.json và các view/window được khai báo từ Block 2.
- Model config/version, calibration/decision thresholds và label map.
- Nhãn huấn luyện tương thích ở mức ca/lát/mask. Split phải theo patient/examination trước khi trích lát.

## Output bắt buộc

- findings.json theo contract chung; gồm presence 3 trạng thái, score/threshold, score/prediction của từng subtype, model/config version, input space_id, warnings/errors.
- mask_ich.nii.gz khi model segmentation được bật; cùng không gian với input đã chỉ định.
- mask_subtype.nii.gz chỉ nếu output thực sự phân lớp voxel theo EDH/SDH/SAH/IPH/IVH và dataset/model được huấn luyện cho task đó.
- Heatmap/bounding-box/candidate-region là output phụ, có confidence, tọa độ và thuật ngữ rõ. Không coi heatmap là mask hoặc ground truth.

## Pipeline đề xuất

1. **Audit nhãn:** tách classification (case/slice), detection (bbox), segmentation (pixel/voxel) và text supervision. Không chuyển nhãn cấp ca thành nhãn slice/mask khi chưa có quy tắc và kiểm định.
2. **Baseline presence/subtype:** 2D model + aggregation theo ca nếu dữ liệu chỉ có nhãn lát; so với 3D model khi đủ volume/compute. Subtype là multi-label vì nhiều loại có thể đồng thời xuất hiện.
3. **Detection/localization:** huấn luyện riêng khi có bounding box hoặc quy tắc localization. RSNA ICH gốc là nhãn ảnh, không có bbox/mask chung; heatmap từ classifier chỉ là saliency/candidate, chưa chứng minh định vị lesion.
4. **Segmentation:** dùng kiến trúc 3D như nnU-Net làm baseline hoặc 2D/2.5D khi spacing anisotropic và số ca hạn chế. Đối chiếu mask tổng với mask subtype theo đúng annotation.
5. **Calibration/decision:** chọn threshold trên validation; báo cáo score raw và calibrated, cách gộp slice thành case, và quy tắc uncertain.
6. **Xuất findings + mask** với space_id/affine, mapping subtype, hash, model version; không postprocess âm thầm.

### ROI/heatmap acceleration — tùy chọn

Có thể thêm bước sàng lọc hoặc region proposal trước mô hình nặng để giảm số crop/slice cần xử lý. Chỉ bật sau khi đo thời gian toàn pipeline và chứng minh không làm giảm sensitivity, đặc biệt với SDH mỏng và tổn thương nhỏ. Ca negative/uncertain hoặc QC kém cần có đường full-volume/fallback. Heatmap subtype có thể được overlay để reviewer thấy candidate; không dùng heatmap để tính diện tích/volume hoặc tuyên bố đó là segmentation mask.

## Dataset và chia tập

- RSNA: nhãn image-level cho presence/subtype, hữu ích cho classifier; không dùng như bbox/mask.
- CQ500: nhãn lâm sàng cấp ca, phù hợp external evaluation nếu định nghĩa label tương thích; không tạo mask giả.
- PhysioNet CT-ICH: volume/mask ICH cùng label subtype theo lát; cohort nhỏ, traumatic, ~5 mm; xác minh bản 75 volume và dạng mask.
- INSTANCE: binary ICH segmentation benchmark; không giả định 5-class voxel labels.
- BHSD: candidate 5-subtype segmentation, phải audit release/splits.
- M3D-Seg: audit anatomy/modality/category thực tế. M3D-Cap/VQA hỗ trợ multimodal pretraining khi subset/task khớp, không tự thay gold label ICH.
- Chia patient/exam trước khi tạo slice hoặc augmented sample để tránh leakage.

## Contract

Block 4 đọc mask và geometry để tính mL. Block 5 đọc findings và mask-derived localization. Nếu model chỉ trả overall ICH mask, location/subtype-specific volume chỉ được nêu ở mức mà bằng chứng hỗ trợ. Missing mask phải là missing/uncertain, không thay bằng heatmap.

## Liên quan

- [Hợp đồng input/output chung](../../docs/INTERFACE_CONTRACT.md)
- [Report Block 3](Report.md)
- [Phân công và bàn giao](../../docs/OWNERSHIP.md)

