# Block 2 — HU, geometry, cửa sổ và QC

**Phụ trách:** Nguyễn Hoàng Anh.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Mục tiêu

Tạo input ổn định cho mô hình ảnh trong khi giữ quan hệ vật lý giữa voxel, slice và mask. Ghi lại mọi phép biến đổi đủ để Block 3/4 và reviewer truy vết.

## Input

- dataset_manifest.json và split manifest của Block 1, cùng source_uri trỏ đến series DICOM/NIfTI gốc. Chọn partition bằng config run, không tự lấy lẫn train/validation/test.
- Profile preprocessing có version: HU conversion policy, target spacing nếu resample, các CT window/normalization và cấu hình diffusion branch.
- Model adapter có thể yêu cầu tensor/channel cụ thể; yêu cầu đó phải được khai báo, không làm thay contract canonical volume.

## Output bắt buộc

- volume_hu.nii.gz: canonical volume có HU nếu nguồn cho phép; nếu input NIfTI đã là HU thì không áp slope/intercept lần hai. Nếu intensity không xác định, dừng hoặc ghi status theo policy, không gắn nhãn HU giả.
- volume_diffusion.nii.gz: output của diffusion denoising/correction khi branch được tạo. Luôn giữ volume_hu.nii.gz làm baseline không diffusion để so sánh; ghi model/config version, loại nhiễu/artefact mục tiêu, transform và QC.
- slices.csv: danh mục axial slices có thứ tự, slice_index 0-based, vị trí vật lý, source volume/space_id và window profile. Ảnh 2D hoặc tensor xuất theo model adapter là dẫn xuất, không thay canonical 3D.
- preprocessing.json: HU conversion, clipping/windowing, resampling, orientation, affine/shape/spacing, profile version, QC, warnings/errors, input/output hash và space_id.
- Bản brain/subdural/bone window hoặc tensor nhiều kênh có thể là dẫn xuất phục vụ model, nhưng phải tham chiếu đúng volume và ghi WL/WW/normalization.

## Luồng xử lý

1. Xác nhận intensity status từ manifest. Với DICOM áp stored_value × RescaleSlope + RescaleIntercept. Không áp slope/intercept lần hai cho dữ liệu đã chuyển sang HU.
2. Giữ HU đầy đủ trong canonical output; window/clip theo model profile. Không mặc định clip toàn bộ cohort vào khoảng [-1000, 4000].
3. Kiểm tra shape, affine, orientation, spacing, slice order, giá trị NaN/Inf và phân bố intensity.
4. Resample chỉ khi model yêu cầu; ghi interpolation, target spacing và affine mới. Mask/heatmap từ Block 3 phải cùng space_id hoặc được ánh xạ ngược chính xác.
5. Sinh các view brain/subdural/bone nếu cần; nhiều window là channel/view chứ không phải nhiều bệnh nhân/ca.
6. Tạo slices.csv theo hướng trục đã khai báo; giữ mapping đến index và tọa độ vật lý trong volume. Nếu export PNG/tensor, lưu profile WL/WW/normalization và không coi ảnh windowed là HU.
7. Phát triển diffusion denoising/correction như một branch có version. Phân biệt giảm nhiễu tổng quát với sửa motion artifact; ghi rõ loại corruption mục tiêu, cách tạo paired/synthetic data nếu dùng và giới hạn domain. Giữ nguyên baseline không diffusion.
8. So sánh raw/preprocessed baseline với corrected branch trên cùng patient split; kiểm tra khả năng giữ tổn thương nhỏ và hiệu năng downstream Block 3 trước khi chọn branch mặc định. Cho đến khi kết quả được chấp thuận, diffusion output là candidate chứ không ghi đè ảnh gốc.
9. Trả QC và trạng thái rõ; không “sửa” ca lỗi mà không ghi biến đổi.

## Interface

Block 3 nhận volume_hu.nii.gz hoặc diffusion branch được chọn tường minh trong run config, cùng preprocessing.json và các view đã khai báo. Block 5 nhận metadata QC/profile dạng gọn để biết evidence đầu vào có bị giới hạn. Block 2 không kết luận hiện diện subtype, mask hay thể tích.

## Liên quan

- [Hợp đồng input/output chung](../../docs/INTERFACE_CONTRACT.md)
- [Report Block 2](Report.md)
- [Phân công và bàn giao](../../docs/OWNERSHIP.md)

