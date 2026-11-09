# Block 4 — Tính thể tích xuất huyết

**Phụ trách:** Nguyễn Văn Đức và Nguyễn Hoàng Anh.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Mục tiêu

Tính thể tích vật lý từ mask voxel và spacing đúng. Đây là phép tính xác định theo mask/geometry, không phải mô hình ngôn ngữ và không tự phân subtype.

## Input

- mask_ich.nii.gz của Block 3; mask_subtype.nii.gz chỉ khi có.
- preprocessing.json hoặc metadata geometry tương ứng với chính mask: shape, affine, voxel_spacing_mm_xyz, space_id.
- Tuỳ chọn: ground-truth mask/reference volume chỉ dành cho benchmark, không lẫn vào inference artifact.

## Output bắt buộc

- volume.json theo contract chung: status, method, mask hash, space_id, spacing, voxel_count, volume_ml, subtype volume nếu được hỗ trợ, quality_flags, warnings/errors.
- Per-subtype volume là null/không phát hành nếu chỉ có mask ICH tổng hoặc label subtype không ánh xạ được voxel.
- Volume tổng là 0 chỉ khi mask rỗng hợp lệ; phân biệt với missing mask, lỗi geometry và mask bị reject.

## Công thức

Với mask nhị phân và spacing voxel theo mm:

V_mL = N_voxel × sx × sy × sz / 1000

Trong đó N_voxel là số voxel foreground và sx, sy, sz lấy từ geometry của cùng không gian mask. Nếu spacing không đồng nhất theo z, dùng voxel volume/thickness theo từng lát hoặc từ affine phù hợp; không lấy một spacing trung bình tùy tiện.

Với mask subtype nhiều lớp, tính count riêng theo mã lớp; tổng các subtype chỉ bằng tổng ICH khi taxonomy/overlap policy được định nghĩa. Nếu mask subtype overlap, ghi quy tắc tránh cộng trùng.

## Luồng xử lý

1. Tải mask và kiểm tra type, shape, affine, space_id, giá trị label và checksum.
2. Xác nhận spacing hữu hạn/dương và tương ứng với affine/mask; kiểm tra resampling transform.
3. Đếm voxel foreground theo loại mask; nhân thể tích voxel vật lý, đổi mm³ sang mL.
4. Phát cờ khi mask rỗng nhưng Block 3 báo positive, mask không rỗng nhưng Block 3 báo negative, geometry mismatch, subtype mapping thiếu hoặc spacing không hợp lệ.
5. Ghi số chưa làm tròn vào artifact; rounding chỉ áp dụng khi render báo cáo.
6. Nếu có reference volume, giữ reference tách biệt và đánh giá sai số ở benchmark, không cho nó chảy vào inference.

## Liên quan

- [Hợp đồng input/output chung](../../docs/INTERFACE_CONTRACT.md)
- [Report Block 4](Report.md)
- [Phân công và bàn giao](../../docs/OWNERSHIP.md)

