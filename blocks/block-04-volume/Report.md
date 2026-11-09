# Report — Block 4: Volume

**Phụ trách:** Nguyễn Văn Đức và Nguyễn Hoàng Anh.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Trạng thái

Công thức và benchmark protocol đã xác định; chưa có code hoặc số đo volume accuracy/runtime của dự án.

## Phương pháp

Dùng thể tích physical voxel: voxel_count × sx × sy × sz / 1000 mL. Nguồn chính là mask do Block 3 xuất và spacing/affine cùng space_id. Nếu model resample, mask phải ở cùng hình học với spacing đưa vào phép tính hoặc được map về một reference space đã ghi rõ. Kết quả theo subtype chỉ tính khi voxel-level subtype mask có thật.

## Benchmark cần báo cáo

| Hạng mục | Cách đo | Kết quả dự án |
|---|---|---|
| Sai số thể tích | MAE, RMSE, signed bias (mL), relative error có xử lý reference gần 0 | Chưa đo |
| Agreement | Bland–Altman limits/agreement; correlation chỉ là phụ | Chưa đo |
| Segmentation-to-volume | So sánh volume từ predicted mask với reference mask/manual volume | Chưa đo |
| Runtime | ms/volume, kích thước mask, CPU, throughput | Chưa đo |
| Độ bền geometry | Sai số do spacing, anisotropy, affine mismatch/resampling | Chưa đo |

## Thiết kế benchmark

- Tính predicted volume từ predicted mask; so với reference volume/mask có đơn vị và định nghĩa contour tương thích.
- Báo cáo riêng total ICH, từng subtype nếu reference subtype-specific, theo kích thước lesion, slice thickness và dataset.
- Dùng paired case list, patient-level split và confidence interval. Nếu chỉ có overall binary mask, không báo per-subtype volume accuracy.
- Kiểm tra sự khác nhau giữa volume từ voxel mask và volume do radiologist/manual ellipsoid; ghi rõ reference method vì chúng không luôn là cùng chuẩn.
- Báo cáo segmentation Dice song song với volume MAE: Dice cao không bảo đảm volume nhỏ sai ít, và ngược lại.

## Giới hạn và lỗi cần giữ rõ

- Sai spacing/affine hoặc mask lệch không gian gây sai volume theo hệ số hình học; không thể khắc phục bằng cách làm tròn output.
- Dữ liệu lát dày/anisotropic có thể gây sai số biên lớn; interpolation khi resample tác động lên mask và volume.
- Một mask tổng không tách được subtype. Heatmap classifier không đủ điều kiện đếm voxel.
- Empty mask cần phân biệt với missing artifact. Empty cùng positive finding là quality conflict, không nên âm thầm xuất 0 mL như kết quả đáng tin.
- Report phải ghi mask source, geometry/spacing, phương pháp và status để VLM/consumer biết khi nào không nên khẳng định volume.

