# Report — Block 2: Preprocessing

**Phụ trách:** Nguyễn Hoàng Anh.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Trạng thái

Đặc tả xử lý và benchmark protocol; người phụ trách là Nguyễn Hoàng Anh. Chưa có kết quả chất lượng/runtime của dự án.

## Phương pháp

HU conversion dùng slope/intercept nguồn, giữ intensity HU gốc và tạo brain/subdural/bone window theo profile versioned. Orientation/affine/spacing được giữ; resampling chỉ dùng khi kiến trúc downstream yêu cầu. Nguyễn Hoàng Anh phụ trách phát triển diffusion denoising/correction branch và so sánh với baseline không diffusion. Mục tiêu nhiễu (noise hay motion artifact), cách tạo corruption/training pair, model version và QC phải ghi rõ; corrected volume không ghi đè raw baseline.

## Benchmark cần báo cáo

| Hạng mục | Cách đo | Kết quả dự án |
|---|---|---|
| Tốc độ | Thời gian/volume và ms/slice riêng cho HU, window, resample, diffusion | Chưa đo |
| Geometry | Tỷ lệ affine/spacing bảo toàn; sai số round-trip tọa độ/voxel nếu có biến đổi | Chưa đo |
| Intensity | Tỷ lệ ca HU đã xác minh; min/max/percentile theo nguồn; số ca intensity unknown | Chưa đo |
| Tài nguyên | CPU/GPU, RAM/VRAM peak, throughput theo kích thước volume | Chưa đo |
| Giá trị diffusion | So sánh downstream sensitivity, AUPRC, Dice/lesion recall trên cùng patient split, paired raw vs corrected | Chưa đo |

## Đánh giá chất lượng và giới hạn

- Windowing chỉ là ánh xạ hiển thị/model input; không ghi đè mất volume HU chuẩn.
- Chỉ số PSNR/SSIM hoặc độ mượt ảnh không đủ chứng minh diffusion giúp phát hiện xuất huyết. Có thể kiểm tra thêm với chuyên gia, nhưng điểm quyết định là hiệu năng downstream và khả năng giữ tổn thương nhỏ.
- Generative correction có thể làm mất hoặc tạo chi tiết. Nếu diffusion làm giảm sensitivity, tăng false negative tổn thương nhỏ, hoặc thay đổi mask đáng kể, không bật ở production path nghiên cứu.
- Nguồn PNG/AVI/windowed có thể không còn HU thật. Ghi intensity unknown và tách cohort; không hồi suy HU.
- Resampling làm thay đổi số voxel và hình học lấy mẫu. Lưu affine/spacing output và transform để phép tính thể tích không dùng spacing cũ.
- Không có một WL/WW, clip range hoặc target spacing tối ưu cho mọi subtype/scanner; cần so sánh trên validation rồi khóa profile trước test.

## Thiết kế benchmark

So sánh baseline không diffusion với diffusion trên cùng ca và cùng model weights/threshold, patient-level split; báo cáo confidence interval và phân nhóm SDH mỏng/tổn thương nhỏ. Đo downstream sensitivity/false negative và segmentation quality, không chỉ PSNR/SSIM. Đo thời gian từng bước trên hardware cố định, nêu warm-up, batch và kích thước input. Chỉ chọn corrected branch cho đường mặc định sau khi kết quả validation được review. Chỉ điền số đo sau khi chạy.

