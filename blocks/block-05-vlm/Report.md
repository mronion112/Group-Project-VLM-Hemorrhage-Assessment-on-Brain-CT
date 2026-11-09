# Report — Block 5: VLM

**Phụ trách:** Nguyễn Hoàng Minh.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Trạng thái

Thiết kế evidence-grounded reporting và benchmark; chưa chọn/triển khai VLM, chưa sinh report hoặc đo latency của dự án.

## Phương pháp

VLM nhận bốn trường bằng chứng (presence, subtype, mask-derived location, volume) cùng cờ QC/version. Tạo text và danh sách claims chuẩn hóa có evidence pointer. Output có thể là tiếng Việt theo yêu cầu sản phẩm, nhưng benchmark phải cố định ngôn ngữ/prompt version. Nếu chỉ có một binary ICH mask, không tạo khẳng định subtype-specific region/volume.

## So sánh bắt buộc (ablation)

| Biến thể | Input VLM | Mục đích |
|---|---|---|
| A. Direct VLM (nếu khả thi) | Ảnh CT và prompt | Baseline VLM đọc ảnh; phải nêu cách chọn slice/volume |
| B. Classification-guided | Presence + subtype labels/scores | Đo tác dụng của bằng chứng phân loại |
| C. Full structured evidence | B + localization/mask + volume | Đo giá trị tăng thêm của mask và volume |
| D. Oracle evidence | Ground-truth findings | Upper bound phân tích; không phải deployment setting |

Cùng bộ case, split patient, ngôn ngữ, template và decoding policy. Báo cáo chênh lệch C–B và confidence interval; tách hiệu quả ngôn ngữ khỏi độ đúng của upstream model.

## Benchmark cần báo cáo

| Hạng mục | Cách chấm | Kết quả dự án |
|---|---|---|
| Presence/subtype | Độ đúng theo field; sensitivity/specificity/F1 nếu output thành nhãn | Chưa đo |
| Location | Exact/normalized region + laterality accuracy; đối chiếu mask nếu output định lượng vùng | Chưa đo |
| Volume | MAE/RMSE mL và tỷ lệ chép chính xác giá trị evidence | Chưa đo |
| Factuality | Tỷ lệ claims được hỗ trợ; unsupported-claim rate; contradiction rate | Chưa đo |
| Completeness | Tỷ lệ trường bắt buộc được diễn đạt đúng hoặc explicitly unknown | Chưa đo |
| Repeatability | Agreement giữa lần chạy lặp; tỷ lệ đổi subtype/volume/claim | Chưa đo |
| Latency/cost | s/report, tokens, throughput, CPU/GPU/VRAM và cấu hình model/API | Chưa đo |

## Rubric đánh giá report

Chấm mỗi ca theo từng trường: 1 = chính xác và có evidence, 0 = sai/mâu thuẫn, NA = không có evidence và đã abstain đúng. Tính riêng accuracy theo trường; thêm:

- Unsupported-claim rate = claims không được input hỗ trợ / tổng claims.
- Contradiction rate = claims mâu thuẫn với findings/volume / tổng claims.
- Evidence coverage = claims có pointer hợp lệ / tổng claims.
- Appropriate-abstention rate = trường thiếu/uncertain được diễn đạt đúng là thiếu/uncertain / tổng trường thiếu/uncertain.

Chọn hai reviewer độc lập cho subset, đo agreement và phân xử bất đồng theo rubric trước khi tổng hợp. BLEU/ROUGE/BERTScore hoặc text similarity có thể bổ sung nhưng không thay đánh giá factuality.

## Giới hạn và tiêu chí chất lượng

- Lỗi upstream có thể được diễn đạt trôi chảy và vẫn sai; report score không thay detection/segmentation score.
- VLM có thể thêm chi tiết y khoa phổ biến nhưng không có bằng chứng. Schema validator và claim audit là một phần của output boundary.
- Không đưa confidence score thành xác suất lâm sàng nếu chưa calibration.
- Không dùng “đầy đủ thông tin từ block trước” nếu mask, volume hoặc QC bị thiếu; report phải ghi missing evidence.
- Text similarity với report tham chiếu không chứng minh đúng subtype/location/volume.
- Ngưỡng chấp nhận (ví dụ tối đa unsupported claim hoặc yêu cầu exact field accuracy) phải được định nghĩa trước khi mở test set; hiện chưa có kết quả hoặc ngưỡng đã đo.

