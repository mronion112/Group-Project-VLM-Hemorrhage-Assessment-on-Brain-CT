# Report — Block 3: Deep Learning

**Phụ trách:** Nguyễn Đình Hữu Minh và Vũ Minh Hoàng.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Trạng thái

Kiến trúc và benchmark protocol được đặc tả; chưa có mô hình, weights hoặc kết quả benchmark của repo.

## Phương pháp và baseline

Dùng baseline riêng cho từng tác vụ và chỉ gộp khi contract output tương thích:

1. Presence/subtype: multi-label classifier; thử 2D slice aggregation nếu nhãn chủ yếu là 2D, đối chiếu 3D khi volume-level data đủ.
2. Localization/detection: chỉ báo detection metric khi có bbox/region annotation. Saliency heatmap là giải thích/candidate, không phải detection output được xác nhận.
3. Segmentation: 3D nnU-Net là baseline khả thi; so với 2D/2.5D nếu volume anisotropic hoặc số volume hạn chế. Tách mask ICH tổng khỏi mask 5-class.
4. Calibration và chọn threshold trên validation. Hỗ trợ label uncertain khi score vùng không đáng tin; ghi rule trong config.
5. M3D-VLM hoặc VLM segmentation không thay benchmark model chuyên biệt nếu chưa chứng minh tương đương trên test độc lập.

## Benchmark cần báo cáo

| Nhiệm vụ | Metric tối thiểu | Phân tầng bắt buộc | Kết quả dự án |
|---|---|---|---|
| Presence | Sensitivity, specificity, AUROC, AUPRC | Mức ca; external site/dataset | Chưa đo |
| 5 subtype | Sensitivity, specificity, macro-F1, confusion matrix | Từng subtype, đồng mắc, mức ca/lát tách biệt | Chưa đo |
| Detection | lesion-wise sensitivity, precision, FROC/mAP nếu có bbox | Theo subtype/kích thước | Chưa đo |
| Segmentation | Dice, HD95, lesion-wise sensitivity | Mask tổng và subtype; tổn thương nhỏ; dataset | Chưa đo |
| Calibration | Brier/ECE hoặc reliability plot; sensitivity theo threshold | Từng subtype, threshold đã khóa | Chưa đo |
| Latency | s/volume, s/slice, batch, throughput, peak VRAM | Mỗi architecture và resolution | Chưa đo |

## Mốc tham khảo từ report/tài liệu (không phải kết quả của dự án)

- Trong thí nghiệm được report tổng hợp trên BHSD, Swin-v2-B có macro-F1 subtype khoảng 0,86, trong khi Gemini 2.0 Flash zero-shot khoảng 0,31. Thiết lập học/đánh giá khác nhau và có cảnh báo split theo slice, nên không so trực tiếp như một benchmark công bằng của repo.
- Nghiên cứu segmentation IPH tự phát được report trích dẫn có Dice khoảng 0,894 ± 0,264; không suy rộng kết quả này sang 5 subtype.
- INSTANCE challenge báo cáo Dice/DSC cùng HD/RVD/NSD; điểm giữa các challenge, split và protocol không thay thế test độc lập của dự án.

## Giới hạn và tiêu chí chất lượng

- RSNA labels không phải spatial annotation; không được báo Dice hoặc detection AP nếu chưa có mask/bbox ground truth phù hợp.
- Bất cân bằng subtype, nhãn yếu/cấp ca, lát liền kề và leakage giữa ca có thể làm điểm slice cao giả tạo. Tất cả split chính theo patient.
- Slice-level score không tự động thành case-level diagnosis. Phải định nghĩa aggregation rồi đánh giá lại ở mức ca.
- Heatmap có thể tô nền/artefact, không cung cấp diện tích hay volume. Block 4 chỉ dùng mask phân đoạn.
- ROI prefilter có thể giảm compute nhưng có thể loại nhầm lát/lesion. Đánh giá end-to-end gồm tỷ lệ lesion lọt qua prefilter, false negative nhỏ, sensitivity theo subtype và thời gian tiết kiệm thực, so với full-volume baseline.
- Lựa chọn threshold cần ưu tiên sensitivity/false-negative tradeoff đã thống nhất trên validation; không chọn theo test.
- Mỗi report benchmark ghi cohort, exclusions, prevalence, split, seed, preprocessing version, model version, threshold, hardware, confidence interval và lỗi theo subtype. Chưa có ngưỡng chấp nhận định lượng của dự án; cần đặt trước khi mở test set.

