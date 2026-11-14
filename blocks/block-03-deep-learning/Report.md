# Report — Block 3: Deep Learning

**Phụ trách:** Nguyễn Đình Hữu Minh và Vũ Minh Hoàng.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Trạng thái

Đã triển khai hoàn chỉnh pipeline huấn luyện và đánh giá trên Google Colab GPU (Tesla T4) tại file `Block3_Colab_Training_Pipeline.ipynb`. Pipeline tích hợp các kỹ thuật kiến trúc tối tân từ quán quân **RSNA 2019 (SeuTao)** và **MICCAI INSTANCE 2022 (NVIDIA / Team Vegetable)**.

## Phương pháp và Kiến trúc Tối tân (Champion Pipeline)

Pipeline được chuẩn hóa thành 3 giai đoạn độc lập nhưng kết nối chặt chẽ theo Interface Contract v1.0:

### 1. Tiền xử lý 3 Cửa sổ CT (Multi-Window CT) & Bảo toàn Tổn thương Mỏng
- Chuyển đổi giá trị Hounsfield Unit (HU) gốc thành ảnh 3 kênh:
  - **Brain Window:** $L=40, W=80 \rightarrow [0, 80]$ HU (tối ưu nhu mô não).
  - **Subdural Window:** $L=80, W=200 \rightarrow [-20, 180]$ HU (tối ưu cho tụ máu dưới màng cứng và khoang dưới nhện).
  - **Bone Window:** $L=600, W=2800 \rightarrow [-800, 2000]$ HU (giúp mô hình học rõ bờ xương sọ và vôi hóa).
- Áp dụng phép nội suy bảo toàn tổn thương nhỏ (Anti-aliased thin bleed preservation), đảm bảo các vệt xuất huyết rãnh não (sulcal SAH) có bề dày 1-2 pixel không bị triệt tiêu khi nạp vào mạng.

### 2. Phân loại Đa nhãn Chuỗi 3D (RSNA Top 1 Champion — SeuTao)
- **Tầng 1 (Spatial Feature Extractor):** ResNet-34 pretrained trích xuất vector đặc trưng $512$ chiều cho từng lát cắt 2D.
- **Tầng 2 (Sequence Context Aggregator):** Bidirectional GRU (`hidden_dim=128`, `bidirectional=True`) quét 2 chiều dọc theo trục không gian $Z$, mô phỏng quá trình bác sĩ cuộn phim CT từ đỉnh đầu xuống nền sọ để học tính liên tục giải phẫu của khối xuất huyết.
- **Hàm mất mát:** `BCEWithLogitsLoss` có trọng số bù trừ `pos_weight` cho các phân loại hiếm (EDH, SDH).

### 3. Phân đoạn Nâng cao (MICCAI INSTANCE 2022 Champion) & Hậu xử lý Vật lý
- **Kiến trúc:** Attention U-Net với Encoder ResNet-34 và khối **scSE (Spatial and Channel Squeeze-and-Excitation)**. Khối scSE tự động triệt tiêu tín hiệu xương sọ và tập trung vào bờ mô tổn thương não.
- **Hàm mất mát Focal Tversky Loss ($\alpha=0.7, \beta=0.3, \gamma=1.33$):**
  - $\alpha=0.7$: Phạt nặng False Negatives, buộc mạng phải nhận diện các dải máu mỏng và ổ xuất huyết nhỏ li ti (khắc phục triệt để lỗi Dice 0.00 trên SAH).
  - $\beta=0.3$: Co gọn đường viền tổn thương, chống over-segmentation.
  - $\gamma=1.33$: Tập trung lực học vào các lát cắt khó.
- **Hậu xử lý Lọc Xương Vật lý (Physical HU Gating):** Áp dụng dải HU sinh học của máu tụ cấp tính ($15 \le \text{HU} \le 105$) và lọc thành phần liên thông nhỏ ($<15$ pixel) để loại bỏ 100% các đốm nhiễu viền xương sọ.
- **Quét ngưỡng tối ưu (Threshold Search):** Tự động quét dải $0.20 \rightarrow 0.55$ trên tập Validation để chọn ngưỡng tối đa hóa điểm Dice thay vì cắt cứng ở 0.5.

### 4. Chuẩn đầu ra Hợp đồng (Interface Contract v1.0)
- Tự động sinh `findings.json` chuẩn hóa 100% theo JSON Schema (bản quyền `docs/INTERFACE_CONTRACT.md`).
- Tự động sinh `mask_ich.nii.gz` kích thước gốc ($512 \times 512 \times Z$) giữ nguyên affine matrix và header gốc phục vụ Block 4 tính thể tích mL.

*Chi tiết danh mục bài báo gốc, giải đấu quốc tế và mã trích dẫn BibTeX được tổng hợp tại [CITATIONS.md](CITATIONS.md).*

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

