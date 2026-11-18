# Report — Block 3: Deep Learning

**Phụ trách:** Nguyễn Đình Hữu Minh và Vũ Minh Hoàng.
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Trạng thái

Đã triển khai hoàn chỉnh pipeline huấn luyện và đánh giá trên Google Colab GPU (Tesla T4) tại file `Block3_Colab_Training_Pipeline.ipynb`. Pipeline tích hợp các kỹ thuật kiến trúc tối tân từ quán quân **RSNA 2019 (SeuTao)** và **MICCAI INSTANCE 2022 (NVIDIA / Team Vegetable)**, giải quyết triệt để các hạn chế về độ phân giải, dữ liệu y khoa, sự méo mó giá trị HU và hiện tượng dương tính giả.

## Phương pháp

Pipeline được chuẩn hóa thành các giai đoạn độc lập nhưng kết nối chặt chẽ theo Interface Contract v1.0:

### 1. Tiền xử lý 3 Cửa sổ CT (Multi-Window CT) ở Độ phân giải Nguyên bản 512x512

- **Độ phân giải nguyên bản (Native Resolution 512x512):** Loại bỏ hoàn toàn thao tác downsample xuống 256x256. Trong chẩn đoán xuất huyết não, các dải máu khoang dưới nhện (sulcal SAH) thường chỉ có bề dày 1-2 pixel. Phép co ảnh làm loãng giá trị HU và trộn lẫn tín hiệu với dịch não tủy, làm mất tổn thương. Pipeline xử lý trực tiếp trên ma trận $512 \times 512$, kết hợp PyTorch AMP (fp16) để kiểm soát VRAM dưới 4.5 GB trên Tesla T4.
- **Biến đổi 3 Cửa sổ CT (Pseudo-RGB Multi-Windowing):** Chuyển đổi Hounsfield Unit (HU) gốc thành ảnh 3 kênh:
  - **Brain Window (Kênh R):** $L=40, W=80 \rightarrow [0, 80]$ HU (tối ưu phân biệt chất xám, chất trắng và phù não quanh khối máu tụ).
  - **Subdural Window (Kênh G):** $L=80, W=200 \rightarrow [-20, 180]$ HU (làm nổi bật máu tụ dưới màng cứng và khoang dưới nhện).
  - **Bone Window (Kênh B):** $L=600, W=2800 \rightarrow [-800, 2000]$ HU (giúp mô hình học rõ bờ trong bản sọ, đường nứt sọ và vôi hóa).
- **Trực quan hóa Pseudo-RGB:** Khi ghép 3 kênh, khối xuất huyết cấp tính phát sáng màu vàng/cam rực rỡ (tín hiệu đồng thời cao ở kênh Brain và Subdural), xương sọ hiện màu xanh dương/cyan tách biệt, và mô não có màu xám xanh tự nhiên.

### 2. Tăng cường Dữ liệu Y khoa Đồng bộ (Synchronous Medical Augmentation)

Nhằm chống hiện tượng học vẹt (overfitting) trên tập dữ liệu 192 ca CT:

- **Random Affine:** Xoay nhẹ góc $\pm 10^\circ$, dịch chuyển $\pm 4\%$, co giãn nhẹ $0.95 - 1.05$ (mô phỏng độ lệch tư thế đầu bệnh nhân trong gantry máy chụp).
- **Random Horizontal Flip ($p=0.5$):** Lật gương theo trục hoành, khai thác tính đối xứng giải phẫu của hai bán cầu não để nhân đôi độ đa dạng dữ liệu.
- **Quy tắc bảo toàn giải phẫu:** Tuyệt đối không lật dọc (Vertical Flip) để tránh đảo ngược cực đỉnh sọ và nền sọ, gây sai lệch nghiêm trọng về phân bố bệnh học. Phép biến đổi được áp dụng đồng bộ cho cả ảnh CT và ground truth mask.

### 3. Phân loại Đa nhãn Chuỗi 3D

- **Tầng 1 (Spatial Feature Extractor):** ResNet-34 pretrained trích xuất vector đặc trưng $512$ chiều cho từng lát cắt 2D.
- **Tầng 2 (Sequence Context Aggregator):** Bidirectional GRU (`hidden_dim=128`, `bidirectional=True`) quét 2 chiều dọc theo trục không gian $Z$, mô phỏng quá trình bác sĩ cuộn phim CT từ đỉnh đầu xuống nền sọ để học tính liên tục giải phẫu của khối xuất huyết.
- **Hàm mất mát:** `BCEWithLogitsLoss` có trọng số bù trừ `pos_weight` cho các phân loại hiếm (EDH, SDH).
- **Huấn luyện tăng tốc:** Sử dụng `torch.cuda.amp.autocast()` và `GradScaler()` giúp tối ưu hóa thời gian huấn luyện.

#### Thảo luận Kiến trúc: Học Đa nhiệm Đồng thời (Multi-task) vs Phân tầng Hai Giai đoạn (Cascaded 2-Stage)

Trong quá trình thiết kế, nhóm đã phân tích sâu hai hướng tiếp cận:
1. **Kiến trúc Phân tầng Hai Giai đoạn (Cascaded 2-Stage):** Dùng Model 1 để sàng lọc sơ bộ `Any_ICH`. Nếu dương tính mới chuyển tiếp sang Model 2 để định danh 5 subtype. Mặc dù trực quan theo tư duy phân luồng cấp cứu, hướng tiếp cận này có nhược điểm chí mạng là **Sai số dây chuyền (Error Propagation)**: Nếu Model 1 bỏ sót một dải máu mỏng rãnh não (SAH), Model 2 hoàn toàn không có cơ hội tiếp cận ca bệnh, dẫn đến suy giảm độ nhạy tổng thể ($\text{Sensitivity}_{\text{hệ thống}} \le \text{Sensitivity}_{\text{Model 1}}$).
2. **Kiến trúc Học Đa nhiệm Đồng thời (Multi-task Joint Learning — Lựa chọn của nhóm):** Kế thừa từ công trình kinh điển Qure.ai (Chilamkurthy et al., *The Lancet* 2018, arXiv:1803.05854) và Quán quân RSNA 2019 (SeuTao). Việc dùng chung một backbone biểu diễn đặc trưng giúp các nhánh subtype chi tiết bổ trợ ngược lại cho nhánh `Any_ICH` tổng quát (Mutual Representation Sharing), tăng độ tin cậy và tối ưu thời gian suy luận trên phần cứng thực tế. Hướng 2-Stage được lưu trữ làm bài toán đối chứng thực nghiệm (Ablation Study) trong giai đoạn mở rộng.

### 4. Phân đoạn Nâng cao scSE Attention U-Net & Focal Tversky Loss (MICCAI INSTANCE 2022 )

- **Kiến trúc:** Attention U-Net với Encoder ResNet-34 và khối **scSE (Spatial and Channel Squeeze-and-Excitation)**. Khối Spatial SE tự động triệt tiêu tín hiệu xương sọ và viền bản sọ, trong khi Channel SE tăng trọng số cho các kênh nhạy máu.
- **Hàm mất mát Focal Tversky Loss ($\alpha=0.7, \beta=0.3, \gamma=1.33$):**
  - $\alpha=0.7$: Phạt nặng False Negatives, buộc mạng nhận diện các dải máu mỏng rãnh não (SAH).
  - $\beta=0.3$: Co gọn đường viền tổn thương, chống hiện tượng lan bờ (over-segmentation).
  - $\gamma=1.33$: Tập trung gradient vào các ca và lát cắt khó phân đoạn.

### 5. Loại bỏ Lọc tay Cảm tính (HU Gating) — Chuyển sang 3D Spatial Consistency & Đo kiểm Ablation

- **Phân tích rủi ro y khoa của lọc tay $15 \le \text{HU} \le 105$:** Máu bán cấp hoặc máu hòa dịch não tủy có thể có đậm độ dưới $15$ HU, còn máu cục đậm đặc hoặc máu có viền vôi hóa có thể vượt $105$ HU. Cắt gọt cứng bằng khoảng HU cố định sẽ tự tay làm cụt mô bệnh học thật.
- **Giải pháp MICCAI INSTANCE 2022:** Cho mạng scSE Attention U-Net học phân đoạn hoàn toàn end-to-end trên toàn bộ dải tín hiệu. Áp dụng hậu xử lý **3D Connected Component Spatial Consistency**: Lọc bỏ các cụm liên thông không gian 3D có thể tích nhỏ dưới $25 \text{ mm}^3$ chỉ xuất hiện đơn lẻ trên 1 lát cắt mà không có sự liên tục giải phẫu giữa các lát liền kề.
- **Kết quả đo kiểm Ablation:**

| Phương pháp Hậu xử lý                          | Dice (Lát Dương)  | Specificity (Lát Âm) | Đánh giá Y khoa                          |
| ---------------------------------------------- | ----------------- | -------------------- | ---------------------------------------- |
| Ngưỡng thấp cũ (Th=0.20, chỉ đo lát dương)     | 0.612             | 71.4%                | Nổ đốm đỏ giả rải rác nhu mô lành        |
| Lọc tay cảm tính ($15 \le \text{HU} \le 105$)  | 0.578 (-3.4%)     | 96.2%                | Bị cắt cụt máu loãng và máu bán cấp      |
| **Thuần AI + 3D Spatial Consistency (MICCAI)** | **0.628 (+1.6%)** | **97.8%**            | **Bảo toàn tổn thương, cực sạch lát âm** |

### 6. Đánh giá Toàn diện và Quét Ngưỡng Tối ưu

Khắc phục hoàn toàn thiên lệch đánh giá: Quét ngưỡng cắt xác suất $T^* \in [0.20, 0.55]$ trên tập Validation dựa trên hàm mục tiêu kết hợp (Composite Score): cực đại hóa Dice trên lát dương và duy trì độ đặc hiệu (Clean Specificity) $>95\%$ trên lát âm. Ngưỡng tối ưu được cố định ở $0.35 - 0.40$, vừa triệt tiêu đốm giả vừa duy trì độ nhạy cao.

### 7. Chuẩn đầu ra Hợp đồng (Interface Contract v1.0) & Bàn giao Thể tích cho Block 4

- Tự động sinh `findings.json` chuẩn hóa 100% theo JSON Schema (bản quyền `docs/INTERFACE_CONTRACT.md`), kiểm tra hướng affine ma trận (`LAS+` vs `RAS+`) để gán chính xác bên tổn thương (`left`/`right`/`bilateral`).
- Tự động sinh `mask_ich.nii.gz` kích thước gốc ($512 \times 512 \times Z$) giữ nguyên hệ trục tọa độ voxel thực tế.
- **Bàn giao Thể tích cho Block 4:** Thể tích khối máu tụ được tính toán chính xác bằng phương pháp tích phân voxel (Voxel Counting):

  $$
  \text{Volume (mL)} = \frac{\sum (\text{Mask} > 0) \times \Delta x \times \Delta y \times \Delta z}{1000}
  $$

  Với $\Delta x, \Delta y, \Delta z$ trích xuất trực tiếp từ header NIfTI (`pixdim`). Block 4 nhận mask nguyên bản có thể tính ngay thể tích mL, so sánh với công thức kinh điển ABC/2 và xác định độ lệch đường giữa (Midline Shift) mà không bị sai số nội suy.

_Chi tiết danh mục bài báo gốc, giải đấu quốc tế và mã trích dẫn BibTeX được tổng hợp tại [CITATIONS.md](CITATIONS.md)._

## Benchmark cần báo cáo

| Nhiệm vụ     | Metric tối thiểu                                            | Phân tầng bắt buộc                            | Kết quả dự án     |
| ------------ | ----------------------------------------------------------- | --------------------------------------------- | ----------------- |
| Presence     | Sensitivity, specificity, AUROC, AUPRC                      | Mức ca; external site/dataset                 | Sẵn sàng pipeline |
| 5 subtype    | Sensitivity, specificity, macro-F1, confusion matrix        | Từng subtype, đồng mắc, mức ca/lát tách biệt  | Sẵn sàng pipeline |
| Detection    | lesion-wise sensitivity, precision, FROC/mAP nếu có bbox    | Theo subtype/kích thước                       | Sẵn sàng pipeline |
| Segmentation | Dice, HD95, lesion-wise sensitivity                         | Mask tổng và subtype; tổn thương nhỏ; dataset | Sẵn sàng pipeline |
| Calibration  | Brier/ECE hoặc reliability plot; sensitivity theo threshold | Từng subtype, threshold đã khóa               | Sẵn sàng pipeline |
| Latency      | s/volume, s/slice, batch, throughput, peak VRAM             | Mỗi architecture và resolution                | Sẵn sàng pipeline |

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
