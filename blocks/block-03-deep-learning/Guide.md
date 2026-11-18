# CẨM NANG TOÀN DIỆN: AI TRONG ĐÁNH GIÁ XUẤT HUYẾT NÃO (ICH) TRÊN CT
> **Dự án:** A VLM-Based System for Intracranial Hemorrhage Assessment on Brain CT  
> **Phạm vi trọng tâm:** Block 3: Deep Learning (Phát hiện, Phân loại 5 Subtype, Phân đoạn Mask 3D)  
> **Tác giả:** Vũ Minh Hoàng & Nguyễn Đình Hữu Minh  
> **Cập nhật:** Chuẩn hóa toàn bộ lý thuyết, bản chất toán học, kiểm thử so sánh & kịch bản bảo vệ

---

## 1. BẢN CHẤT LÂM SÀNG & VẬT LÝ HÌNH ẢNH

### 1.1. Bài toán cấp cứu & Vai trò của Block 3
Khi nghi ngờ bệnh nhân chấn thương đầu hoặc đột quỵ xuất huyết não (Intracranial Hemorrhage - ICH), bác sĩ cấp cứu cần trả lời ngay 3 câu hỏi:
1. **Có xuất huyết không?** (`Any_ICH`: Yes / No).
2. **Xuất huyết khoang nào trong 5 loại?**
   - **EDH** (Epidural): Ngoài màng cứng (hình thấu kính lồi hai mặt, cạnh hộp sọ).
   - **SDH** (Subdural): Dưới màng cứng (hình liềm ôm sát vòm sọ).
   - **SAH** (Subarachnoid): Dưới nhện (máu len lỏi thành các dải mỏng 1-2 pixel vào các rãnh não).
   - **IPH** (Intraparenchymal): Trong nhu mô (cục máu nằm trong thịt não).
   - **IVH** (Intraventricular): Trong não thất (máu tràn vào buồng dịch não tủy).
1. **Vị trí và đường viền khối máu tụ ở đâu?** (Phân đoạn Mask nhị phân 3D để phục vụ tính thể tích).

### 1.2. Tại sao bắt buộc dùng CT không cản quang (NCCT) thay vì X-Ray?
* **X-quang thường (X-Ray 2D):** Là ảnh chiếu phẳng 2D (chồng toàn bộ mô từ trước ra sau). Xương sọ ($+1000\text{ HU}$) cản tia X cực mạnh, trong khi máu tươi ($+50\text{ đến }+80\text{ HU}$) và nhu mô não ($+30\text{ đến }+40\text{ HU}$) chỉ chênh lệch nhau $15 - 30\text{ HU}$. Toàn bộ tín hiệu của máu bị xương sọ xóa sạch $\rightarrow$ **Về mặt vật lý, mắt người lẫn AI đều không thể thấy máu trên X-quang thường**.
* **CT sọ não không cản quang (Non-Contrast Head CT - NCCT):** Máy quét $360^\circ$ tái tạo các lát cắt mỏng $0.5 - 5\text{ mm}$, loại bỏ hoàn toàn hiện tượng chồng hình, độ phân giải tương phản cực cao $\rightarrow$ **Tiêu chuẩn vàng bất biến trong cấp cứu thần kinh**.

### 1.3. Luồng dữ liệu toàn hệ thống & Ranh giới trách nhiệm giữa các Block
```
[Ảnh CT 3D] ──> [Block 2: Preprocess] ──> [Block 3: Deep Learning]
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   ▼                                                               ▼
           findings.json                                                   mask_ich.nii.gz
     (Any_ICH + 5 Subtypes)                                              (Mask nhị phân 3D)
                   │                                                               │
                   │                                                               ▼
                   │                                                     [Block 4: Tính thể tích]
                   │                                                               │
                   │                                                           volume.json
                   │                                                          (mL thực tế)
                   │                                                               │
                   └───────────────────────────────┬───────────────────────────────┘
                                                   ▼
                                         [Block 5: VLM Model]
                                                   │
                                                   ▼
                                     Báo cáo y khoa có trích dẫn
```

* **Lưu ý ranh giới sống còn giữa Block 3 và Block 4:**
  - **Block 3 KHÔNG tính thể tích ra số mL cho người dùng.** Nhiệm vụ tính thể tích, đo độ lệch đường giữa (Midline Shift) và áp dụng công thức ABC/2 là của **Block 4**!
  - Nhưng Block 4 muốn tính được mL, nó bắt buộc phải lấy file `mask_ich.nii.gz` của Block 3 xuất ra.
  - Block 3 chịu trách nhiệm bảo toàn nguyên vẹn độ phân giải $512 \times 512 \times Z$ và ma trận tọa độ gốc (`affine`), giúp Block 4 chỉ việc lấy tổng số voxel nhân với thể tích 1 ô là ra số mL chính xác tuyệt đối mà không bị méo mó hình học.

---

## 2. BẢN CHẤT DỮ LIỆU & KỸ THUẬT 3 CỬA SỔ CT (MULTI-WINDOW PSEUDO-RGB)

### 2.1. Bản chất ảnh CT trong máy tính
* Một lát cắt CT $512 \times 512$ là một **ma trận 2 chiều gồm 262.144 ô số**.
* Giá trị mỗi ô là cường độ cản tia X theo đơn vị Hounsfield (HU):
  * Không khí: $-1000\text{ HU}$
  * Nước / Dịch não tủy (CSF): $0 \rightarrow +15\text{ HU}$
  * Mô não lành: $+30 \rightarrow +40\text{ HU}$
  * Máu tụ mới cấp tính: $+50 \rightarrow +80\text{ HU}$
  * Máu tụ bán cấp / hòa dịch: $+15 \rightarrow +45\text{ HU}$
  * Máu cục đậm / vôi hóa: $+90 \rightarrow +120\text{ HU}$
  * Xương sọ: $+1000\text{ HU}$

### 2.2. Kỹ thuật 3 Cửa sổ CT (Multi-Windowing - Quán quân RSNA 2019)
Mắt người và mô hình thị giác máy tính tiêu chuẩn chỉ nhận ảnh màu 3 kênh ($0 \rightarrow 1$). Dải HU của CT trải rộng từ $-1000$ đến $+3000$, nếu ép chung vào 1 ảnh xám sẽ mất hết tương phản.  
Ta lọc 3 dải HU chuyên dụng và gán vào 3 kênh màu RGB:

$$\text{Windowed}(HU) = \text{clip}\left(\frac{HU - (L - W/2)}{W}, 0.0, 1.0\right)$$

| Kênh | Cửa sổ CT | Level ($L$) | Width ($W$) | Dải HU hiển thị | Mục đích lâm sàng |
|:---:|:---:|:---:|:---:|:---:|:---|
| **R** | **Brain Window** | 40 | 80 | $[0, 80]$ | Thể hiện nhu mô não, phân biệt chất trắng/xám và phù não quanh khối máu |
| **G** | **Subdural Window** | 80 | 200 | $[-20, 180]$ | **Cực nhạy với xuất huyết dưới màng cứng (SDH) và khoang dưới nhện (SAH)** |
| **B** | **Bone Window** | 600 | 2800 | $[-800, 2000]$ | Cấu trúc xương sọ, đường nứt sọ và vôi hóa |

* **Trực quan hóa Pseudo-RGB:**
  - **Máu tụ cấp tính:** Sáng rực màu **vàng/cam** (tín hiệu đồng thời cao ở cả kênh Brain và Subdural).
  - **Hộp sọ:** Ánh màu **xanh dương/cyan** tách biệt (bão hòa ở kênh Bone nhưng tắt ở 2 kênh kia).
  - **Nhu mô não:** Màu **xám xanh** dịu tự nhiên.

### 2.3. Tại sao giữ nguyên độ phân giải gốc 512x512 (Cấm downsample 256x256)?
- Các dải máu dưới nhện (SAH) trong các rãnh não thường chỉ dày **1 đến 2 pixel** trên ma trận $512 \times 512$.
- Nếu dùng phép nội suy co ảnh về $256 \times 256$, giá trị HU của 1-2 pixel máu này sẽ bị hòa tan với dịch não tủy xung quanh ($0-15$ HU) $\rightarrow$ Vệt máu biến mất hoặc nhòe nhoẹt.
- **Giải pháp tối ưu GPU:** Giữ nguyên $512 \times 512$, sử dụng **PyTorch AMP (Automatic Mixed Precision fp16)**. VRAM tiêu thụ chỉ ~4.5 GB trên Tesla T4, chạy cực êm và không làm mất 1 pixel tổn thương nào!

### 2.4. Quy tắc chia tập (Split)
* **Dataset chính:** BHSD (192 ca CT 3D có Mask voxel, 6.404 lát cắt).
* **Nguyên tắc sống còn:** **Chia tập theo Bệnh nhân (Patient-Level Split)**.
  * Train: 134 ca (133 bệnh nhân, 4.437 lát)
  * Validation: 28 ca (28 bệnh nhân, 952 lát)
  * Test: 30 ca (30 bệnh nhân, 1.015 lát)
  * Độ trùng lặp: $\text{Overlap} = 0$. Tuyệt đối không chia ngẫu nhiên theo lát (gây rò rỉ dữ liệu / Data Leakage).

---

## 3. NHIỆM VỤ 1: PHÂN LOẠI ĐA NHÃN CHUỖI 3D (RESNET-34 + BI-GRU)

### 3.1. Bản chất luồng trích xuất đặc trưng
```
ẢNH 3 KÊNH (3 x 512 x 512)
        │
        ▼
[ BỘ TRÍCH XUẤT ĐẶC TRƯNG: RESNET-34 ]
• Các tầng tích chập (Conv 3x3) quét viền, cạnh, góc, hình khối.
• Đường nhảy cóc (Skip connection: F(x) + x) chống triệt tiêu gradient.
• Nén toàn bộ bức ảnh thành 1 vector đặc trưng duy nhất.
        │
        ▼
FEATURE VECTOR (v): 1 hàng gồm 512 con số đặc trưng
        │
        ▼
[ MÔ PHỎNG CUỘN PHIM: BIDIRECTIONAL GRU (Quét 2 chiều Z) ]
• Xâu chuỗi 35 lát cắt: Quét từ cằm lên đỉnh đầu và từ đỉnh đầu xuống cằm.
• Tổng hợp tính liên tục không gian 3D của khối máu tụ.
        │
        ▼
[ ĐẦU PHÂN LOẠI (Classification Head) ]
• Phép nhân ma trận: y = v * W + b
  - v có kích thước 1 x (hidden_dim * 2) = 1 x 256
  - W có kích thước 256 x 6 (gồm 6 cột trọng số song song)
• Cho qua hàm Sigmoid: P = 1 / (1 + e^(-y))
        │
        ▼
VECTOR XÁC SUẤT 6 NHÃN [p_ICH, p_EDH, p_SDH, p_SAH, p_IPH, p_IVH]
```

### 3.2. Bản chất con số 18 và 34 trong ResNet là gì?
Con số 18 hay 34 đại diện cho **số lượng tầng có chứa trọng số học được (Weight Layers)** trong mạng:
- **ResNet-18 (18 tầng):** Gồm 1 tầng tích chập đầu tiên + 8 khối nén (mỗi khối gồm 2 tầng conv $3 \times 3$, tổng là $8 \times 2 = 16$) + 1 tầng Linear cuối cùng = $1 + 16 + 1 = 18$ tầng.
- **ResNet-34 (34 tầng):** Gồm 1 tầng tích chập đầu tiên + 16 khối nén (tổng $16 \times 2 = 32$) + 1 tầng Linear cuối = $1 + 32 + 1 = 34$ tầng.

**Điểm mấu chốt:** 
Cả ResNet-18 và ResNet-34 **đều cho ra một vector đặc trưng có kích thước y hệt nhau là $1 \times 512$**!
- *Tại sao lại chọn 34 thay vì 18?*  
  ResNet-18 hơi ít tầng, trên ảnh CT sọ não có những vệt máu rãnh não (SAH) cực kỳ mỏng và tinh vi, ResNet-18 dễ bị "học non" (underfitting) bỏ qua các vân mỏng này. ResNet-34 có thêm 16 tầng để học các chi tiết vân viền sắc nét hơn mà vẫn nhẹ (chỉ 21 triệu tham số, không bị học vẹt như ResNet-50).

### 3.3. Tại sao có 6 nhãn đầu ra thay vì 5? Nhãn `Any_ICH` có thừa không?
Đầu ra là vector 6 xác suất: $\mathbf{p} = [p_{\text{ICH}}, p_{\text{EDH}}, p_{\text{SDH}}, p_{\text{SAH}}, p_{\text{IPH}}, p_{\text{IVH}}]$.
Nhãn `Any_ICH` (Có xuất huyết hay không) **là nhãn quan trọng số 1**:
1. **Phân luồng cấp cứu khẩn cấp (Triage):** Bác sĩ cấp cứu chỉ cần biết ca này có máu hay không trước tiên. Nếu `Any_ICH = 0`, loại trừ ngay nguy cơ xuất huyết.
2. **Không thể thay bằng `max(5 subtype)`:** Có những ca xuất huyết lan tỏa, hoặc máu nằm ở ranh giới phức tạp, điểm từng subtype con có thể chỉ lơ lửng ở $0.40 - 0.45$ (chưa đạt ngưỡng 0.5 để khẳng định là loại nào cụ thể), nhưng nhìn tổng quan toàn não thì chắc chắn có máu (`Any_ICH = 0.88`).
3. **Quy tắc giải đấu quốc tế:** RSNA phạt điểm nặng gấp đôi nếu đoán sai `Any_ICH`.

### 3.4. Bidirectional GRU là gì? Mô phỏng bác sĩ cuộn phim CT thế nào?
* ResNet-34 chỉ nhìn từng lát cắt 2D riêng lẻ, không biết lát cắt bên trên và bên dưới có gì.
* **Hành vi bác sĩ:** Bác sĩ cuộn con lăn chuột lên đỉnh đầu rồi cuộn ngược xuống cằm để xemáy làm server thì cho cơ bản kiểu kiểu ubuntu m đốm trắng có liên tục qua 3-4 lát cắt liền kề hay không (để phân biệt máu thật với vôi hóa/nhiễu xương).
* **Bi-GRU (Bidirectional Gated Recurrent Unit):** Là mạng học chuỗi quét theo 2 chiều:
  - Chiều tiến: Quét từ lát 1 $\rightarrow$ lát 35 (dưới cằm lên đỉnh đầu).
  - Chiều lùi: Quét từ lát 35 $\rightarrow$ lát 1 (đỉnh đầu xuống cằm).
  $\rightarrow$ Tại lát cắt thứ $z$, mô hình tổng hợp ngữ cảnh của cả các lát bên dưới và các lát bên trên, mô phỏng chính xác 100% hành vi cuộn chuột của bác sĩ!

### 3.5. Hàm mất mát Phân loại: BCEWithLogitsLoss
Vì đây là bài toán phân loại đa nhãn (1 bệnh nhân có thể bị cùng lúc 2-3 loại xuất huyết), ta dùng Binary Cross Entropy trên từng nhãn:
$$\mathcal{L}_{\text{cls}} = -\sum_{c=1}^6 w_c \left[ y_c \log(\sigma(z_c)) + (1 - y_c) \log(1 - \sigma(z_c)) \right]$$
Trong đó $w_c$ (`pos_weight = [2.0, 5.0, 3.0, 3.0, 3.0, 3.0]`) gán trọng số cao hơn cho các thể hiếm gặp (như EDH) để mô hình không bị thiên lệch.

---

## 4. NHIỆM VỤ 2: PHÂN ĐOẠN TỔN THƯƠNG (scSE ATTENTION U-NET)

### 4.1. Kiến trúc U-Net và vai trò của Cầu nối ngang (Skip Connection)
```
ẢNH ĐẦU VÀO (512 x 512)                                    MẶT NẠ MASK (512 x 512)
        │                                                        ▲
        ▼                                                        │
[ Lớp nén 1 ] (256x256) ──────── CẦU NỐI BẢO TOÀN ─────────> [ Lớp dãn 1 ] (256x256)
        │                                                        ▲
        ▼                                                        │
[ Lớp nén 2 ] (128x128) ──────── CẦU NỐI BẢO TOÀN ─────────> [ Lớp dãn 2 ] (128x128)
        │                                                        ▲
        ▼                                                        │
[ Lớp nén 3 ] (64x64)   ──────── CẦU NỐI BẢO TOÀN ─────────> [ Lớp dãn 3 ] (64x64)
        │                                                        ▲
        ▼                                                        │
[ ĐÁY CHỮ U ] (16x16)   ───────────────────────────────────> [ Lớp dãn 4 ] (32x32)
   (NỬA NÉN - ENCODER)                                          (NỬA DÃN - DECODER)
```

1. **Nửa thu nhỏ (Encoder):** Giảm dần kích thước ma trận ($512 \rightarrow 256 \rightarrow ... \rightarrow 16$). Mục đích: Nắm bắt **ngữ cảnh toàn cục (Context)**, biết được ca này có khối máu nằm ở vùng nào của não bộ (dùng nửa đầu ResNet-34 pretrained).
2. **Nửa phóng to (Decoder):** Dùng phép tích chập chuyển vị hoặc nội suy phóng to ma trận ngược lại ($16 \rightarrow 32 \rightarrow ... \rightarrow 512$).
3. **Cây cầu nối ngang (Skip Connection): Điểm mấu chốt:**
   - Ở tầng đáy $16 \times 16$, thông tin ngữ cảnh rất rõ nhưng **mất hoàn toàn tọa độ viền sắc nét**.
   - Cầu nối ngang copy nguyên vẹn ma trận độ phân giải cao từ tầng Encoder tương ứng ghép sang tầng Decoder.
   - **Công thức thành công:**  
     $$\text{Đáy chữ U (Hiểu ngữ cảnh có tổn thương)} + \text{Tầng Encoder trên (Tọa độ biên sắc nét)} = \text{Vẽ viền chính xác 100\%}$$

### 4.2. Khối scSE Attention Gate là gì?
Khối **scSE (Spatial and Channel Squeeze-and-Excitation)** là 2 chiếc kính lọc gắn thêm vào U-Net:
- **Spatial SE (Lọc không gian):** Tự động tạo mặt nạ làm mờ hộp sọ, chỉ tập trung mắt nhìn vào bên trong nhu mô não.
- **Channel SE (Lọc kênh màu):** Tự động tăng trọng số cho kênh màu nào đang phát hiện dấu hiệu máu (Brain / Subdural) và hạ bớt kênh xương.

### 4.3. Hàm mất mát Phân đoạn: Focal Tversky Loss ($\alpha, \beta, \gamma$)
Diện tích cục máu chỉ chiếm $< 1\%$ diện tích não ($99\%$ là não lành). Nếu dùng loss thường, AI sẽ lười đoán toàn 0 (não lành) mà vẫn đạt 99% accuracy!
Ta dùng chỉ số **Tversky Index**:

$$\text{Tversky} = \frac{\text{TP}}{\text{TP} + \alpha \cdot \text{FN} + \beta \cdot \text{FP}}$$

* $\text{TP}$ (True Positive): Số pixel máu AI đoán đúng.
* $\text{FP}$ (False Positive): Đoán thừa (mô lành mà tô đỏ).
* $\text{FN}$ (False Negative): **Bỏ sót máu (máu thật mà không nhận ra)**.
* **Tại sao đặt $\alpha = 0.7$ và $\beta = 0.3$?**  
  Trong y khoa, **bỏ sót máu ($\text{FN}$) nguy hiểm hơn nhiều so với đoán dư ($\text{FP}$)**. Đặt $\alpha = 0.7 > \beta = 0.3$ nghĩa là phạt AI thật nặng nếu dám bỏ sót các dải máu mỏng.
* **Hệ số $\gamma = 1.33$ (Focal):**
  $$\text{Focal Tversky} = (1 - \text{Tversky})^\gamma$$
  Lát cắt nào có cục máu to dễ vẽ rồi thì giảm điểm phạt; lát nào có vệt máu mỏng tang rãnh não khó nhận biết thì nhân lũy thừa phạt nặng lên để ép AI phải học bằng được!

---

## 5. HẬU XỬ LÝ, QUÉT NGƯỠNG & BẢNG ĐO KIỂM SO SÁNH (ABLATION STUDY)

### 5.1. Bỏ lọc tay cảm tính $15 \le \text{HU} \le 105$: Chuyển sang 3D Spatial Consistency
* **Sai lầm của lọc tay HU:** Máu loãng (bán cấp, hòa dịch não tủy) có thể có đậm độ dưới $15$ HU, còn máu cục đậm hoặc vôi hóa có thể vượt $105$ HU. Cắt gọt cứng $15-105$ HU sẽ tự tay cắt cụt mô bệnh học thật.
* **Giải pháp Quán quân MICCAI INSTANCE 2022:** 
  - Để scSE U-Net tự học bờ tổn thương thuần End-to-End.
  - Áp dụng **3D Connected Component Spatial Consistency**: Lọc bỏ các cụm liên thông không gian 3D có thể tích quá nhỏ ($< 25 \text{ mm}^3$) chỉ xuất hiện lác đác 1 lát cắt đơn lẻ mà không có sự liên tục trên các lát liền kề.

### 5.2. Quét ngưỡng toàn diện & Composite Score (Khắc phục nổ đốm đỏ giả)
* U-Net xuất ra xác suất từ $0.0$ đến $1.0$ cho mỗi pixel. Ta phải chọn một ngưỡng cắt (Threshold): trên ngưỡng thì tô đỏ (máu), dưới ngưỡng thì bỏ.
* **Đo kiểm trên 2 nhóm lát cắt:**
  - *Lát dương (lát có máu):* Tính điểm **Dice** (đo độ trùng khớp giữa hình AI vẽ với bác sĩ vẽ).
  - *Lát âm (lát não bình thường):* Đo **Specificity** = $\frac{\text{TN}}{\text{TN} + \text{FP}}$ (Trong 100 lát não lành, AI phải đoán đúng sạch bóng bao nhiêu % mà không vẽ bậy bất kỳ đốm đỏ nào).
* **Composite Score:**
  $$\text{Composite} = 0.6 \times \text{Dice}_{\text{lát dương}} + 0.4 \times \text{Specificity}_{\text{lát âm}}$$
  Nhờ tối ưu hàm này, ta tìm ra ngưỡng cắt tối ưu là **$0.35$**: vừa vẽ trúng máu ở lát bệnh, vừa sạch bóng ở lát lành (Specificity đạt $97.8\%$).

### 5.3. Bảng đo kiểm Ablation Study (Bằng chứng thuyết phục khi thuyết trình)

| Phương pháp Hậu xử lý | Dice (Lát Dương) | Specificity (Lát Âm) | Nhận xét Y khoa |
|---|---|---|---|
| 1. Ngưỡng thấp cũ ($Th=0.20$, chỉ đo lát dương) | 0.612 | 71.4% | **Nổ đốm đỏ giả** rải rác trong nhu mô lành |
| 2. Lọc tay cảm tính ($15 \le \text{HU} \le 105$) | 0.578 (-3.4%) | 96.2% | Bị **cắt cụt máu loãng** và máu bán cấp |
| **3. AI thuần + 3D Spatial Consistency (Đề xuất)** | **0.628 (+1.6%)** | **97.8% (Cực sạch)** | **Bảo toàn tổn thương, không vẽ bậy ở não lành** |

---

## 6. CHUẨN ĐẦU RA BÀN GIAO (CONTRACT DELIVERABLES)

Theo thỏa thuận tại `INTERFACE_CONTRACT.md`, Block 3 bàn giao đúng 2 artifact:

### 6.1. File `findings.json` (Giao cho Block 4 & Block 5)
```json
{
  "schema_version": "1.0",
  "case_id": "ID_0237f3c9_ID_40015688b9",
  "status": "ok",
  "created_at_utc": "2026-11-12T20:00:00Z",
  "model_name": "resnet34_bigru_classifier+scse_unet_focaltversky",
  "model_version": "v2.0-champion-colab",
  "input_space_id": "ID_0237f3c9:block2:v1",
  "presence": {
    "label": "positive",
    "score": 0.9712,
    "threshold": 0.5
  },
  "subtypes": {
    "EDH": { "score": 0.012, "present": false },
    "SDH": { "score": 0.884, "present": true },
    "SAH": { "score": 0.035, "present": false },
    "IPH": { "score": 0.021, "present": false },
    "IVH": { "score": 0.008, "present": false }
  },
  "masks": {
    "ich": { "artifact": "mask_ich.nii.gz", "space_id": "ID_0237f3c9:block2:v1" },
    "subtype": null
  },
  "localization": {
    "status": "mask-derived",
    "side": "right",
    "regions": [],
    "coordinate_system": "RAS+"
  }
}
```

### 6.2. File `mask_ich.nii.gz` (Giao cho Block 4 để tính thể tích mL)
* Định dạng NIfTI 3D ($512 \times 512 \times Z$).
* Phải bảo toàn nguyên vẹn ma trận hình học (`affine`) và kích thước voxel (`spacing_xyz_mm`) của volume CT gốc.
* Công thức tính thể tích của Block 4:
  $$V_{\text{mL}} = \frac{N_{\text{voxel foreground}} \times s_x \times s_y \times s_z}{1000}$$

---
---

## 8. KỊCH BẢN THUYẾT TRÌNH 4 BƯỚC BỎ TÚI (DÀNH CHO BẠN)

Khi thuyết trình, bạn chỉ cần trình bày đúng 4 ý theo thứ tự sau:

> **1. Vấn đề của phương pháp tiếp cận thông thường:**  
> *"Nếu chỉ dùng các mô hình 2D thông thường, chúng ta sẽ gặp 3 bế tắc: Một là mất ngữ cảnh 3D giữa các lát cắt; Hai là nếu hạ ảnh xuống $256 \times 256$ sẽ làm biến mất các dải máu mỏng rãnh não (SAH); Ba là nếu dùng quy tắc lọc tay 15-105 HU sẽ vô tình cắt cụt vùng máu loãng bán cấp."*
>
> **2. Giải pháp kế thừa từ 2 Quán quân quốc tế (RSNA & MICCAI):**  
> *"Nhóm đã tích hợp 2 kiến trúc chuẩn:*  
> *- Với bài toán Phân loại: Dùng **ResNet-34 kết hợp Bi-GRU 2 chiều** để mô phỏng chính xác cách bác sĩ cuộn chuột qua lại giữa các lát cắt.*  
> *- Với bài toán Phân đoạn: Dùng **scSE Attention U-Net** kết hợp hàm mất mát **Focal Tversky Loss** để phạt nặng việc bỏ sót máu mỏng, đồng thời giữ nguyên độ phân giải gốc $512 \times 512$ bằng PyTorch AMP fp16."*
>
> **3. Bằng chứng thực nghiệm (Ablation Study):**  
> *"Nhóm đã thực hiện đo kiểm so sánh trực tiếp:*  
> *- Nếu chỉ tối ưu lát dương ở ngưỡng thấp 0.20: Độ đặc hiệu lát âm chỉ đạt 71.4%, gây nổ đốm đỏ giả ở não lành.*  
> *- Nếu dùng bộ lọc tay cố định 15-105 HU: Điểm Dice giảm 3.4% vì cắt cụt máu loãng.*  
> *- Phương pháp của nhóm (AI thuần + 3D Spatial Consistency): Đạt điểm Dice cao nhất (0.628) và độ đặc hiệu trên lát âm sạch tới 97.8%."*
>
> **4. Bàn giao Hợp đồng cho Block 4:**  
> *"Block 3 tuân thủ nghiêm ngặt ranh giới hợp đồng: Chúng tôi xuất file `findings.json` và file mask 3D `mask_ich.nii.gz` ở kích thước gốc $512 \times 512 \times Z$. Nhờ bảo toàn chính xác ma trận tọa độ, Block 4 chỉ việc nhận mask này để tính thể tích (mL) và đo độ lệch đường giữa mà không bị bất kỳ sai số hình học nào."*

---

## 8. MA TRẬN ĐỐI CHIẾU 3 ĐỘI ĐỈNH CAO THẾ GIỚI & CHECKLIST TIẾN ĐỘ

### 8.1. Ma trận so sánh 3 công trình lớn vs Phân hệ Block 3 của chúng ta

| Tiêu chí Kỹ thuật | Đội 1: Qure.ai (*The Lancet* 2018) | Đội 2: SeuTao (Quán quân RSNA 2019) | Đội 3: NVIDIA & Team Vegetable (Quán quân MICCAI INSTANCE 2022) | Phân hệ Block 3 của chúng ta |
| :--- | :--- | :--- | :--- | :--- |
| **Mục tiêu nghiên cứu** | Phát hiện ICH, 5 subtype, gãy xương, đường giữa não | Phân loại đa nhãn chuỗi 3D cho 6 nhãn ICH | Phân đoạn voxel 3D xuất huyết não không cản quang | **Tích hợp cả 2:** Phân loại đa nhãn chuỗi 3D + Phân đoạn voxel 3D |
| **Dữ liệu huấn luyện** | 313.318 ca CT (tập nội bộ) + 500 ca CQ500 | 25.000 ca CT thi đấu RSNA | 200 ca CT 3D NCCT đa trung tâm | Tập BHSD (192 ca CT 3D có Mask voxel, chia theo bệnh nhân) |
| **Độ phân giải xử lý** | $512 \times 512$ nguyên bản | Co về $256 \times 256$ hoặc $384 \times 384$ để train nhanh | $512 \times 512$ trục $X, Y$, giữ nguyên dị hướng trục $Z$ | **$512 \times 512$ nguyên bản** (dùng PyTorch AMP fp16 để tối ưu VRAM) |
| **Kỹ thuật Cửa sổ CT (Windowing)** | Đa cửa sổ kết hợp lọc ngưỡng HU | 3 Cửa sổ chuyên dụng: Brain (40, 80), Subdural (80, 200), Bone (600, 2800) | Chuẩn hóa HU dị hướng z-score theo từng lát cắt | **Pseudo-RGB 3 Cửa sổ:** Brain (R), Subdural (G), Bone (B) |
| **Kiến trúc Phân loại** | ResNet sâu + Random Forest | **ResNet-34 Feature Extractor + Bidirectional GRU** | Không làm phân loại (chỉ làm phân đoạn) | **Kế thừa Quán quân RSNA:** ResNet-34 + Bi-GRU quét 2 chiều trục $Z$ |
| **Chiến lược Phân loại** | **Học Đa nhiệm (Multi-task):** Cùng học `Any_ICH` và 5 subtype | **Học Đa nhiệm (Multi-task):** Vector 6 nhãn đồng thời | Không áp dụng | **Học Đa nhiệm (Multi-task):** Tránh sai số dây chuyền Cascading Error |
| **Kiến trúc Phân đoạn** | Sơ khai (chưa có mask voxel 3D chi tiết) | Không làm phân đoạn | **2.5D / 3D nnU-Net kết hợp Spatial & Channel Attention** | **scSE Attention U-Net:** ResNet-34 Encoder + scSE Attention Gates |
| **Hàm mất mát Phân đoạn** | Cross-Entropy tiêu chuẩn | BCEWithLogitsLoss có pos_weight | Dice Loss + Binary Cross-Entropy | **Focal Tversky Loss** ($\alpha=0.7, \beta=0.3, \gamma=1.33$): Đặc trị dải máu nhỏ |
| **Hậu xử lý (Post-processing)** | Lọc ngưỡng xác suất cố định | Ensembling đa mô hình | **3D Connected Component Consistency** | **3D Spatial Consistency** (loại bỏ nốt giả $<25\text{ mm}^3$ không liên tục) |
| **Đầu ra hệ thống** | Bảng xác suất chẩn đoán | Vector xác suất 6 nhãn | Mặt nạ nhị phân 3D | **Chuẩn Hợp đồng Nhóm:** `findings.json` (Schema v1.0) + `mask_ich.nii.gz` (mL thể tích) |

---

### 8.2. Bảng Checklist Tiến độ (Cái gì làm rồi [x], cái gì xem xét đưa vào [ ], cái gì loại bỏ [-])

#### Nhóm 1: `[x]` Đã hoàn thành 100% trong Block 3 MVP (Đã có sản phẩm thật kiểm chứng)
- [x] **Tiền xử lý 3 Cửa sổ CT (Brain, Subdural, Bone):** Chuyển HU thành ảnh Pseudo-RGB chuẩn.
- [x] **Giữ nguyên độ phân giải $512 \times 512$:** Không co ảnh, bảo toàn tổn thương nhỏ dạng rãnh.
- [x] **Cơ chế lưu đệm SSD (On-Disk Caching):** Giảm RAM tiêu thụ từ $>20\text{ GB}$ xuống $<800\text{ MB}$, dứt điểm lỗi crash OOM.
- [x] **Tăng cường dữ liệu y khoa (Medical Augmentation):** Xoay $\pm 10^\circ$, dịch $\pm 4\%$, lật ngang bán cầu não. Không lật dọc.
- [x] **Mô hình Phân loại chuỗi 3D (ResNet-34 + Bi-GRU):** Đã train, AUROC cao, xuất `best_classifier.pth` ($87.3\text{ MB}$).
- [x] **Mô hình Phân đoạn scSE Attention U-Net 512x512:** Huấn luyện với Focal Tversky Loss, xuất `best_unet.pth` ($98.4\text{ MB}$).
- [x] **Hậu xử lý 3D Spatial Consistency:** Thay thế lọc tay HU, xóa sạch nốt giả nhu mô não lành ($97.8\%$ độ sạch).
- [x] **Bàn giao Hợp đồng Nhóm (Interface Contract v1.0):**
  - Xuất `findings.json` đạt $100\%$ JSON Schema, định danh đúng IPH, IVH và bên tổn thương (`left`).
  - Xuất `mask_ich.nii.gz` ($512 \times 512 \times 33$) tính ra thể tích $0.76\text{ mL}$ bàn giao cho Block 4.

#### Nhóm 2: `[ ]` Đang xem xét đưa vào giai đoạn 2 (Roadmap tối ưu hóa độ chính xác)
- [ ] **Tối ưu ngưỡng động theo từng Subtype (Per-class Threshold):** Quét tìm ngưỡng riêng (hạ ngưỡng EDH xuống $\approx 0.25$, SAH xuống $\approx 0.30$) để tăng vọt độ nhạy tổn thương hiếm.
- [ ] **Test-Time Augmentation (TTA lúc suy luận):** Chạy lật ngang bán cầu não lúc chẩn đoán để tăng thêm $1-2\%$ Dice hoàn toàn miễn phí.
- [ ] **Huấn luyện chu kỳ dài (Extended Training 30 Epochs):** Tăng số epoch từ 10 lên 30 với `CosineAnnealingWarmRestarts`.
- [ ] **Đầu vào 2.5D Multi-slice Stacking $[z-1, z, z+1]$:** Đưa 3 lát cắt liên tiếp vào U-Net để mạng học ngữ cảnh 3 chiều ngay từ đầu vào.
- [ ] **Thử nghiệm Encoder tiên tiến hơn (ConvNeXt-Tiny / EfficientNet-B3):** Thử nghiệm kiến trúc hiện đại hơn ResNet.
- [ ] **Mặt nạ đa nhãn voxel từng Subtype (Multi-class Mask):** Bóc tách riêng từng màu cho từng loại xuất huyết (EDH, IPH, IVH, SAH, SDH).

#### Nhóm 3: `[-]` Đã nghiên cứu thử nghiệm và loại bỏ (Ablation Rejections kèm lý do)
- [-] **Kiến trúc phân tầng 2 giai đoạn (Cascaded 2-Stage: Any_ICH $\rightarrow$ Subtypes):** Loại bỏ vì gây sai số dây chuyền (*Error Propagation*). Chọn Multi-task Joint Learning theo Lancet 2018 và RSNA 2019 để chia sẻ đặc trưng bổ trợ lẫn nhau.
- [-] **Lọc cứng bằng khoảng HU ($15 \le \text{HU} \le 105$):** Loại bỏ vì làm giảm $-3.4\%$ điểm Dice (cắt cụt máu loãng và máu bán cấp).
- [-] **Co ảnh về $256 \times 256$:** Loại bỏ vì làm hòa tan dải máu mỏng rãnh não SAH chỉ rộng 1-2 pixel.
- [-] **Lật dọc ảnh (Vertical Flip):** Loại bỏ vì vi phạm giải phẫu (đảo lộn đỉnh sọ và nền sọ).

