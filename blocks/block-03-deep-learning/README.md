# Block 3 — Deep Learning (ICH Presence, 5 Subtypes & 3D Lesion Segmentation)

> **Trạng thái:** Đã hoàn thiện pipeline huấn luyện tối tân tích hợp kỹ thuật Quán quân **RSNA 2019 (SeuTao)** & **MICCAI INSTANCE 2022 (NVIDIA)** tại `notebooks/Block3_Colab_Training_Pipeline.ipynb`.
> Đầu ra tuân thủ nghiêm ngặt [Interface Contract v1.0](../../docs/INTERFACE_CONTRACT.md): sinh tự động `findings.json` (100% đạt chuẩn JSON Schema) và `mask_ich.nii.gz` ($512 \times 512 \times Z$) phục vụ tính toán thể tích mL (Block 4) và sinh báo cáo VLM (Block 5).

---

## 🗂️ Cấu Trúc Thư Mục Tinh Gọn (Production-Ready)

```text
blocks/block-03-deep-learning/
├── README.md               # Sơ đồ chỉ mục và hướng dẫn vận hành Block 3
├── Report.md               # Báo cáo chính thức (Phương pháp, baseline, kết quả benchmark)
├── Guide.md                # Đặc tả nhiệm vụ và yêu cầu từ đề bài dự án
├── CITATIONS.md            # Toàn văn trích dẫn học thuật, phân tích paper và BibTeX
├── requirements.txt        # Danh mục thư viện Python
│
├── notebooks/              # Sổ tay huấn luyện GPU Tesla T4 (RSNA + MICCAI)
│   └── Block3_Colab_Training_Pipeline.ipynb
│
├── scripts/                # Script sinh tự động và kiểm thử notebook
│   └── build_colab_notebook.py
│
├── block3/                 # Package Python chính thức của Block 3
│   ├── configs/            # Config phần cứng, siêu tham số và JSON Schema kiểm định
│   ├── data/               # Danh sách phân chia tập (splits) và kiểm kê dataset
│   ├── src/                # Mã nguồn tiền xử lý, kiểm tra, xác thực đầu ra
│   ├── tests/              # Bộ kiểm thử tự động (Unit tests & Schema validator)
│   └── results/            # Kết quả đo đạc độ trễ và benchmarks
│
└── data_raw/               # [Git-ignored] Thư mục chứa dữ liệu CT thô cục bộ (16.5 GB)
    ├── MBH_Train_2025_case-label/
    └── MBH_Train_2025_voxel-label/
```

---

## 🚀 Hướng Dẫn Sử Dụng Nhanh

### 1. Huấn luyện Mô hình Đột phá trên Google Colab (GPU Tesla T4)
1. Tải file [notebooks/Block3_Colab_Training_Pipeline.ipynb](notebooks/Block3_Colab_Training_Pipeline.ipynb) lên [Google Colab](https://colab.research.google.com/).
2. Chọn **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Chọn **T4 GPU**.
3. Chạy toàn bộ (**Run all**). Pipeline sẽ:
   - Tự động tải 192 ca CT chuẩn BHSD từ Hugging Face.
   - Huấn luyện Bộ phân loại chuỗi 3D (ResNet34 + Bi-GRU).
   - Huấn luyện Bộ phân đoạn scSE Attention U-Net với Focal Tversky Loss.
   - Quét ngưỡng tối ưu + lọc nhiễu xương vật lý (Physical HU Gating $15 \le \text{HU} \le 105$).
   - Xuất `findings.json`, `mask_ich.nii.gz` ($512 \times 512$) và đóng gói tải về `Block3_Champion_Handover.zip`.

### 2. Kiểm thử Cục bộ (Local Verification — Không cần tải dữ liệu nặng)
Chạy tại thư mục `blocks/block-03-deep-learning/`:

```bash
# Kiểm tra phân chia tập bệnh nhân (134 Train / 28 Val / 30 Test, 0 overlap)
python block3/src/check_splits.py

# Chạy kiểm thử pipeline end-to-end trên dữ liệu tổng hợp
python block3/src/smoke_test.py --synthetic

# Kiểm tra tính hợp lệ của file findings.json theo Interface Contract v1.0
python block3/src/validate_output.py <đường_dẫn_tới_findings.json>
```

---

## 📖 Tham Khảo Học Thuật
Mọi bài báo gốc (MICCAI, IEEE TMI, ISBI, Nature Scientific Reports, Radiology: AI) cùng mã nguồn vô địch RSNA 2019 và mã BibTeX chuẩn được lưu trữ đầy đủ tại [CITATIONS.md](CITATIONS.md).
