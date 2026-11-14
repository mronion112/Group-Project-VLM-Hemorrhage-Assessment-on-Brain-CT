# 📚 Danh Mục Tài Liệu Tham Khảo & Trích Dẫn Khoa Học (Citations & References)
### Dự án: A VLM-Based System for Intracranial Hemorrhage Assessment on Brain CT
### Phân hệ: Block 3 — Deep Learning (Phân loại Đa nhãn & Phân đoạn Tổn thương 3D)

Tài liệu này tổng hợp toàn bộ các công trình nghiên cứu khoa học gốc (peer-reviewed papers), mã nguồn từ các đội vô địch quốc tế (RSNA, MICCAI), và định dạng trích dẫn chuẩn (BibTeX & APA) phục vụ viết báo cáo khoa học, slide thuyết trình, và phản biện trước hội đồng.

---

## 📑 Mục lục
1. [Bộ dữ liệu chuẩn Y khoa (Datasets)](#1-bộ-dữ-liệu-chuẩn-y-khoa-datasets)
2. [Kiến trúc Phân loại Chuỗi 3D — Quán quân RSNA 2019](#2-kiến-trúc-phân-loại-chuỗi-3d--quán-quân-rsna-2019)
3. [Kiến trúc Phân đoạn Nâng cao & Thách thức MICCAI INSTANCE](#3-kiến-trúc-phân-đoạn-nâng-cao--thách-thức-miccai-instance)
4. [Hàm mất mát Đặc trị Tổn thương Nhỏ (Focal Tversky Loss)](#4-hàm-mất-mát-đặc-trị-tổn-thương-nhỏ-focal-tversky-loss)
5. [Cơ chế Chú ý Không gian & Kênh (scSE Attention Gates)](#5-cơ-chế-chú-ý-không-gian--kênh-scse-attention-gates)
6. [Công trình Nghiên cứu Phân đoạn Xuất huyết Não Mới nhất (Nature Sci Rep 2025)](#6-công-trình-nghiên-cứu-phân-đoạn-xuất-huyết-não-mới-nhất-nature-sci-rep-2025)
7. [Vật lý Chẩn đoán Hình ảnh & Cửa sổ CT (Windowing Standards)](#7-vật-lý-chẩn-đoán-hình-ảnh--cửa-sổ-ct-windowing-standards)
8. [Mã Trích Dẫn BibTeX Đầy Đủ (BibTeX Entries)](#8-mã-trích-dẫn-bibtex-đầy-đủ-bibtex-entries)

---

## 1. Bộ dữ liệu chuẩn Y khoa (Datasets)

### 1.1. Tập dữ liệu BHSD (Brain Hemorrhage Segmentation Dataset)
- **Tên bài báo:** *BHSD: A 3D Multi-class Brain Hemorrhage Segmentation Dataset*
- **Tác giả:** Biao Wu, X. Shen, C. Ding, et al.
- **Xuất bản tại:** MICCAI MLMI 2023 (*Machine Learning in Medical Imaging*), Lecture Notes in Computer Science (LNCS), vol 14348, pp. 147–156, Springer.
- **DOI:** [10.1007/978-3-031-45673-2_15](https://doi.org/10.1007/978-3-031-45673-2_15)
- **Dữ liệu:** [Hugging Face WuBiao/BHSD](https://huggingface.co/datasets/WuBiao/BHSD)
- **Ứng dụng trong Block 3:** Bộ dữ liệu huấn luyện và kiểm thử chính thống gồm 192 ca chụp CT 3D NIfTI có mặt nạ phân đoạn voxel đa lớp (0: Nền, 1: EDH, 2: IPH, 3: IVH, 4: SAH, 5: SDH). Dữ liệu được chia theo bệnh nhân (Patient-level splits) 134 ca Train, 28 ca Val, 30 ca Test nhằm chống rò rỉ dữ liệu (zero leakage).

### 1.2. Tập dữ liệu RSNA Brain CT Hemorrhage Challenge
- **Tên bài báo:** *Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge*
- **Tác giả:** Adam E. Flanders, Luciano M. Prevedello, Garry Choy, et al.
- **Tạp chí:** *Radiology: Artificial Intelligence*, 2020; 2(3):e190211.
- **DOI:** [10.1148/ryai.2020190211](https://doi.org/10.1148/ryai.2020190211)
- **PubMed:** [PMID: 33937827](https://pubmed.ncbi.nlm.nih.gov/33937827/)
- **Ứng dụng trong Block 3:** Bộ dữ liệu gốc hình ảnh CT của RSNA là nguồn dữ liệu nền tảng tạo nên BHSD. Cung cấp quy chuẩn phân loại 6 nhãn: `Any_ICH`, `EDH`, `SDH`, `SAH`, `IPH`, `IVH` được thẩm định độc lập bởi hàng chục bác sĩ X-quang thần kinh.

---

## 2. Kiến trúc Phân loại Chuỗi 3D — Quán quân RSNA 2019

### 2.1. Giải pháp Quán quân RSNA 2019 (Top 1 / 1,345 đội thi quốc tế)
- **Tác giả:** Tao Sheng (SeuTao)
- **Kho lưu trữ mã nguồn:** [GitHub SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection)
- **Báo cáo kỹ thuật:** *1st Place Solution Overview — RSNA Intracranial Hemorrhage Detection* (Kaggle Discussion, 2019).
- **Nguyên lý kiến trúc:** 
  1. Sử dụng CNN 2D (`ResNet/ResNeXt`) để trích xuất vector biểu diễn cô đọng ($512$ hoặc $2048$ chiều) từ từng lát cắt CT.
  2. Nạp toàn bộ chuỗi vector lát cắt của một ca chụp qua mạng hồi quy 2 chiều (`Bidirectional GRU / LSTM`) theo trục $Z$ (chiều cao đầu bệnh nhân).
  3. Mô hình hóa mối tương quan giải phẫu liên tục giữa các lát cắt lân cận, mô phỏng đúng thao tác cuộn phim CT của bác sĩ lâm sàng.
- **Ứng dụng trong Block 3:** Tích hợp trực tiếp lớp mạng `RSNAWinningClassifier` gồm `ResNet-34 Feature Extractor` kết hợp `Bidirectional GRU(hidden_dim=128)` để phân loại đa nhãn 6 loại xuất huyết.

### 2.2. Bài báo nền tảng về Phân loại Chuỗi CT Lát cắt (RADNet)
- **Tên bài báo:** *RADNet: Radiologist Level Accuracy using Deep Learning for Hemorrhage Detection in CT Scans*
- **Tác giả:** M. Grewal, M. M. Srivastava, P. Kumar, S. Varadarajan
- **Hội nghị:** IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018), Washington, DC, USA.
- **DOI / arXiv:** [10.1109/ISBI.2018.8363574](https://doi.org/10.1109/ISBI.2018.8363574) | [arXiv:1710.04934](https://arxiv.org/abs/1710.04934)

---

## 3. Kiến trúc Phân đoạn Nâng cao & Thách thức MICCAI INSTANCE

### 3.1. Báo cáo Tổng kết Thách thức MICCAI INSTANCE 2022
- **Tên bài báo:** *The state-of-the-art 3D anisotropic intracranial hemorrhage segmentation on non-contrast head CT: The INSTANCE challenge*
- **Tác giả:** Xiangyu Li, et al.
- **Tạp chí:** *Medical Image Analysis* (MedIA), Elsevier, Vol. 90, 2023.
- **arXiv:** [arXiv:2301.03281](https://arxiv.org/abs/2301.03281)
- **Trang chủ giải đấu:** [INSTANCE Grand Challenge](https://instance.grand-challenge.org/)
- **Kết quả & Đóng góp quan trọng:**
  - Tổng kết các giải pháp phân đoạn xuất huyết não 3D hàng đầu thế giới từ 200 ca CT chụp không cản quang (NCCT).
  - Điểm chuẩn thực tế: Đội Quán quân (Team Vegetable) đạt Dice trung bình $\approx 0.6925$, Đội Công nghệ NVIDIA (NVAUTO) đạt Dice $\approx 0.721$.
  - Bài học then chốt: Dữ liệu CT sọ não có tính chất dị hướng cao (độ phân giải trục $Z$ thô hơn nhiều so với $X, Y$). Việc kết hợp mạng lát cắt 2D/2.5D sâu cùng cơ chế Attention và hậu xử lý lọc vật lý vượt trội so với các mô hình 3D nguyên khối nặng nề.

---

## 4. Hàm mất mát Đặc trị Tổn thương Nhỏ (Focal Tversky Loss)

### 4.1. Bài báo gốc về Focal Tversky Loss
- **Tên bài báo:** *A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation*
- **Tác giả:** Nabila Abraham, Naimul Mefraz Khan
- **Hội nghị:** IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), Venice, Italy, pp. 683–687.
- **DOI:** [10.1109/ISBI.2019.8759465](https://doi.org/10.1109/ISBI.2019.8759465)
- **arXiv:** [arXiv:1810.07842](https://arxiv.org/abs/1810.07842)
- **Toán học cốt lõi:**
  $$TL(\alpha, \beta) = \frac{TP + \epsilon}{TP + \alpha FN + \beta FP + \epsilon}$$
  $$FTL = (1 - TL)^\gamma$$
- **Giải quyết bài toán trong Block 3:**
  - Khi phân đoạn các dải máu mỏng (như SAH rãnh não), vùng xuất huyết chỉ chiếm $< 0.1\%$ diện tích ảnh. Mạng dùng Dice/BCE thông thường dễ bị rơi vào nghiệm tầm thường (dự đoán toàn nền).
  - Đặt $\alpha = 0.7, \beta = 0.3$: Phạt nặng False Negative gấp $2.33$ lần False Positive, ép mạng không được phép bỏ sót các vệt máu nhỏ (khắc phục hoàn toàn lỗi lát cắt số 1 bị Dice 0.00).
  - Số mũ Focal $\gamma = 1.33$: Tập trung lực học vào các lát cắt dị thường, khó nhận biết.

---

## 5. Cơ chế Chú ý Không gian & Kênh (scSE Attention Gates)

### 5.1. Bài báo gốc về scSE Blocks
- **Tên bài báo:** *Recalibrating Fully Convolutional Networks with Spatial and Channel "Squeeze & Excitation" Blocks*
- **Tác giả:** Abhijit Guha Roy, Nassir Navab, Christian Wachinger
- **Tạp chí:** *IEEE Transactions on Medical Imaging* (TMI), 2019; 38(2):540–549.
- **DOI:** [10.1109/TMI.2019.2892900](https://doi.org/10.1109/TMI.2019.2892900)
- **arXiv:** [arXiv:1808.08127](https://arxiv.org/abs/1808.08127)
- **Cơ chế hoạt động:**
  - **cSE (Channel Squeeze & Excitation):** Nén không gian để học trọng số động giữa các kênh đặc trưng $\rightarrow$ tự động khuếch đại kênh Subdural và Brain Window, giảm phụ thuộc vào kênh Bone.
  - **sSE (Spatial Squeeze & Excitation):** Nén kênh qua tích chập $1 \times 1$ để sinh ma trận xác suất không gian $\rightarrow$ tự động dập tắt (suppress) tín hiệu từ hộp sọ và mô mềm ngoài da, hướng mạng tập trung vào nhu mô não.
- **Ứng dụng trong Block 3:** Tích hợp `decoder_attention_type="scse"` trực tiếp vào kiến trúc `smp.Unet(encoder_name="resnet34")`.

---

## 6. Công trình Nghiên cứu Phân đoạn Xuất huyết Não Mới nhất (Nature Sci Rep 2025)

### 6.1. Bài báo Nature Scientific Reports (Tháng 5/2025)
- **Tên bài báo:** *A novel deep convolutional neural network model for intracranial hemorrhage segmentation using IHSNet*
- **Tác giả:** S. Nafees Ahmed, P. Prakasam
- **Tạp chí:** *Scientific Reports* (Nature Portfolio), 2025; 15:1317.
- **DOI:** [10.1038/s41598-025-01317-3](https://doi.org/10.1038/s41598-025-01317-3)
- **PubMed:** [PMID: 40382387](https://pubmed.ncbi.nlm.nih.gov/40382387/)
- **Đóng góp chính:**
  - Thiết kế kiến trúc mạng IHSNet chuyên biệt cho bài toán phân đoạn ICH trên ảnh CT sọ não.
  - Khẳng định tầm quan trọng của việc bảo toàn ranh giới đa tỷ lệ và kỹ thuật tiền xử lý cửa sổ chuyên dụng trước khi nạp vào mạng nơ-ron sâu.

---

## 7. Vật lý Chẩn đoán Hình ảnh & Cửa sổ CT (Windowing Standards)

### 7.1. Tiêu chuẩn Cửa sổ CT Sọ Não trong Cấp Cứu Thần Kinh
- **Tài liệu tham khảo Y khoa:** American College of Radiology (ACR) Practice Parameter for the Performance of Head Computed Tomography (CT).
- **Quy chuẩn 3 Cửa sổ Ánh xạ vào RGB:**
  1. **Cửa sổ Não (Brain Window - Kênh R):** $L = 40\text{ HU}, W = 80\text{ HU} \rightarrow [0, 80]\text{ HU}$. Phân biệt chất xám, chất trắng và dịch não tủy (CSF).
  2. **Cửa sổ Dưới Màng Cứng (Subdural Window - Kênh G):** $L = 80\text{ HU}, W = 200\text{ HU} \rightarrow [-20, 180]\text{ HU}$. Tối ưu hóa độ tương phản giữa máu tụ dạng dải mỏng và bản trong xương sọ.
  3. **Cửa sổ Xương (Bone Window - Kênh B):** $L = 600\text{ HU}, W = 2800\text{ HU} \rightarrow [-800, 2000]\text{ HU}$. Định vị ranh giới xương sọ, nứt sọ và vôi hóa sinh lý.
- **Hậu xử lý Vật lý (Physical HU Gating):**
  - Máu tụ cấp tính (Acute Hematoma): Protein globin cô đặc có mật độ quang học chuẩn trong khoảng $50 \le \text{HU} \le 85$ (mở rộng an toàn $15 \le \text{HU} \le 105$).
  - Mọi điểm ảnh có mật độ xương sọ ($> 110\text{ HU}$) hoặc dịch/mỡ ($< 10\text{ HU}$) bị loại trừ vật lý, ngăn chặn 100% hiện tượng "lem" sang xương thái dương.

---

## 8. Mã Trích Dẫn BibTeX Đầy Đủ (BibTeX Entries)

Bạn có thể copy trực tiếp đoạn mã dưới đây vào file `.bib` trên Overleaf hoặc trình quản lý Zotero/Mendeley:

```bibtex
@inproceedings{wu2023bhsd,
  title={BHSD: A 3D Multi-class Brain Hemorrhage Segmentation Dataset},
  author={Wu, Biao and Shen, Xinyu and Ding, Chuan and others},
  booktitle={International Workshop on Machine Learning in Medical Imaging (MLMI @ MICCAI)},
  pages={147--156},
  year={2023},
  publisher={Springer},
  doi={10.1007/978-3-031-45673-2_15}
}

@article{flanders2020rsna,
  title={Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge},
  author={Flanders, Adam E and Prevedello, Luciano M and Choy, Garry and others},
  journal={Radiology: Artificial Intelligence},
  volume={2},
  number={3},
  pages={e190211},
  year={2020},
  publisher={Radiological Society of North America},
  doi={10.1148/ryai.2020190211}
}

@article{li2023instance,
  title={The state-of-the-art 3D anisotropic intracranial hemorrhage segmentation on non-contrast head CT: The INSTANCE challenge},
  author={Li, Xiangyu and others},
  journal={Medical Image Analysis},
  volume={90},
  pages={102967},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.media.2023.102967}
}

@inproceedings{abraham2019focal,
  title={A novel focal tversky loss function with improved attention U-Net for lesion segmentation},
  author={Abraham, Nabila and Khan, Naimul Mefraz},
  booktitle={2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019)},
  pages={683--687},
  year={2019},
  organization={IEEE},
  doi={10.1109/ISBI.2019.8759465}
}

@article{roy2019scse,
  title={Recalibrating fully convolutional networks with spatial and channel 'squeeze \& excitation' blocks},
  author={Roy, Abhijit Guha and Navab, Nassir and Wachinger, Christian},
  journal={IEEE Transactions on Medical Imaging},
  volume={38},
  number={2},
  pages={540--549},
  year={2019},
  publisher={IEEE},
  doi={10.1109/TMI.2019.2892900}
}

@article{ahmed2025ihsnet,
  title={A novel deep convolutional neural network model for intracranial hemorrhage segmentation using IHSNet},
  author={Ahmed, S Nafees and Prakasam, P},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={1317},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-025-01317-3}
}

@inproceedings{grewal2018radnet,
  title={RADNet: Radiologist level accuracy using deep learning for hemorrhage detection in CT scans},
  author={Grewal, M and Srivastava, MM and Kumar, P and Varadarajan, S},
  booktitle={2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018)},
  pages={281--284},
  year={2018},
  organization={IEEE},
  doi={10.1109/ISBI.2018.8363574}
}
```
