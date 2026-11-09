# Đánh giá xuất huyết nội sọ trên CT não bằng DL–VLM

> **Mục tiêu:** nhận một ca CT não 3D, tạo các lát 2D có thứ tự để xử lý, phân tích bằng mô hình chuyên biệt, tính thể tích từ mask và sinh phần giải thích bằng VLM dựa trên bằng chứng.

> Đây là khung nghiên cứu và hợp đồng tích hợp cho các nhóm phát triển độc lập. Repo hiện chứa tài liệu thiết kế; các chỉ số thời gian và chất lượng của chính dự án vẫn cần được đo trên dữ liệu và phần cứng đã chốt.

## Pipeline tổng quát

~~~mermaid
flowchart LR
    A[Block 1<br/>Dataset curation + patient-level splits] -->|dataset manifest + split lists + source references| B[Block 2<br/>HU, geometry, windows, diffusion, QC]
    B -->|preprocessed volume + metadata| C[Block 3<br/>presence, subtype, mask]
    C -->|findings.json + mask(s)| D[Block 4<br/>mask × voxel spacing]
    C --> E[Block 5<br/>VLM evidence-grounded report]
    D -->|volume.json| E
    B -->|QC + preprocessing metadata| E
    E -->|report.json + report text| F[Người đánh giá / ứng dụng]
    F -. phản hồi và model version trong tương lai .-> G[Block 6<br/>Feedback ledger — placeholder]
~~~

### Ranh giới cố định giữa các block

| Block | Input | Output bắt buộc | Không thuộc trách nhiệm |
|---|---|---|---|
| [1. Dataset và chia tập](blocks/block-01-dataset-split/Guide.md) | Danh sách dataset/ca, metadata, nhãn và nguồn DICOM/NIfTI | dataset_manifest.json và split manifests theo patient; không sửa/nhân bản ảnh | Preprocessing, inference hoặc gán nhãn mới |
| [2. Preprocessing + Diffusion](blocks/block-02-preprocessing/Guide.md) | Split manifest và source reference của Block 1 | volume_hu.nii.gz, map lát 2D có thứ tự, preprocessing.json, QC và nhánh volume_diffusion.nii.gz có phiên bản | Tự kết luận xuất huyết; thay ảnh gốc hoặc phát hành diffusion trước benchmark |
| [3. Deep Learning](blocks/block-03-deep-learning/Guide.md) | Output Block 2 | findings.json, mask ICH; mask subtype chỉ khi mô hình/nhãn hỗ trợ; heatmap là tùy chọn | Tính thể tích mL hoặc viết báo cáo ngôn ngữ |
| [4. Volume](blocks/block-04-volume/Guide.md) | Mask từ Block 3 và spacing/affine cùng không gian | volume.json với voxel count, mL, phương pháp và cờ chất lượng | Phân loại subtype từ ảnh hoặc tự sửa mask |
| [5. VLM](blocks/block-05-vlm/Guide.md) | QC Block 2 + findings/mask-derived location Block 3 + thể tích Block 4 | report.json và text giải thích, mỗi claim liên kết bằng chứng | Tự suy ra chẩn đoán/định vị/thể tích chưa có trong bằng chứng |

Hợp đồng dữ liệu chung chi tiết ở [docs/INTERFACE_CONTRACT.md](docs/INTERFACE_CONTRACT.md). Nếu thiếu input bắt buộc, block trả partial/failed cùng mã lỗi; không tự tạo giá trị thay thế. Artifact theo-ca gắn case_id; artifact cấp dataset gắn dataset_id/release. Cả hai có schema_version, còn model_version và space_id được ghi ở nơi tương ứng.

## Phân công

| Phần | Người phụ trách | Phạm vi |
|---|---|---|
| Block 1 — Dataset và chia tập | Chưa chỉ định | Audit dataset/nhãn, patient-level split, manifest và tránh leakage |
| Block 2 — Preprocessing + Diffusion | Nguyễn Hoàng Anh | HU, geometry, windowing/QC; xây và đánh giá diffusion denoising |
| Block 3 — Deep Learning | Nguyễn Đình Hữu Minh, Vũ Minh Hoàng | Presence, subtype, detection/localization và segmentation |
| Block 4 — Volume | Nguyễn Văn Đức, Nguyễn Hoàng Anh | Tính mL từ mask/spacing và kiểm chứng sai số |
| Block 5 — VLM | Nguyễn Hoàng Minh | Sinh giải thích từ evidence có cấu trúc và đánh giá factuality |
| Block 6 — Feedback ledger | Nguyễn Tùng Sơn | Thiết kế phần lưu feedback/model version; hiện để placeholder |
| CI/CD | Nguyễn Quang Vũ Hoàng | Điều phối CI/CD, tư vấn quy trình và theo dõi tích hợp/benchmark giữa các block |

Chi tiết phạm vi và bàn giao theo người phụ trách: [docs/OWNERSHIP.md](docs/OWNERSHIP.md). Phân công CI/CD không làm thay đổi owner kỹ thuật của từng block.

## Luồng dữ liệu và chuẩn CT

- Một ca 3D là volume có thứ tự; lát dày 3–5 mm thường có khoảng 20–50 lát, còn 0,5–1,25 mm có thể có 150–300 lát hoặc hơn. Đây chỉ là ước lượng; số lát thực tế lấy từ từng ca và metadata DICOM.
- Giữ volume 3D làm nguồn chuẩn. Có thể tạo lát 2D theo trục axial cho mô hình 2D nhưng phải giữ slice_index và ánh xạ về đúng voxel/không gian 3D.
- Đổi stored pixel value sang HU bằng RescaleSlope/RescaleIntercept khi metadata nguồn cho phép. Không giả định mọi PNG/NIfTI đã giữ HU, và không áp một ngưỡng cắt HU chung [-1000, 4000] cho mọi nguồn. Giữ HU gốc; clipping/windowing là cấu hình có phiên bản.
- Giữ orientation, affine và voxel spacing. Mọi mask phải cùng shape/affine/space_id với volume mà nó phủ, hoặc phải được biến đổi tường minh về không gian đó trước Block 4.

## Bốn nhóm bằng chứng đưa vào VLM

1. **Có/không/không chắc có xuất huyết** và confidence/quality flag.
2. **Subtype đa nhãn:** EDH, SDH, SAH, IPH, IVH; có thể đồng thời nhiều loại.
3. **Vị trí:** bên/vùng giải phẫu và extent suy ra từ mask đã kiểm tra; heatmap không thay mask.
4. **Thể tích:** mL do Block 4 tính, kèm mask nguồn và trạng thái hợp lệ.

VLM mặc định chỉ nhận bằng chứng có cấu trúc và metadata QC cần thiết. Đưa ảnh CT trực tiếp cho VLM là một nhánh ablation riêng, không thay đổi hợp đồng đầu vào mặc định.

Tối ưu tùy chọn: Block 3 có thể dùng ROI/heatmap để chọn candidate trước mô hình nặng, nhưng phải giữ full-volume fallback và đo sensitivity theo subtype/tổn thương nhỏ cùng thời gian tiết kiệm. VLM orchestration có thể gọi công cụ Block 3/4 với output theo contract; VLM không tự đoán subtype hay tính thể tích.

## Dataset: vai trò và điều cần xác minh

Số liệu bên dưới là quy mô được mô tả bởi nguồn dataset, không phải số mẫu đã tải hoặc đã lọc trong repo. Cần audit manifest, license/DUA, dữ liệu trùng bệnh nhân, modality, HU, spacing và label trước khi chia train/validation/test.

| Dataset | Quy mô theo nguồn | Nhãn/dạng hữu ích | Vai trò đề xuất và giới hạn |
|---|---:|---|---|
| [M3D-Cap](https://github.com/BAAI-DCAI/M3D) | 120.092 ảnh 3D và 42.496 text trong bảng phát hành M3D | Ghép ảnh-text đa cơ quan | Tiền huấn luyện alignment/report nếu tìm được tập con CT sọ phù hợp; không xem 120.092 là số ca ICH hoặc CT sọ đã xác nhận. |
| [M3D-VQA](https://github.com/BAAI-DCAI/M3D) | 96.170 ảnh 3D, 509.755 Q&A theo bảng phát hành | Câu hỏi–trả lời 3D | Đây là nhánh M3D có Q&A; cần lọc câu hỏi liên quan ICH và kiểm tra đáp án trước khi dùng làm ground truth. |
| [M3D-Seg](https://github.com/BAAI-DCAI/M3D) | 5.772 ảnh 3D và mask/text theo bảng phát hành | Ảnh 3D, mô tả category, segmentation mask | Candidate segmentation/multimodal; quy mô phát hành không chứng minh có 1.000 ca CT sọ xuất huyết. Audit anatomy, modality và label thực tế. |
| [RSNA ICH](https://atlas.rsna.org/cards/bd41bbe2-f161-4c19-841c-85d8b645e306) | Hơn 25.000 CT examinations; 874.035 ảnh 2D | Nhãn ảnh cho ICH và 5 subtype | Phân loại ảnh/lát và tổng hợp theo ca. Bản gốc không cung cấp mask pixel hay bounding box tổng quát. |
| [CQ500](https://www.kaggle.com/datasets/crawford/qureai-headct) | 491 examinations; số lát tùy series/reconstruction được chọn | Đánh giá chẩn đoán ở mức ca | Classification và external evaluation. Không có mask ICH tổng quát; không coi số lát cố định là thuộc tính dataset. |
| [PhysioNet CT-ICH](https://physionet.org/content/ct-ich/1.3.1/) | Thu thập 82 ca; 75 volume NIfTI có trong bản phát hành 1.3.1; khoảng 30 lát/ca | Mask vùng ICH và nhãn subtype theo lát | Segmentation/classification thử nghiệm. Dữ liệu chấn thương, lát 5 mm, cohort nhỏ; mask không mặc nhiên là mask 5 subtype riêng. Cần đăng ký và chấp thuận DUA. |
| [INSTANCE 2022](https://arxiv.org/abs/2301.03281) | 200 volume: 100 train có nhãn, 30 validation và 70 test của challenge | Mask ICH 3D | Phù hợp benchmark segmentation/volume; mask benchmark không tự động là mask phân lớp 5 subtype. Xác minh điều kiện tải và split hiện có. |
| [BHSD](https://arxiv.org/abs/2308.11298) | Bài báo mô tả 192 volume có mask pixel-level và 2.200 volume có nhãn lát cho 5 nhóm ICH | Segmentation đa lớp + slice labels | Candidate quan trọng cho subtype segmentation; kiểm tra phiên bản, quyền truy cập, split và annotation trước khi dùng. |

**Lưu ý đối chiếu report:** M3D-Cap là image-text; M3D-VQA mới là nhánh Q&A; M3D-Seg là nhánh có mask. RSNA có hơn 25 nghìn **ca**, không phải 25 nghìn ảnh. PhysioNet mô tả 82 ca thu thập nhưng chỉ 75 NIfTI được công khai ở bản 1.3.1. Các con số này cần được xác nhận lại từ manifest đúng phiên bản trước khi báo cáo kết quả.

## Chấm điểm hệ thống

- **Block 3:** sensitivity/specificity, AUROC/AUPRC và macro-F1 theo từng subtype; segmentation Dice/HD95 và lesion-wise sensitivity; báo cáo riêng tổn thương nhỏ và dữ liệu ngoài miền.
- **Block 4:** MAE/RMSE mL, sai lệch thể tích, tương quan/agreement với reference; phân tích riêng theo subtype và spacing.
- **Block 5:** độ đúng từng trường (presence/subtype/location/volume), tỷ lệ claim có bằng chứng, tỷ lệ claim không được hỗ trợ, mâu thuẫn, thiếu trường và tính ổn định qua lần chạy. Text similarity chỉ là chỉ số phụ.
- **Toàn pipeline:** patient-level split, so sánh VLM trực tiếp (nếu có), VLM + nhãn phân loại, VLM + đủ bằng chứng cấu trúc và oracle evidence (chỉ upper bound). Không chia ngẫu nhiên theo lát giữa các split.
- **Thời gian:** đo riêng đọc/convert, preprocessing, inference DL, tính thể tích, sinh VLM và end-to-end; ghi CPU/GPU, RAM/VRAM, batch, độ phân giải, số lát và warm-up. Repo chưa có số đo runtime của mình; xem [Report từng block](blocks/).

## Cấu trúc repo

~~~text
blocks/
  block-01-dataset-split/  Guide.md  Report.md
  block-02-preprocessing/  Guide.md  Report.md
  block-03-deep-learning/  Guide.md  Report.md
  block-04-volume/         Guide.md  Report.md
  block-05-vlm/            Guide.md  Report.md
  block-06-feedback-ledger/README.md  # placeholder, chưa triển khai
docs/
  INTERFACE_CONTRACT.md
  OWNERSHIP.md
~~~

Block 6 được giữ làm chỗ dành trước cho feedback/model version và kế hoạch retraining; chưa có blockchain, lưu trữ dữ liệu bệnh nhân, hoặc dịch vụ ghi dữ liệu.

