# Block 5 — VLM sinh text dựa trên bằng chứng

**Phụ trách:** Nguyễn Hoàng Minh.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

## Mục tiêu

Nhận bằng chứng đã có cấu trúc từ các block trước và diễn đạt thành text mạch lạc, truy vết được. VLM không thay thế classifier, segmenter hoặc thuật toán volume.

## Input mặc định

- preprocessing.json: QC, intensity status, model-input caveat; không cần pixel volume cho đường mặc định.
- findings.json: presence, subtype scores/labels, model version, warnings.
- mask_ich.nii.gz và localization metadata nếu đã tạo; chỉ dùng vùng có mask hợp lệ.
- volume.json: volume mL, mask/spacing source, quality flags.
- Prompt/template version, output language và schema version.

Các input phải cùng case_id; findings và masks phải cùng space_id; volume.json phải tham chiếu đúng mask đó. Nếu thiếu một thành phần, giữ kết quả partial và ghi evidence còn thiếu.

## Bốn trường evidence và output

1. Presence: positive, negative hoặc uncertain.
2. Subtype: danh sách EDH/SDH/SAH/IPH/IVH cùng score/threshold, cho phép đồng mắc.
3. Location: bên/vùng/extent chỉ khi xác định từ mask và mapping giải phẫu có phiên bản.
4. Volume: mL do Block 4 tính; không tự tính lại từ ảnh hoặc mask text.

Output report.json gồm report_text, claims chuẩn hóa, evidence reference cho từng claim, warnings/missing_evidence, model/prompt version. Text không được chứa claim mâu thuẫn với structured claims.

## Luồng xử lý

1. Validate case_id, status, schema, model version và space_id của mọi artifact.
2. Chuyển findings/volume thành prompt context có cấu trúc; truyền riêng unknown, uncertain, quality conflict và missing mask.
3. Sinh báo cáo ngắn theo template và ngôn ngữ đã chọn. Không khuyến khích VLM kể lại độ tin cậy như chắc chắn.
4. Parse/validate output thành report.json; nếu VLM trả sai schema thì retry theo policy giới hạn hoặc trả failed.
5. Gắn evidence pointer cho từng claim. Claim không trỏ được về input phải bỏ hoặc đưa vào unsupported_claims, không phát hành như kết luận.

Nếu orchestration triển khai tool call, các tool chỉ gọi chức năng đã định nghĩa của Block 3 (classification/segmentation) và Block 4 (volume); trả output đúng JSON contract ở trên. Tool call là cách điều phối, không trao quyền cho VLM tự suy subtype hoặc volume. Mỗi lần gọi cần ghi tool/version/input hash và kết quả để có thể tái lập.

## Chính sách

- VLM không được suy ra subtype, bên, vùng giải phẫu hoặc thể tích mà input không chứng minh.
- Nếu Block 3 uncertain, thể hiện uncertain; nếu volume invalid/missing, không có số mL.
- Heatmap không phải segmentation. Có thể mô tả candidate region riêng nếu protocol cho phép nhưng không chuyển nó thành volume.
- Ảnh CT trực tiếp vào VLM chỉ là ablation riêng. Default contract nhận structured evidence để kiểm tra rõ đóng góp của từng block.
- Oracle evidence chỉ dùng làm upper-bound trong nghiên cứu, không dùng ở pipeline inference.

## Liên quan

- [Hợp đồng input/output chung](../../docs/INTERFACE_CONTRACT.md)
- [Report Block 5](Report.md)
- [Phân công và bàn giao](../../docs/OWNERSHIP.md)

