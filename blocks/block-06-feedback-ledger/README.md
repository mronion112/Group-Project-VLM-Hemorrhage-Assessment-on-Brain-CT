# Block 6 — Feedback ledger (placeholder)

**Phụ trách:** Nguyễn Tùng Sơn.  
**CI/CD:** Nguyễn Quang Vũ Hoàng phụ trách, cố vấn và giám sát xuyên suốt.

Để dành cho giai đoạn sau: lưu feedback đã được kiểm duyệt, model/config version và thông tin lineage để hỗ trợ đánh giá/retraining.

Hiện chưa triển khai blockchain, database, API, smart contract, lưu ảnh/PHI, hoặc luồng tự động đưa feedback vào train. Không block nào trong pipeline 1–5 phụ thuộc Block 6.

Trước khi bắt đầu block này, cần chốt schema feedback, người có quyền ghi/duyệt, chính sách dữ liệu và retention, cách gắn feedback với model/dataset version, audit trail, và tiêu chí để feedback được phép tạo training label. Feedback không được biến trực tiếp thành ground truth nếu chưa được xác minh.
