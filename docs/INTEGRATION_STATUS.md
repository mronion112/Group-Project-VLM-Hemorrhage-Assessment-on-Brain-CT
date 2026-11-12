# Trạng thái tích hợp

Bảng theo dõi mức sẵn sàng hợp đồng của từng block. Không ghi tiến độ chưa có artifact thật.

| Block | Trạng thái hợp đồng | Benchmark | Owner sign-off | Ghi chú | Cập nhật lần cuối |
|---|---|---|---|---|---|
| 1 — Dataset và chia tập | Chưa có artifact | N/A | Chưa có | Chưa chỉ định owner; chưa có `dataset_manifest.json` | 2026-11-12 |
| 2 — Preprocessing + Diffusion | Chưa có artifact | N/A | Chưa có | Chưa có `preprocessing.json` cho ca thật | 2026-11-12 |
| 3 — Deep Learning | Chưa có artifact | N/A | Chưa có | Chưa có `findings.json` và mask | 2026-11-12 |
| 4 — Volume | Chưa có artifact | N/A | Chưa có | Chưa có `volume.json` | 2026-11-12 |
| 5 — VLM | Chưa có artifact | N/A | Chưa có | Chưa có `report.json` | 2026-11-12 |
| CI/CD | Schema pass | N/A | Chưa có | Gate tài liệu và hợp đồng đã chạy; fixture `case-ok` và `case-partial` pass | 2026-11-12 |

## Chú giải

- `Chưa có artifact`: block chưa bàn giao artifact nào đúng contract.
- `Schema pass`: artifact mẫu của block (nếu có) validate được với schema và qua `check_case_artifacts`.
- `Lỗi schema`: artifact có nhưng không validate được.
- `N/A`: không áp dụng cho block hoặc chưa có gì để đo.
- `Owner sign-off`: owner kỹ thuật của block đã xác nhận artifact và benchmark.

Bổ sung một dòng trạng thái mới chỉ khi có artifact thật trong `contract-cases/` hoặc trong PR của block, kèm lệnh đã chạy. `contract-cases/<case_id>/` được CI kiểm tra bằng `python3 scripts/check_case_artifacts.py --cases-root contract-cases`.
