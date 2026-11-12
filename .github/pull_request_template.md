## Tóm tắt

-

## Checklist

- [ ] Artifact tuân thủ `docs/INTERFACE_CONTRACT.md`: đúng field, đúng `schema_version`, không đổi ngầm ý nghĩa field
- [ ] `python3 scripts/validate_contract.py` và `python3 -m pytest tests -q` chạy pass ở local
- [ ] Số benchmark ghi kèm cấu hình, phần cứng và tập dữ liệu dùng để đo
- [ ] Không có dữ liệu bệnh nhân, ảnh y tế hoặc file `.nii*`, `.dcm`, `.nrrd` trong PR
- [ ] Không có đường dẫn nội bộ, hostname, IP, token hay URL nội bộ trong code, tài liệu và log
- [ ] Guide/Report của block liên quan đã cập nhật nếu hành vi hoặc contract thay đổi

## Ghi chú cho reviewer

-
