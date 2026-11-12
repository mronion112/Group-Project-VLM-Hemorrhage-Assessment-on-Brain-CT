# CI/CD và quy ước tích hợp

Phạm vi hiện tại: kiểm tra tài liệu, hợp đồng artifact và workflow. Gate cho code Python của từng block sẽ bổ sung khi block có code và môi trường riêng; block tự pin dependency trong môi trường của mình, không dùng chung `requirements-ci.txt`.

## Các gate

| Gate | Mức bắt buộc | Điều kiện chạy | Lệnh local |
|---|---|---|---|
| Cấu trúc tài liệu | Bắt buộc | Mọi push vào `main` và mọi pull request | `python3 scripts/check_structure.py` |
| Liên kết, anchor và ảnh Markdown | Bắt buộc | Mọi push vào `main` và mọi pull request | `python3 scripts/check_links.py` |
| markdownlint | Bắt buộc | Mọi push vào `main` và mọi pull request | `markdownlint-cli2 "**/*.md"` nếu máy có cài |
| Hợp đồng trong docs | Bắt buộc | Mọi push vào `main` và mọi pull request | `python3 scripts/validate_contract.py` |
| Artifact ca nộp | Bắt buộc | Mọi push vào `main` và mọi pull request | `python3 scripts/check_case_artifacts.py --cases-root contract-cases` |
| Artifact ca tham chiếu | Bắt buộc | Mọi push vào `main` và mọi pull request | `python3 scripts/check_case_artifacts.py --case-dir tests/contract/fixtures/case-ok` |
| Test Python | Bắt buộc | Mọi push vào `main` và mọi pull request | `python3 -m pytest tests -q` |
| ruff cho `scripts/` và `tests/` | Bắt buộc | Mọi push vào `main` và mọi pull request | `ruff check scripts tests` |
| actionlint | Bắt buộc | Mọi push vào `main` và mọi pull request | `docker run --rm -v "$PWD:/repo" -w /repo rhysd/actionlint:1.7.12` |
| Tiêu đề PR theo Conventional Commits | Cảnh báo | Mọi pull request | Không có; chỉ chạy trên GitHub |

Không workflow nào lọc `paths`: một gate bắt buộc mà bị lọc theo path sẽ để pull request ở trạng thái chờ check vô hạn. Gate cảnh báo không chặn merge; gate bắt buộc phải xanh trước khi merge vào `main`.

## Phạm vi kiểm tra hợp đồng

`scripts/check_case_artifacts.py` kiểm tra các ràng buộc mà JSON Schema không diễn đạt được:

- `case_id` giống nhau giữa các artifact trong cùng ca.
- `preprocessing.space_id`, `findings.input_space_id`, `volume.space_id` và `space_id` của từng mask trong `findings.masks` phải trùng nhau.
- `volume.source_mask` phải được khai báo trong `findings.masks`; `by_subtype_ml` phải là `null` khi không có mask subtype. Có mask subtype nhưng chỉ báo thể tích tổng vẫn hợp lệ.
- `volume_ml` phải khớp `voxel_count × thể tích voxel vật lý`, dung sai 0.01 mL hoặc 1%.
- `status` không được giảm mức nghiêm trọng theo chiều downstream (`ok` < `partial` < `failed`).
- `partial` phải có `warnings[]`, `failed` phải có `errors[]`; mọi artifact có `created_at_utc` dạng timestamp ISO-8601 UTC và `source_artifact_ids`.
- Mỗi claim trong `report.json` phải trích `*.json#/json-pointer` hợp lệ; giá trị scalar của claim phải khớp node được trích. Evidence dạng khác bị từ chối thay vì bỏ qua.

## Quy ước branch và PR

- `main` là nhánh tích hợp; không push trực tiếp, không force push.
- Nhánh làm việc đặt theo loại thay đổi: `feat/...`, `fix/...`, `docs/...`, `ci/...`, `chore/...`.
- Tiêu đề PR theo Conventional Commits: `type(scope): mô tả`, tối đa 72 ký tự cho phần mô tả.
- Một PR nên gói một thay đổi logic. Thay đổi contract phải cập nhật `docs/INTERFACE_CONTRACT.md`, schema tương ứng và Guide của block liên quan.

## Phiên bản hợp đồng

- `schema_version` trong artifact theo dạng `MAJOR.MINOR`.
- Thêm field tùy chọn: tăng MINOR.
- Đổi ý nghĩa field, đổi đơn vị, đổi enum hoặc thêm field bắt buộc: tăng MAJOR và tag `contract-v2.0` (hiện tại là dòng `contract-v1.x`).
- Tag `contract-v1.x` đánh dấu mốc schema và fixture đã chốt để đối chiếu khi tích hợp. Chưa tạo tag; chỉ tạo khi block đầu tiên bàn giao artifact đạt schema trong `contract-cases/`.

## Artifact ca thật

Đặt artifact cần kiểm tra vào `contract-cases/<case_id>/`. Quy ước chi tiết ở `contract-cases/README.md`. Không đặt dữ liệu ảnh hoặc thông tin bệnh nhân trong thư mục này.

## Required status checks cho branch protection

Các check cần bật cho `main`:

- `docs / docs structure and links`
- `docs / markdownlint`
- `contract / schemas, artifacts and tests`
- `workflow-lint / actionlint`

Check `pr-title / conventional PR title (advisory)` không đưa vào danh sách bắt buộc.

## Checklist branch protection (thủ công trên GitHub)

1. Vào Settings → Branches → thêm rule cho `main`.
2. Bật Require a pull request before merging, tối thiểu 1 approval.
3. Bật Require status checks to pass, chọn các check ở mục trên.
4. Bật Require branches to be up to date before merging.
5. Chặn force push và xóa branch `main`.
6. Kiểm tra CODEOWNERS: mặc định `@mronion112`; dòng owner từng block còn comment, cần thay handle thật rồi mở lại.

## Chạy toàn bộ kiểm tra ở local

~~~bash
python3 scripts/check_structure.py
python3 scripts/check_links.py
python3 scripts/validate_contract.py
python3 scripts/check_case_artifacts.py --cases-root contract-cases
python3 scripts/check_case_artifacts.py --case-dir tests/contract/fixtures/case-ok
python3 -m pytest tests -q
ruff check scripts tests
~~~

Không cần cài dependency mạng cho các lệnh trên nếu máy đã có `jsonschema`, `pytest` và `ruff`. Bản pin trong `requirements-ci.txt` chỉ dùng cho CI.
