# Artifact ca nộp để kiểm tra hợp đồng

Thư mục này là chỗ duy nhất để đưa artifact thật vào CI. Mỗi ca là một thư mục con:

~~~text
contract-cases/
  <case_id>/
    preprocessing.json   # tùy chọn, theo tập con của contract
    findings.json
    volume.json
    report.json
~~~

Quy ước:

- Tên thư mục là `case_id` giả danh, không chứa tên bệnh nhân, MRN, ngày sinh.
- Chỉ đặt file JSON theo `docs/INTERFACE_CONTRACT.md`. Không đặt `.nii.gz`, `.dcm`, `.nrrd`, ảnh, mask hay dữ liệu bệnh nhân; CI từ chối các định dạng đó.
- Được nộp tập con: ca chỉ có `findings.json` và `volume.json` vẫn được kiểm tra các ràng buộc liên quan.
- Artifact mẫu phải chứa `created_at_utc`, `source_artifact_ids`, `status`, `warnings`, `errors` đúng quy ước chung.

Chạy kiểm tra tại local:

~~~bash
python3 scripts/check_case_artifacts.py --cases-root contract-cases
~~~
