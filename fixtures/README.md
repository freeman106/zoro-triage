# 119 데모 입력

신고 문구는 [119 합성 테스트셋](../bench/report119/test119.json)에서 가져왔다. 사진은 [QCRI CrisisMMD v2](https://huggingface.co/datasets/QCRI/CrisisMMD)의 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 자료다. 사진 파일은 설치 환경의 `/ephemeral/crisis_bench/crisismmd/<image_path>`에 있다. 합성 데이터 기준이며 실제 119 성능은 미검증이다.

선별한 고유 사진 100장과 원본 신고 ID를 고정했다. [촬영용 100장 썸네일](../evidence/field-100.jpg)에서 순서와 ID를 볼 수 있다. 벤치마크가 있는 상위 작업공간에서 `python3 demo119/fixtures/build_reports.py --check`로 원본 글·사진 경로를 대조할 수 있다.

| 역할 | ID | 화면 구성 |
|---|---|---|
| 최우선 | test-0062 | 지진 신고와 무너진 건물 사진 |
| 불일치 | test-0019 | 태풍 신고와 건물 화재 사진 |
| 장애 | test-0074 | 어제 태풍 현장에서 사람이 쓰러져 의식이 없었다는 신고. 모의 장애에서 최우선+확인 필요 |
| 규칙 변경 | test-0393 | 거동이 어려운 고령자가 집에 고립됐다는 태풍 신고 |
| 과거 신고·사진 속 위험 | test-0395 | 지난주 화재 보험 문의와 촬영 당시 연기가 보이는 사진 |
