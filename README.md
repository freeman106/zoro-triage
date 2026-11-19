# 119 사진+글 신고 1차 분류 에이전트

Trillion Labs AI Agent Hackathon 2026 · Track 1, Zoro-1 Arena · Two Drone · 발표 3번

신고 사진과 글이 도착하면 접수요원이 상황판에서 먼저 보고, Zoro-1 판정이 끝나면 근거·확률과 함께 순위가 갱신된다. 운영자가 현장 우선순위 조건을 한 줄 추가하면 기존 신고도 재판정한다. 사람의 수정은 원래 모델 판단과 함께 학습용 라벨로 저장한다. **훈련용 데모. 사진+글 1차 분류는 합성 신고로 평가했으며 실제 119 접수 현장의 성능은 미검증이다.**

## 실행

Python 3.10 이상, Pillow, 로컬 또는 SSH로 접근 가능한 Zoro-1 서버가 필요하다. 인스턴스에서는 기존 `~/venvs/bench/bin/python`을 사용할 수 있다.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# Optional when the CrisisMMD directory is already available.
pip install huggingface_hub
python fixtures/fetch_images.py "$PWD/runtime/crisismmd"
CRISIS_ROOT="$PWD/runtime/crisismmd" ZORO_URL=http://127.0.0.1:30001 python server.py
```

기본 주소는 `127.0.0.1:8119`, 사진 루트는 `/ephemeral/crisis_bench/crisismmd`다. `PORT`, `DEMO_HOME`, `DEMO_WORKERS`로 각각 포트, 영속 디렉터리, 동시 판정 건수를 설정한다. 현재 데모는 동시 판정 4건으로 운영한다. 기존 인스턴스에서 6건으로 늘리면 폭주 완료 시간은 줄지만 건당 지연이 길어져 4건을 선택했다. 기본 신고는 글·사진 두 요청을 병렬로 처리한다.

새 인스턴스 `l40s-zoro-119`에서 기존 접수 내역을 이어 쓰려면 다음 명령을 실행한다.

```sh
cd ~/demo119 && test -s runtime-field-live/access.json && test -s runtime-field-live/inbox.sqlite && \
  DEMO_HOME="$PWD/runtime-field-live" DEMO_WORKERS=4 ~/venvs/bench/bin/python -u server.py
```

맥북에서 다음 SSH 연결을 열면 상황판은 `127.0.0.1:8119/board`, 폰 신고 페이지는 맥북 Tailscale 주소의 8120번 포트로 접속한다. 기존 신고 QR과 접수요원 키를 그대로 쓴다. Zoro 호출은 인스턴스 내부에서 처리한다.

```sh
phone_ip=$(tailscale ip -4) && ssh -NT -S none -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8119:127.0.0.1:8119 -L "${phone_ip}:8120:127.0.0.1:8119" l40s-zoro-119
```

최초 실행 시 `DEMO_HOME/access.json`에 서로 다른 `report`·`admin` 키가 생성된다. 신고 화면은 `/#token=<report>`, 상황판은 `/board#token=<admin>`이다. 토큰은 URL fragment로 받아 브라우저 주소에서 지운다. 실제 사진·자유 입력·토큰은 로컬 실행 디렉터리에 보관한다. 제출용 증빙에는 합성 사례만 포함한다.

## 새 GPU 인스턴스에서 처음부터 실행 (팀원용)

Brev L40S(AWS g6e, Ubuntu 22.04, CUDA 13.2 이미지) 기준이다. 모델 가중치, 사진, 토큰은 저장소에 없다.

1. `git clone` 또는 `git pull` 후 `bash server/setup_instance.sh` (5~8분). venv 두 개, trida-stack 고정 커밋과 패치, `~/zoro-serve.sh`, systemd `zoro` 유닛을 만든다. 마지막 줄 `STAGE1_DONE: 0.5.23 2.14.1+cu130 True`를 확인한다.
2. Hugging Face에서 `trillionlabs/Zoro-1` 약관에 동의한 계정으로 `~/venvs/zoro/bin/hf auth login`을 실행하고, `~/venvs/zoro/bin/hf download trillionlabs/Zoro-1`로 모델을 받는다.
3. `sudo systemctl enable --now zoro` 후 `tail -F ~/zoro-serve.log`. 첫 기동은 약 3분 걸린다. `curl localhost:30001/health`로 확인한다.
4. 데모 사진: `~/venvs/bench/bin/python fixtures/fetch_images.py /ephemeral/crisis_bench/crisismmd` (폭주·예시에 쓰는 사진만 받는다).
5. 데모 실행: `CRISIS_ROOT=/ephemeral/crisis_bench/crisismmd ZORO_URL=http://127.0.0.1:30001 DEMO_WORKERS=4 ~/venvs/bench/bin/python server.py`. 접근 키는 `DEMO_HOME/access.json`에 새로 생성된다.
6. 판정 동일성 확인(선택): `server/zoro_equiv.py`의 기준값 파일은 저장소에 없다. 같은 서버 버전이면 데모 테스트 `python -m unittest discover -s tests`와 상황판 확인으로 충분하다.

폰 연결(Tailscale·SSH 중계)은 각자 맥북에서 설정한다. AI Hub 119 데이터는 재배포가 금지돼 있어 저장소에 넣지 않는다.

## 발표 순서

발표용 무음 영상은 `slideA.mp4`, `slideB.mp4`, `slideC.mp4` 순서로 재생한다. 영상 파일은 저장소에 포함하지 않는다.

1. **slideA · 폰 제보와 불일치 확인.** iPhone에서 보낸 화재 신고가 1.16초 만에 상황판에 나타나고, 제출 후 1.23초에 최우선으로 분류된다. 이어서 글은 화재를 주장하지만 사진에는 태풍에 쓰러진 전봇대가 보이는 신고에 ‘사람 확인 필요’가 붙는다.
2. **slideB · 규칙 변경과 100건 폭주.** 예시 신고 6건에 `고립된 고령자·어린이는 최우선`을 추가한다. 고립된 노부모 신고가 1.6초 만에 긴급 4위에서 최우선 1위로 올라간다. 같은 규칙으로 신고 100건을 추가하면 30.8초에 정렬을 마친다(3.25건/초).
3. **slideC · 모의 장애 복구.** 모의 장애 중 호흡이 멈춘 사람을 발견했다는 신고를 폰으로 보낸다. 같은 카드에 ‘최우선’, ‘폴백 판정’, ‘사람 확인 필요’가 뜬다. 연결이 복구되면 최초 판정 5초 후 그 접수 번호로 자동 재판정이 시작되고, 성공하면 ‘재판정 완료’가 표시된다.

## 판정 및 측정

- 백엔드는 전송 직전에 원본 면적이 200,704픽셀보다 큰 사진을 Zoro의 32픽셀 격자와 BICUBIC 필터로 줄이고 JPEG 품질 95로 인코딩한다. 가로세로 비율은 격자가 허용하는 한 유지한다. 작은 이미지는 수신 바이트 그대로 쓴다. 업로드의 미리보기는 별도로 만든다.
- 일반 폴백은 게시 5초 후 백그라운드 재판정 대상이 된다. 실패하면 최초 폴백을 유지하고, 성공하면 모델 판정만 갱신한다. 검수자가 저장한 등급은 보존한다. 새 규칙 판정이 폴백이 된 경우에도 그 판정에 대해 한 번만 재시도한다. 모의 장애에서 복구 토글이 꺼져 있으면 복구 대기 상태를 유지한다. 폭주 지표의 폴백 수는 최초 판정을 기준으로 한다.
- POST `/v1/systemone`에 글 요청에 인명 상태·시점·신고 재난 종류를, 사진 요청에 사진 피해·현장 사진 여부·시각적 재난 종류·촬영 시점에 진행 중인 위험을 묶는다. 생명 위협과 진행 중 위험의 확률을 카드와 로그에 표시하고 긴급도에 반영한다. 두 선택 확률이 모두 0.50 이상이고 화재 대 비화재가 갈릴 때 불일치를 표시한다. 사진 유형 불명·글 유형 기타·지진 대 태풍은 경고를 보류한다. 기본 시연 규칙과 `글:`로 시작하는 조건은 글 요청에, `사진:`으로 시작하는 조건은 사진 요청에 포함한다. 그 밖의 자유 입력 규칙은 두 증거를 함께 쓸 수 있도록 사진+글 요청 한 번에 모아 처리한다.
- 글에서 과거로 판단한 사건은 사진에 촬영 당시 진행 위험이 보이면 긴급+확인 필요, 그렇지 않으면 비긴급이다. 현재 신고는 생명 위협, 또는 고립과 사진 속 진행 위험이 함께 있으면 최우선이다. 고립 단독·경상은 긴급이고, 위험한 사람 언급이 없으면 사진 속 진행 위험·심각한 피해 중 하나라도 있을 때 긴급, 모두 없을 때 일반이다. 추가 운영 조건에 해당하는 현재 신고는 최우선으로 올린다. 인명 상태는 생명 위협 확률이 0.34 이상이면 생명 위협, 그 아래면 고립·경상·없음 중 확률이 가장 높은 상태를 쓴다. 시점·사진 속 진행 위험 기준값은 0.50이다. 물적 피해는 심각·경미·거의 없음 중 하나를 묻고 P(심각) ≥ 0.60이면 심각으로 판정한다.
- 진행 여부·위험 인원·사진 피해 확률이 0.35~0.65이면 해당 판단 하나만 뒤집어 같은 정책으로 등급을 재계산한다. 등급이 달라지면 ‘판단 불확실’ 배지를 붙이고, 같으면 사유에만 ‘참고’로 남긴다. 재난 종류 불일치, 현장 사진 ‘예’ 확률 0.50 미만, 폴백, 지난 사건과 사진 속 진행 위험이 함께 나온 경우는 항상 확인 대상으로 올린다. **화재 대 비화재, 합성 신고 400건**에서 불일치 판정은 정밀도 100%(82/82), 재현율 84.5%(82/97)였다. 현장성 질문은 35/400건을 비현장으로 표시했다. [촬영용 사진 100장](evidence/field-100.jpg) · [비현장 표시 예시](evidence/non-field-examples.jpg)
- 오류나 2초 제한 초과 시 글 키워드로 임시 분류한다. 명시적 생명 위협은 과거 표현과 함께 있어도 최우선+확인 필요이고, 모델 확률은 미측정으로 표시한다.

카드의 **판정 지연**은 해당 신고의 전처리·모델 호출·판정에 걸린 시간이다. 폭주와 겹친 신규 신고에는 접수부터 최초 판정까지의 **대기 포함** 시간도 표시한다. 재판정 후에는 현재 판정 지연과 최초 대기 포함 시간을 각각 확인할 수 있다. 폭주 패널의 **처리량**은 완료 건수 ÷ 총 경과시간(초), **평균 처리 간격**은 총 경과시간 ÷ 완료 건수다. workers 4로 합성 100건을 처리한 한 회차에서 총 27.169초, 3.68건/초, 평균 간격 272ms였고 개별 판정 지연 중앙값은 1,130ms였다. [건별 실측](evidence/damage-review/after.json)

### 실제 119 통화 전사

AI Hub 검증 자료에서 긴급도 상·중·하를 각각 300건씩 뽑았다. 같은 **통화 전사 900건**에서 신고 라벨 없이 실행한 Zoro와, 별도 학습 통화로 훈련한 글자 2~4그램 TF-IDF + 로지스틱 회귀의 정확도를 비교했다. 구급 여부는 대분류 결과에서 구급과 나머지를 구분해 채점했다. [평가 설계 및 결과](bench/BENCHMARK_LOG.md#19-실제-119-신고-통화-전사-평가-ai-hub-2026-11-20-claude-세션)

| 과제 | Zoro 0건 | TF-IDF 100건 | 1,000건 | 10,000건 | 127,178건 |
|---|---:|---:|---:|---:|---:|
| 구급 여부 | 94.7% | 80.8% | 91.1% | 95.1% | 95.7% |
| 대분류 4종(구급·구조·화재·기타) | 89.4% | 76.8% | 84.8% | 91.1% | 93.0% |
| 긴급도 상·중·하 | 48.7% | 44.8% | 49.0% | 51.9% | 55.2% |

### 신고 글과 사진이 달라질 때

- **사진 피해.** 시나리오에 쓰지 않은 CrisisMMD 사진 193장에서 심각 피해 여부를 평가했다. Zoro는 **77.2%**(149/193)였다. 같은 사진에서 CLIP ViT-L/14의 예측을 심각·비심각으로 묶으면 **80.8%**(156/193), CLIP ViT-B/32의 이진 분류기는 **81.3%**(157/193)였다. 두 CLIP 분류기는 피해 라벨이 있는 사진 2,468장으로 학습했다. Zoro 질문 문구는 시나리오 사진으로 다듬고 P(심각) 기준값 0.60은 학습 사진 1,087장에서 정했다. 기존에 기록한 CLIP-L의 **71.1%**(376/529)는 전체 테스트 사진 529장의 **피해 3종** 분류 정확도다. [193장 평가셋](bench/claude_eval/severity_holdout.py) · [Zoro 채점](evidence/damage-review/image-severity.json) · [CLIP-L 실험](bench/BENCHMARK_LOG.md#3-crisismmd-결과-hf-test-같은-사진같은-라벨) · [CLIP-B 가중치 해시](bench/report119/results/noisy-shift/weights-manifest.json)
- **흐린 사진.** PR #2에서는 합성 신고 100건의 사진(고유 사진 98장)에 흐림을 적용했다. 심각 피해 여부 정확도는 Zoro 개발 버전 65%, 신고 1,500건과 사진 라벨 2,468장으로 학습한 KcELECTRA+CLIP 73%였다. [PR #2 비교 결과](https://github.com/freeman106/zoro-119-triage/blob/b026a35ffe5d0c4a5f062f18a71fd9ce2fe8ae56/bench/BENCHMARK_LOG.md#22-사진-노이즈-평가의-대조-모델-비교-2026-11-20)

**지저분한 신고 글.** 합성 신고 400건의 사진과 정답을 고정하고 글만 오타·사투리·받아쓰기 오류 등으로 다시 썼다. 아래는 사진과 글의 재난 종류가 일치한 **303건**에서 이전 정책의 4단계 긴급도 정답으로 채점한 결과다. KcELECTRA+CLIP은 원문 신고와 별도 피해 라벨이 있는 학습 사진 2,468장을 사용했다. [Zoro 결과](bench/BENCHMARK_LOG.md#18-zoro-쪽-신고-글-분포-이동과-추론-모드-재판단-2026-11-20-claude-세션) · [비교 모델 결과](bench/report119/results/noisy-shift/)

| 모델 | 깨끗한 학습 신고 | 원문 정확도 | 잡음본 정확도 |
|---|---:|---:|---:|
| Zoro | 0건 | 71.6% | 71.9% |
| KcELECTRA+CLIP+규칙 | 50건 | 67.3% | 65.0% |
| KcELECTRA+CLIP+규칙 | 200건 | 73.9% | 68.0% |
| KcELECTRA+CLIP+규칙 | 1,500건 | 77.9% | 78.2% |

## 검증과 저장

`python -m unittest discover -s tests`로 단위 검사를 실행한다. `proof.py`는 동일 호스트에서 비어 있는 별도의 데모 인스턴스에 실제 요청을 보내며, 환경변수 `ADMIN_TOKEN`·`REPORT_TOKEN` 및 선택적 URL 인자를 사용한다. 이 검사는 106개의 합성 신고와 검수 라벨, 운영 규칙을 만든다.

완료 카드는 접수요원의 유효 등급, 같은 등급 안에서 미검수 확인 필요 여부, 오래된 접수 시각 순으로 정렬한다. 분석 대기 카드는 별도 상단 띠에 표시한다. SQLite에 접수·판정·라벨을 기록한다. JSONL은 `DEMO_HOME/logs/`에 저장한다. `input.image`는 `input.image_root`(CRISIS_ROOT 또는 DEMO_HOME)를 기준으로 한 상대 경로다. `scenario-top.jsonl`, `scenario-mismatch.jsonl`, `scenario-outage.jsonl`은 세 시나리오의 로그다. `labels.jsonl`은 입력(글·사진 참조), 당시 모델 판단·근거, 사람이 선택한 등급을 담는다. [실행 증빙](evidence/)에서 합성 사례 로그와 통합 검증 수치를 확인할 수 있다.

## 저장소 구성

- 루트: 데모 앱(`server.py`, `store.py`, `decision.py`, `static/`), 단위 테스트(`tests/`), 실행 증빙(`evidence/`)
- `bench/`: 벤치마크 기록(`BENCHMARK_LOG.md`)과 스크립트·결과. D-Fire, CrisisMMD, 119 합성 신고 400건 비교
- `server/`: Zoro-1 서빙 설정(systemd, 실행 스크립트), 전처리 병렬화 패치(`zoro-patch.diff`), 판정 동일성·동시성 측정 스크립트, 새 인스턴스 설치 스크립트
- `docs/`: 국내·해외 도입 근거 조사, 독립 검증 요청서, Zoro 서버 사용 안내

## 라이선스

코드는 [Apache-2.0](LICENSE)이다. 사진은 [QCRI CrisisMMD v2](https://huggingface.co/datasets/QCRI/CrisisMMD)에서 가져오며 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 조건을 따른다. 원본 사진은 이 저장소에 포함하지 않는다. 발표용 미리보기는 크기를 줄인 사진이다.

현재 버전은 `person-choice-v1-damage-choice-v1-review-v2`다. 기존 두 noul을 person_state choice 하나로 대체해 기본 경로는 글 3문항·사진 4문항, 두 요청이다. train 700건에서 생명 위협 기준을 0.34로 고정하고 합성 test 400건을 한 번 채점했다. 생명 위협 재현율 112/117(95.7%), 최우선 비율 153/400(38.25%)로 두 채택 조건을 충족했다. 사진 진행 위험에는 독립 정답이 없으며, 합성 데이터의 경상 정답 사례는 0건이다. [인명 상태 정책 검증](evidence/resize-retry-run/) · [현재 버전 실서버 검증](evidence/damage-review/proof.json).
