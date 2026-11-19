# Zoro-1 서버 사용 가이드 (Two Drone)

> 현재 119 데모 인스턴스는 `l40s-zoro-119`다. 백엔드 기동과 맥북·폰 중계 명령은 [데모 실행 안내](../README.md#실행)를 참조한다. 아래 CLI와 호스트 명령은 이전 인스턴스의 실험 기록이다.

팀 Brev 인스턴스에 Zoro-1 서버가 떠 있다. 이 문서는 팀원 맥북(또는 그 위의 코딩 에이전트)에서 이 서버를 호출하는 방법을 설명한다. 모델을 새로 받거나 설치할 필요는 없다. 이미 떠 있는 서버 하나를 같이 쓴다.

## 빠른 시작

Brev CLI 로그인(`brev login`)과 인스턴스 SSH 권한이 있으면 맥북 터미널에서 두 줄이면 된다.

```bash
scp l40s-twodrone-instance:zoro ~/zoro      # 테스트 CLI 받기 (한 번만)
~/zoro                                       # 대화형 실행. 터널은 CLI가 알아서 연다
```

## 구성

| 항목 | 값 |
|---|---|
| 인스턴스 | `l40s-twodrone-instance` (Brev, AWS 서울, L40S 48GB) |
| 서버 주소 | 인스턴스 내부의 `http://127.0.0.1:30001` |
| 서빙 스택 | `trida-stack/inference/zoro-1/serve.py --mode decide` (SGLang 0.5.23) |
| 실행 방식 | systemd 서비스 `zoro`. 항상 떠 있고, 죽으면 자동으로 다시 뜬다 |
| 접근 방식 | 인스턴스 밖에서는 SSH 터널로만 접근 (서버에 인증 없음, 외부 포트는 닫혀 있음) |

```
[맥북] http://127.0.0.1:30001 ══ SSH 터널 ══▶ [인스턴스] 127.0.0.1:30001 ─▶ Zoro-1 (GPU)
```

## 1. 사전 조건

1. Brev 팀에 가입했고 인스턴스 SSH 권한을 받았다.
2. Brev CLI를 설치하고 로그인했다.

```bash
brev login
brev ls        # l40s-twodrone-instance 가 RUNNING 으로 보여야 한다
ssh l40s-twodrone-instance 'curl -s 127.0.0.1:30001/health'   # {"ok":true,...} 이면 서버 정상
```

`ssh l40s-twodrone-instance`가 호스트를 못 찾으면 `brev refresh`를 실행한 뒤 다시 시도한다.

## 2. 터널

맥북에서 서버를 호출하려면 터널이 열려 있어야 한다. `zoro` CLI는 로컬에 서버가 안 보이면 터널을 스스로 열기 때문에, CLI만 쓸 때는 이 단계를 건너뛴다. 한 번 열린 터널은 백그라운드에 남아 이후 호출과 에이전트 코드가 그대로 쓴다.

CLI 없이 API를 직접 호출하는 코드에서는 터널을 직접 연다.

```bash
# 백그라운드로 열기
ssh -f -N -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L 30001:127.0.0.1:30001 l40s-twodrone-instance

# 확인
curl -s http://127.0.0.1:30001/health

# 닫기 (~/zoro --close-tunnel 과 같다)
pkill -f "30001:127.0.0.1:30001 l40s-twodrone-instance"
```

- `ControlMaster=no -o ControlPath=none`은 빼지 않는다. Brev의 SSH 설정은 연결을 공유하는데, 이 옵션이 없으면 터널이 공유 연결에 붙어서 `pkill`로 닫히지 않고 다른 SSH 세션이 끝날 때 같이 끊긴다.
- 터널은 연 맥북에서만 통한다. 팀원마다 각자 연다.
- 맥북 쪽 포트는 바꿔도 된다. `-L 8000:127.0.0.1:30001`로 열면 `http://127.0.0.1:8000`으로 호출한다.
- 맥북이 잠자기에 들어가거나 네트워크가 바뀌면 터널이 끊긴다. `Connection refused`가 나오면 터널부터 다시 연다.

## 3. 테스트 CLI `zoro`

쿼리 하나와 상품 후보 여러 개를 넣으면, 상품별 판단을 매칭 확률 순으로 정렬해 표로 보여준다. 파이썬 표준 라이브러리만 쓴다.

```bash
scp l40s-twodrone-instance:zoro ~/zoro      # 인스턴스에서 받아오기
~/zoro "무설탕 탄산수 2L" "제로 탄산수 2L 6개" "콜라 1.5L 6개" "탄산수 제조기"
```

터널이 없으면 `opening an SSH tunnel ...` 한 줄이 먼저 나오고 결과가 이어진다.

```
query  무설탕 탄산수 2L

  #  item                match                       action        conf   ms
  1  제로 탄산수 2L 6개  0.91 ██████████████████░░  add_to_cart   0.81    94
  2  탄산수 제조기       0.16 ███░░░░░░░░░░░░░░░░░  skip          0.55    95
  3  콜라 1.5L 6개       0.12 ██░░░░░░░░░░░░░░░░░░  skip          0.72    71
```

| 열 | 의미 |
|---|---|
| `match` | "이 상품이 쿼리를 만족하는가"에 대한 예 확률 |
| `action` / `conf` | `add_to_cart`, `skip`, `ask_user` 중 선택된 것과 그 확률 |
| `ms` | 상품별 서버 처리 시간 (네트워크 제외) |

| 사용 형태 | 설명 |
|---|---|
| `~/zoro "쿼리" "상품A" "상품B" ...` | 한 줄 실행. 에이전트는 이 형태를 쓴다 |
| `~/zoro` | 대화형. 쿼리 입력 후 상품을 한 줄씩 넣고 빈 줄에서 실행, `q`로 종료 |
| `-q "질문"` | 예/아니오 질문 교체. 예: `-q "Is this item under 10,000 won?"` |
| `--reason` | 추론을 먼저 쓰고 판단 (느림) |
| `--raw` | 원본 JSON 출력. 프로그램에서 파싱할 때 사용 |
| `--close-tunnel` | CLI가 연 터널 닫기 |

- 서버 주소를 바꾸려면 `ZORO_URL=http://127.0.0.1:8000 ~/zoro ...`처럼 환경변수를 준다. 이 경우 터널은 자동으로 열지 않는다.
- 출력이 터미널이 아니면(파이프, 에이전트 캡처) 색상 코드는 빠진다.

## 4. API 직접 호출

에이전트 코드에서는 CLI 대신 HTTP로 직접 호출한다.

| 경로 | 용도 |
|---|---|
| `POST /v1/systemone` | typed decision: 상태(state)에 대한 질문을 확률과 함께 답한다 |
| `POST /v1/chat/completions` | OpenAI 호환 chat. `tools`, 스트리밍 지원 |
| `GET /health`, `GET /v1/models` | 상태 확인 |

### 4.1 Typed decision

요청 하나에 질문을 여러 개 담을 수 있다. `model` 필드는 필수지만 아무 이름이나 넣어도 된다.

```bash
curl -s http://127.0.0.1:30001/v1/systemone -H 'content-type: application/json' -d '{
  "model": "zoro-1",
  "state": {"query": "무설탕 탄산수 2L", "item": {"title": "제로 탄산수 2L 6개", "price": 5900}},
  "questions": {
    "match":  {"type": "noul",   "instructions": "Does this item satisfy the shopping query?"},
    "action": {"type": "choice", "instructions": "What should the agent do?",
               "criteria": {"add_to_cart": "it matches", "skip": "it does not match", "ask_user": "unclear"}},
    "fit":    {"type": "score",  "instructions": "How well does the item fit the query?",
               "criteria": ["poor", "partial", "exact"]}}}'
```

실제 응답:

```json
{"model": "bf1b76b9...",
 "answers": {
   "match":  {"type": "noul", "noul": 0.9159},
   "action": {"type": "choice", "choice": "add_to_cart", "confidence": 0.6521,
              "probabilities": {"add_to_cart": 0.768, "skip": 0.1096, "ask_user": 0.1223}},
   "fit":    {"type": "score", "score": 1.7105, "legend": {"0": "poor", "1": "partial", "2": "exact"},
              "confidence": 0.5658, "probabilities": {"0": 0.0398, "1": 0.2099, "2": 0.7503}}},
 "usage": {"input_tokens": 187, "output_tokens": 0},
 "timing_ms": 59.0}
```

| 질문 타입 | `criteria` | 답 |
|---|---|---|
| `noul` | 없음 | `noul` = 예일 확률 (0~1) |
| `choice` | `{키: 설명}` | `choice` = 선택된 키, `probabilities` = 키별 확률 |
| `score` | 낮은 단계부터 나열한 리스트 | `score` = 기대 단계 (0부터), `probabilities` = 단계별 확률 |

- `state`는 문자열이나 JSON 객체 모두 가능하다. 판단에 필요한 정보(쿼리, 상품명, 가격, 옵션 등)를 여기에 담는다.
- 임계값 판단에는 `probabilities`를 쓴다. `confidence`는 `probabilities`의 최댓값과 다른 값으로 나오며, 정의는 확인하지 못했다.
- `state`는 약 7,500 토큰까지다. 상품 상세 페이지 전체를 넣지 말고 필요한 필드만 추려 넣는다.
- 검증 오류는 422와 `{"detail": [...]}`로 돌아온다.

### 4.2 Reasoned decision

요청에 `"reasoning": true`를 넣으면 모델이 먼저 추론을 쓰고, 그 추론을 읽은 뒤 판단한다. 수량 계산이나 여러 조건을 따져야 하는 판단에 쓴다.

```json
{"model": "zoro-1", "state": {...}, "questions": {...}, "reasoning": true, "max_reasoning_tokens": 1024}
```

- 응답에 `reasoning`(추론 텍스트)과 `answers_source`가 추가된다.
- `answers_source`가 `"reasoned"`인지 확인한다. `"direct"`이면 추론이 잘렸거나 비어서 직접 판단으로 대체된 것이다(`cut` 필드에 이유).
- `max_reasoning_tokens`는 1~4096이다.
- 측정값: 직접 판단은 30~60ms, 추론 포함 판단은 약 1.7초(추론 127토큰)였다.
- 권장 패턴은 먼저 직접 판단하고, 최고 확률이 0.7 미만일 때만 `reasoning`으로 다시 묻는 것이다(모델 카드의 권장 기본값).

### 4.3 Chat과 tool call

```bash
curl -s http://127.0.0.1:30001/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "zoro-1", "max_tokens": 512,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "장바구니에 제로 탄산수 2L 6개를 2개 담아줘"}],
  "tools": [{"type": "function", "function": {"name": "add_to_cart", "description": "Add a product to the cart",
    "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "quantity": {"type": "integer"}},
                   "required": ["title", "quantity"]}}}]}'
```

- 응답의 `choices[0].message.tool_calls`에 호출이 담기고 `finish_reason`은 `"tool_calls"`다. 위 요청은 `add_to_cart(title="제로 탄산수 2L", quantity=2)`를 돌려줬다.
- `enable_thinking`을 켜면 생각 과정이 `reasoning_content`로 따로 온다.
- OpenAI SDK를 쓰려면 `base_url="http://127.0.0.1:30001/v1"`, `api_key`는 아무 값으로 둔다.

### 4.4 파이썬 예제

```python
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:30001"
session = requests.Session()          # reuse the connection: each new connection costs ~0.3 s through the tunnel

MATCH = {"match": {"type": "noul", "instructions": "Does this item satisfy the shopping query?"}}


def decide(state, questions, **extra):
    body = {"model": "zoro-1", "state": state, "questions": questions, **extra}
    resp = session.post(f"{URL}/v1/systemone", json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()


def rank(query, items):
    """Judge all candidates concurrently; the server batches them on the GPU."""
    def one(item):
        answers = decide({"query": query, "item": item}, MATCH)["answers"]
        return answers["match"]["noul"], item
    with ThreadPoolExecutor(max_workers=16) as pool:
        return sorted(pool.map(one, items), key=lambda pair: -pair[0])


print(rank("무설탕 탄산수 2L", [{"title": "제로 탄산수 2L 6개"}, {"title": "콜라 1.5L 6개"}]))
```

## 5. 성능과 호출 요령

| 측정 | 값 |
|---|---|
| 판단 1건 서버 처리 | 약 33ms |
| 동시 32건 (2개 클라이언트 × 16건) | 전체 0.7~0.9초 |
| 터널 경유 왕복 (연결 재사용) | 요청당 약 0.28초 |
| 터널 경유 왕복 (매번 새 연결) | 요청당 약 0.55초 |

맥북에서 호출하면 서버 처리보다 터널 왕복이 훨씬 길다. 그래서:

- 후보 상품은 순차로 묻지 말고 **동시에** 보낸다. 서버가 묶어서 처리하므로 여러 건이 한 번의 왕복 시간에 끝난다.
- 한 상품에 대한 질문 여러 개는 **한 요청**에 담는다.
- HTTP 연결을 **재사용**한다(`requests.Session`, `httpx.Client`).
- 여러 사람이 동시에 써도 된다. 줄 세우지 않고 배칭한다.

## 6. 서버 관리와 주의사항

서버는 인스턴스의 systemd 서비스 `zoro`로 돈다. 터미널이나 tmux와 무관하게 떠 있고, 프로세스가 죽으면 10초 뒤 자동으로 다시 뜨며, 인스턴스를 재부팅해도 올라온다.

| 할 일 | 인스턴스에서 실행 |
|---|---|
| 상태 확인 | `systemctl status zoro` |
| 로그 보기 | `tail -F ~/zoro-serve.log` |
| 재시작 | `sudo systemctl restart zoro` (다시 뜨는 데 최대 3분) |
| 완전히 끄기 | `sudo systemctl stop zoro` (이 명령으로만 꺼진다) |
| 동작 확인 | `~/zoro-smoke.sh` |

- tmux 세션 `zoro`(`tmux attach -t zoro`)의 위 칸은 로그를 보여주기만 한다. 거기서 `Ctrl-C`를 눌러도 서버는 꺼지지 않는다.
- `~/zoro-serve.sh`는 서비스가 실행하는 스크립트다. 서비스가 떠 있을 때 손으로 실행하면 포트가 겹쳐 실패한다.
- 서버가 GPU 메모리의 80%(약 37.7GB)를 잡고 있다. 같은 GPU에 다른 모델을 올리려면 `~/zoro-serve.sh`의 실행 줄에 `--mem-fraction 0.5`를 추가하고 서비스를 재시작한다.
- 서버에 인증이 없으므로 포트를 외부에 직접 열지 않는다.
- 대회 규정상 System 1 판단(빠른 진단·라우팅·검증)에는 Zoro-1을 써야 한다. 이 서버의 `/v1/systemone`이 그 용도다.

## 7. 문제 해결

| 증상 | 원인과 조치 |
|---|---|
| 맥북에서 `Connection refused` | 터널이 닫혔다. `~/zoro`를 한 번 실행하거나 2번의 명령으로 다시 연다 |
| 터널은 열렸는데 응답 없음 | `ssh l40s-twodrone-instance 'curl -s 127.0.0.1:30001/health'`로 서버 상태를 본다. 실패하면 재시작 중일 수 있으니 3분쯤 기다리고, 그래도 안 되면 `sudo systemctl restart zoro` |
| `could not open the tunnel` | SSH 접속 자체가 안 된다. `brev ls`에서 인스턴스가 RUNNING인지 보고 `brev refresh` 후 재시도한다 |
| `bind: Address already in use` | 맥북의 30001 포트를 이미 쓰고 있다. 기존 터널이면 그대로 쓰고, 아니면 다른 로컬 포트로 연다 |
| HTTP 422 | 요청 형식 오류. `model` 누락, 잘못된 질문 `type`, `criteria` 형식을 확인한다 |
| `answers_source`가 `direct` | 추론이 예산에서 잘렸다. `max_reasoning_tokens`를 늘리거나 `state`를 줄인다 |
| `ssh: Could not resolve hostname` | `brev refresh` 후 재시도. `brev ls`에 인스턴스가 보이는지 확인한다 |

## 참고

- 모델 카드: https://huggingface.co/trillionlabs/Zoro-1
- 서빙 코드와 API 상세: https://github.com/trillion-labs/trida-stack/tree/main/inference/zoro-1
