"""Run the report119 triage rubric against Zoro-1 with a text-only fallback."""

import base64
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from http.client import HTTPException
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image

from mismatch_types import PHOTO_QUESTION, TEXT_QUESTION, extract, is_mismatch
from person_choice import QUESTION as PERSON_QUESTION, choose as choose_person, validate as validate_person
from scene_question import FIELD_PHOTO_QUESTION


ZORO_MODEL = "zoro-1"
DEFAULT_ZORO_URL = "http://127.0.0.1:30001"
DECISION_TIMEOUT = 2.0
DEFAULT_WORKERS = 4
MAX_PHOTO_PIXELS = 448 * 448
_PHOTO_GRID = 32  # Zoro's vision patch size (16) times spatial merge size (2).
_MAX_GRID_ASPECT_RATIO = 200
MAX_RESPONSE_BYTES = 64 * 1024
MIN_UNCERTAIN = 0.35
MAX_UNCERTAIN = 0.65
DAMAGE_THRESHOLD = 0.60
DAMAGE_QUESTION = {
    "type": "choice",
    "instructions": "사진에 보이는 물적 피해의 정도는?",
    "criteria": {
        "severe": (
            "심각한 피해: 건물이 불길에 휩싸이거나 다 타버림, 주택가·도로가 사람 허리 이상 깊이로 넓게 침수됨, "
            "건물·구조물 붕괴, 지붕 소실, 도로·다리 유실, 넓은 지역의 잔해."
        ),
        "mild": (
            "경미한 피해: 일부 지붕·벽·창문·울타리 손상, 쓰러진 나무나 전선, 약간의 잔해나 얕은 물웅덩이, "
            "구조물은 서 있고 수리 가능한 상태."
        ),
        "none": "피해가 거의 없음: 건물이나 시설에 눈에 띄는 피해가 없거나 아주 사소함.",
    },
}
MISMATCH_THRESHOLD = json.loads(
    Path(__file__).with_name("mismatch_config.json").read_text(encoding="utf-8")
)["threshold"]

PERSON_THRESHOLD = json.loads(
    Path(__file__).with_name("person_threshold.json").read_text(encoding="utf-8")
)["threshold"]
if type(PERSON_THRESHOLD) not in (int, float) or not math.isfinite(PERSON_THRESHOLD) or not 0 <= PERSON_THRESHOLD <= 1:
    raise ValueError("Invalid person-state threshold")

# Keep the modality boundaries aligned with bench/report119/evaluate.py.
QUESTIONS = {
    "now": "글은 사건이 현재 진행 중이라고 신고하는가? 과거 사진 공유나 문의는 아니오.",
    "field_photo": FIELD_PHOTO_QUESTION,
    "ongoing_hazard": (
        "촬영 시점에 사람의 안전을 즉각 위협할 정도로 진행 중인 위험이 사진에 직접 보이는가? "
        "큰 불길, 거세게 흐르는 침수 물살 등은 예. "
        "그을음, 잔해, 고인 물 등 이미 발생한 피해 흔적만 보이거나 위험의 진행을 추정해야 하는 경우는 아니오."
    ),
}
REVIEW_LABELS = {
    "people": "인명 위험",
    "now": "진행 여부",
    "severe": "사진 피해",
}
_DISPLAY_PROBS = ("people", "life_threat", "severe", *QUESTIONS, "mismatch")
_DEMO_TEXT_RULE = "고립된 고령자·어린이는 최우선"

_POOL = ThreadPoolExecutor(
    max_workers=max(12, 2 * int(os.environ.get("DEMO_WORKERS", DEFAULT_WORKERS))),
    thread_name_prefix="zoro",
)
_PAST = re.compile(
    r"보험\s*(?:처리|문의|접수)|기록\s*(?:용|차)|이미\s*.{0,12}(?:끝났|종료|진화|정리)|"
    r"(?:어제|지난\s*(?:주|달|해)|작년|예전|며칠\s*전|과거|예전에)(?!부터)"
    r".{0,30}(?:사진|공유|문의|났던|있었던)|뉴스에서\s*본\s*사진"
)
_ONGOING = re.compile(
    r"(?:지금|아직|현재).{0,16}(?:진행|번지|차오|갇|고립|못\s*나|침수|타고|불이|흔들)|"
    r"(?:구조|구해|살려)\s*(?:주|주세요|줘)|(?:부터|계속).{0,16}(?:번지|차오|갇|고립|못\s*나)"
)
_PEOPLE_RISK = re.compile(
    r"고립|갇[혀혔힌힘]|부상|다친|다쳐|다쳤|쓰러[져졌진]|"
    r"(?:사람|아[이가]|가족|어르신).{0,12}못\s*나|(?:구조|구해|살려)\s*(?:주|줘)"
)
_NO_PEOPLE_RISK = re.compile(
    r"(?:다친|부상당한|갇힌|고립된)\s*(?:사람|분|인원)?(?:은|는|이|도)?\s*(?:아무도\s*)?없|"
    r"(?:부상자|부상|인명\s*피해)(?:은|는|가|이|도)?\s*(?:전혀\s*)?없"
)


def _systemone(url, state, questions, image, deadline):
    body = {
        "model": ZORO_MODEL,
        "state": state,
        "questions": questions,
    }
    if image is not None:
        body["images"] = [image]

    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    with urlopen(request, timeout=remaining) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("Oversize Zoro response")

    result = json.loads(payload)
    answers = {}
    for key, question in questions.items():
        answer = result["answers"][key]
        if key == "person_state":
            answers[key] = validate_person(answer["probabilities"])
            continue
        if question["type"] == "choice":
            if not isinstance(answer, dict) or not isinstance(answer.get("probabilities"), dict):
                raise ValueError("Invalid Zoro choice response")
            selected = extract(answer, question)
            if key == "severe":
                distribution = answer["probabilities"]
                if abs(sum(distribution.values()) - 1) > 0.01:
                    raise ValueError("Non-normalized damage probabilities")
                answers[key] = distribution["severe"]
                answers["damage_probabilities"] = distribution
            else:
                answers[key] = selected
            continue
        value = answer["noul"]
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError("Invalid Zoro probability")
        answers[key] = float(value)

    model = result.get("model")
    return answers, model if isinstance(model, str) and model else None


def _noul(instructions):
    return {"type": "noul", "instructions": instructions}


def _photo_size(width, height):
    # Match Zoro's smart_resize grid so its processor can use the same dimensions.
    # For aspect ratios outside that processor's range, keep proportional dimensions.
    grid = _PHOTO_GRID if max(width, height) <= _MAX_GRID_ASPECT_RATIO * min(width, height) else 1
    rounded = (round(width / grid) * grid, round(height / grid) * grid)
    if rounded[0] * rounded[1] <= MAX_PHOTO_PIXELS:
        return rounded

    divisor = math.sqrt((width * height) / MAX_PHOTO_PIXELS)
    width = max(grid, math.floor(width / divisor / grid) * grid)
    height = max(grid, math.floor(height / divisor / grid) * grid)
    if width * height > MAX_PHOTO_PIXELS:
        # Flooring can exceed the budget when the short edge is pinned to one grid cell.
        if width > height:
            width = (MAX_PHOTO_PIXELS // (height * grid)) * grid
        else:
            height = (MAX_PHOTO_PIXELS // (width * grid)) * grid
    return width, height


def prepare_photo(image: bytes) -> bytes:
    """Return the image bytes to send to Zoro."""
    with Image.open(BytesIO(image)) as photo:
        width, height = photo.size
        if width * height <= MAX_PHOTO_PIXELS:
            return image

        # Like Zoro's decoder, convert directly to RGB and use the stored pixel orientation.
        with (
            photo.convert("RGB") as rgb,
            rgb.resize(_photo_size(width, height), Image.Resampling.BICUBIC) as resized,
            BytesIO() as output,
        ):
            resized.save(output, "JPEG", quality=95)
            return output.getvalue()


def _model_evidence(text, image, rules, deadline):
    url = (os.environ.get("ZORO_URL") or DEFAULT_ZORO_URL).rstrip("/")
    if not url.endswith("/v1/systemone"):
        url += "/systemone" if url.endswith("/v1") else "/v1/systemone"
    encoded_image = base64.b64encode(prepare_photo(image)).decode("ascii")
    text_state = {"신고": text}
    photo_state = "첨부 사진"
    text_questions = {
        "person_state": PERSON_QUESTION,
        "now": _noul(QUESTIONS["now"]),
        "claim_kind": TEXT_QUESTION,
    }
    photo_questions = {
        "severe": DAMAGE_QUESTION,
        "field_photo": _noul(QUESTIONS["field_photo"]),
        "photo_kind": PHOTO_QUESTION,
        "ongoing_hazard": _noul(QUESTIONS["ongoing_hazard"]),
    }
    combined_questions = {}
    for key, condition in rules.items():
        prefix, _, requirement = condition.partition(":")
        requirement = requirement.strip()
        if condition == _DEMO_TEXT_RULE or prefix == "글" and requirement:
            text_questions[key] = _noul(
                "신고 글에 다음 우선 처리 기준을 충족하는 상황이 언급되는가? "
                f"우선 처리 기준: {requirement or condition}"
            )
        elif prefix == "사진" and requirement:
            photo_questions[key] = _noul(
                "첨부 사진에 다음 우선 처리 기준을 충족하는 상황이 보이는가? "
                f"우선 처리 기준: {requirement}"
            )
        else:
            combined_questions[key] = _noul(
                "신고 내용에 다음 우선 처리 기준을 충족하는 상황이 있는가? "
                "글에 나온 사실은 사진에 보이지 않아도 인정하라. "
                f"우선 처리 기준: {condition}"
            )

    requests = [
        (text_state, text_questions, None),
        (photo_state, photo_questions, encoded_image),
    ]
    if combined_questions:
        requests.append((text_state, combined_questions, encoded_image))

    # The per-socket timeout alone does not bound total time through a slow tunnel.
    futures = [_POOL.submit(_systemone, url, *request, deadline) for request in requests]
    try:
        done, pending = wait(
            futures,
            timeout=max(0, deadline - time.monotonic()),
            return_when=FIRST_EXCEPTION,
        )
        for future in done:
            future.result()
        if pending:
            raise TimeoutError

        evidence = {}
        model = None
        for future in futures:
            answers, reported_model = future.result()
            evidence.update(answers)
            model = model or reported_model
        return evidence, model or ZORO_MODEL
    finally:
        for future in futures:
            future.cancel()


def _rule_results(rules, probs, now):
    details = []
    for key, condition in rules.items():
        probability = probs[key]
        matched = None if probability is None else probability >= 0.5
        details.append({
            "key": key,
            "condition": condition,
            "matched": matched,
            "applied": bool(now and matched),
        })
    return details


def _urgency(now, person_state, severe, ongoing_hazard, rule_matched):
    if not now:
        return "urgent" if ongoing_hazard else "non_urgent"
    if rule_matched or person_state == "life_threat" or (person_state == "trapped" and ongoing_hazard):
        return "top"
    if person_state != "none" or ongoing_hazard or severe:
        return "urgent"
    return "normal"


def _uncertainty(probs, person_probabilities, policy, urgency):
    relevant = []
    advisory = []
    for key, label in REVIEW_LABELS.items():
        if not MIN_UNCERTAIN <= probs[key] <= MAX_UNCERTAIN:
            continue
        alternative = policy.copy()
        if key == "people":
            # Keep the calibrated life-threat cutoff while flipping reported person presence.
            alternative["person_state"] = (
                max(("trapped", "minor_injury"), key=person_probabilities.__getitem__)
                if policy["person_state"] == "none" else "none"
            )
        else:
            alternative[key] = not policy[key]
        if _urgency(**alternative) != urgency:
            relevant.append(key)
        else:
            advisory.append(label)
    return relevant, advisory


def _model_result(evidence, model, rules):
    person_probabilities = evidence["person_state"]
    person_state = choose_person(person_probabilities, PERSON_THRESHOLD)
    probs = {
        "people": 1 - person_probabilities["none"],
        "life_threat": person_probabilities["life_threat"],
        "severe": evidence["severe"],
        **{key: evidence[key] for key in (*QUESTIONS, *rules)},
        "mismatch": None,
    }
    claim = evidence["claim_kind"]
    photo = evidence["photo_kind"]
    severe = probs["severe"] >= DAMAGE_THRESHOLD
    now = probs["now"] >= 0.5
    ongoing_hazard = probs["ongoing_hazard"] >= 0.5
    past_with_hazard = not now and ongoing_hazard
    details = _rule_results(rules, probs, now)
    policy = dict(now=now, person_state=person_state, severe=severe, ongoing_hazard=ongoing_hazard,
                  rule_matched=any(rule["matched"] for rule in details))
    urgency = _urgency(**policy)
    if not now:
        reasons = [
            "지난 사건으로 분류됐지만 사진에 촬영 당시 즉각적인 위험이 보여 검수 필요"
            if past_with_hazard else "지난 사건으로 분류"
        ]
    else:
        reasons = ["진행 중인 사건"]
        if urgency == "normal":
            reasons.append("인명 위험·사진 속 진행 위험·심각한 물적 피해 신호 없음")
        if person_state == "life_threat":
            reasons.append("신고 글에 즉각적인 생명 위협")
        elif person_state == "trapped":
            reasons.append("신고 글에 인명 고립")
        elif person_state == "minor_injury":
            reasons.append("신고 글에 경상")
        if ongoing_hazard:
            reasons.append("사진에 촬영 당시 진행 중인 위험")
        if severe:
            reasons.append("사진에 심각한 물적 피해")
        for rule in details:
            if rule["applied"]:
                reasons.append(f"운영 규칙 적용 · {rule['condition']}")

    mismatch = is_mismatch(claim, photo, MISMATCH_THRESHOLD)
    if mismatch:
        reasons.append("글과 사진의 재난 유형이 다를 수 있음")
    non_field_photo = probs["field_photo"] < 0.5
    if non_field_photo:
        reasons.append("첨부 사진에서 재난 현장을 확인하기 어려움")
    uncertain, advisory = _uncertainty(probs, person_probabilities, policy, urgency)
    if uncertain:
        reasons.append("판단 불확실 · " + ", ".join(REVIEW_LABELS[key] for key in uncertain))
    if advisory:
        reasons.append("참고 · 등급에 영향 없는 불확실 판단 · " + ", ".join(advisory))

    return {
        "urgency": urgency,
        "probs": probs,
        "mismatch": mismatch,
        "non_field_photo": non_field_photo,
        "person_state": person_state,
        "person_probabilities": person_probabilities,
        "person_threshold": PERSON_THRESHOLD,
        "claim_kind": claim["choice"],
        "claim_prob": claim["prob"],
        "photo_kind": photo["choice"],
        "photo_prob": photo["prob"],
        "review_uncertainty": uncertain,
        "damage_probabilities": evidence["damage_probabilities"],
        "damage_threshold": DAMAGE_THRESHOLD,
        "review": bool(past_with_hazard or mismatch or non_field_photo or uncertain),
        "reasons": reasons,
        "mode": "zoro",
        "model": model,
        "rules": details,
    }


_CRUSHED_PERSON = (
    r"(?:사람|환자|아이|아기|어린이|어르신|노인|가족|친구|동료|시민|학생|행인|"
    r"남성|여성|남자|여자|운전자|작업자|인부|부모님|아버지|어머니|아빠|엄마|"
    r"누군가|제가|저|(?:한|두|세|\d+)\s*(?:명|분))"
)
_LIFE_THREAT = re.compile(
    r"(?<![가-힣])숨(?:을|이|은|도)?\s*(?:(?:전혀|잘|제대로)\s*)?"
    r"(?:(?:안|못)\s*(?:쉬|쉼|쉽)|쉬지\s*(?:않|못)|쉴\s*수(?:가)?\s*없|멎|막[혀혔]|끊[겨겼]|없)|"
    r"호흡\s*(?:을|이|은|도)?\s*(?:(?:전혀|제대로)\s*)?"
    r"(?:곤란|정지|없|멎|멈췄|(?:안|못)\s*(?:함|하|해|했|돼|되|됨)|하지\s*(?:못|않))|"
    r"의식(?:을|이|은|도)?\s*(?:전혀\s*)?(?:없|잃(?!\s*지)|불명)|"
    rf"{_CRUSHED_PERSON}(?!(?:이|가)?\s*없(?:는|던))(?:이|가|은|는|도)?\s*"
    r"(?:[^.!?,\n]{1,16}(?:에|아래|밑)\s+)?"
    r"깔(?:림|렸|려\s*있(?:는|던)\s*상태|려(?!\s*있(?:는|던))|린\s*(?:채|상태|것))|"
    rf"깔(?:린|려\s*있(?:는|던))\s*{_CRUSHED_PERSON}|"
    # Match subjectless crush descriptions only as complete clauses.
    r"(?:^|[.!?\n])\s*(?:깔림|(?:(?:건물|무너진)\s+)?"
    r"(?:잔해|더미|차|차량|자동차|트럭|버스|건물|벽|기둥|철판|철골|기계|구조물)"
    r"(?:에|\s*(?:아래|밑)(?:에)?)\s*"
    r"깔(?:림|렸(?:어요|습니다|다|음|어)?|려\s*있(?:어요|습니다|음|다)))(?=\s*(?:[.!?,\n]|$))|"
    r"(?:심한|심각한|(?:다량|대량)(?:의)?)\s*출혈|"
    r"출혈(?:이|은|도)?\s*(?:(?:아주|매우|너무)\s*)?(?:심(?:함|합|해|했|한|하다)|거셈)|"
    r"(?<![가-힣])피(?:가|를)?\s*(?:(?:아주|매우|너무)\s*)?"
    r"(?:많이\s*(?:남|납니다|나요|났|흐르|흐름|흘(?:러|려|렸))|(?:멈추지\s*않|안\s*멈춰))"
)
_NO_LIFE_THREAT = re.compile(
    r"(?:다는|다는\s*말은|다는\s*것은)\s*(?:사실이\s*)?아[니닙님]|"
    r"\s*(?:(?:(?:은|는)\s*)?(?:증상|환자|사람|경우)\s*)?"
    r"(?:은|는|이|가|도)?\s*(?:(?:전혀|아무도)\s*)?"
    r"(?:없|아[니닙님]|안\s*(?:보|나)|(?:있지|하지|하진|지|진)\s*않)|"
    r"\s*(?:(?:이|가|은|는)\s*있)?(?:는|은|인)\s*(?:건|것(?:은|이)?)\s*아[니닙님]"
)


def _fallback(text, rules, reason):
    probs = dict.fromkeys((*_DISPLAY_PROBS, *rules))
    reasons = [reason, "신고 글 키워드로 잠정 분류"]
    if any(not _NO_LIFE_THREAT.match(text, match.end()) for match in _LIFE_THREAT.finditer(text)):
        urgency = "top"
        reasons.append("즉각적인 생명 위협 키워드 감지")
    elif _PAST.search(text) and not _ONGOING.search(text):
        urgency = "non_urgent"
        reasons.append("지난 사건 관련 문구 감지")
    elif _PEOPLE_RISK.search(_NO_PEOPLE_RISK.sub(" ", text)):
        urgency = "urgent"
        reasons.append("인명 위험 키워드 감지")
    else:
        urgency = "normal"
        reasons.append("인명 위험 키워드 미검출")
    reasons.append("사진의 진행 위험·물적 피해·현장 여부·글과 사진 일치 여부 미확인")
    if rules:
        reasons.append("운영 규칙 미평가")

    return {
        "urgency": urgency,
        "probs": probs,
        "mismatch": None,
        "non_field_photo": None,
        "person_state": None,
        "person_probabilities": None,
        "person_threshold": PERSON_THRESHOLD,
        "claim_kind": None,
        "claim_prob": None,
        "photo_kind": None,
        "photo_prob": None,
        "review_uncertainty": [],
        "damage_probabilities": None,
        "damage_threshold": DAMAGE_THRESHOLD,
        "review": True,
        "reasons": reasons,
        "mode": "fallback",
        "rules": _rule_results(rules, probs, now=False),
    }


def decide(text: str, image: bytes, rules: list[str], force_fallback: bool = False) -> dict:
    """Return a decision and its evidence within the model's shared time budget."""
    started = time.monotonic()
    rule_map = {
        f"rule_{index}": condition
        for index, condition in enumerate((rule.strip() for rule in rules if rule.strip()), 1)
    }

    if force_fallback:
        result = _fallback(text, rule_map, "Zoro 연결 장애 모의 · 키워드 모드")
    elif not image:
        result = _fallback(text, rule_map, "첨부 사진 없음 · 키워드 모드")
    else:
        try:
            evidence, model = _model_evidence(text, image, rule_map, started + DECISION_TIMEOUT)
        except TimeoutError:
            result = _fallback(text, rule_map, "Zoro 응답 시간 초과 · 키워드 모드")
        except URLError as error:
            reason = "Zoro 응답 시간 초과" if isinstance(error.reason, TimeoutError) else "Zoro 연결 또는 응답 오류"
            result = _fallback(text, rule_map, reason + " · 키워드 모드")
        except (OSError, HTTPException, Image.DecompressionBombError, ValueError, KeyError, TypeError):
            result = _fallback(text, rule_map, "Zoro 응답을 확인할 수 없음 · 키워드 모드")
        else:
            result = _model_result(evidence, model, rule_map)

    result["policy_version"] = "person-choice-v1-damage-choice-v1-review-v2"
    result["ms"] = round((time.monotonic() - started) * 1000, 1)
    return result
