"""Run the report119 triage rubric against Zoro-1 with a text-only fallback."""

import base64
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from mismatch_types import PHOTO_QUESTION, TEXT_QUESTION, extract, is_mismatch
from scene_question import FIELD_PHOTO_QUESTION


ZORO_MODEL = "zoro-1"
DEFAULT_ZORO_URL = "http://127.0.0.1:30001"
DECISION_TIMEOUT = 2.0
MAX_RESPONSE_BYTES = 64 * 1024
MIN_UNCERTAIN = 0.35
MAX_UNCERTAIN = 0.65
MISMATCH_THRESHOLD = json.loads(
    Path(__file__).with_name("mismatch_config.json").read_text(encoding="utf-8")
)["threshold"]

# Keep the baseline prompts and modality boundaries aligned with bench/report119/evaluate.py.
QUESTIONS = {
    "people": "신고 글에서 사건 당시 고립되거나 부상당한 등 위험한 사람이 언급되는가?",
    "now": "글은 사건이 현재 진행 중이라고 신고하는가? 과거 사진 공유나 문의는 아니오.",
    "life_threat": (
        "신고 글에 지금 사람의 생명이 위태롭거나 즉각적인 구조가 필요한 상황이 명시되어 있는가? "
        "불길 속에 갇힘, 의식이나 호흡이 없음, 물에 잠겨 구조가 필요한 경우는 예. "
        "이미 끝난 상황, 단순 부상이나 재산 피해, 사람의 위험을 추정해야 하는 경우는 아니오."
    ),
    "severe": "사진에 구조물 붕괴, 지붕 소실, 도로 유실 등 심각한 물적 피해가 보이는가?",
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
_DEMO_TEXT_RULE = "고립된 고령자·어린이는 최우선"

_POOL = ThreadPoolExecutor(max_workers=12, thread_name_prefix="zoro")
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
    r"고립|갇[혀혔힌]|부상|다친|다쳐|다쳤|쓰러[져졌진]|의식(?:이|을)?\s*(?:없|잃)|"
    r"(?:잔해|더미)에\s*깔[린려렸]|숨(?:을|이)?\s*.{0,7}(?:못|안\s*쉬)|호흡\s*곤란|"
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
        if question["type"] == "choice":
            if not isinstance(answer, dict) or not isinstance(answer.get("probabilities"), dict):
                raise ValueError("Invalid Zoro choice response")
            answers[key] = extract(answer, question)
            continue
        value = answer["noul"]
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError("Invalid Zoro probability")
        answers[key] = float(value)

    model = result.get("model")
    return answers, model if isinstance(model, str) and model else None


def _noul(instructions):
    return {"type": "noul", "instructions": instructions}


def _model_evidence(text, image, rules, deadline):
    url = (os.environ.get("ZORO_URL") or DEFAULT_ZORO_URL).rstrip("/")
    if not url.endswith("/v1/systemone"):
        url += "/systemone" if url.endswith("/v1") else "/v1/systemone"
    encoded_image = base64.b64encode(image).decode("ascii")
    text_state = {"신고": text}
    photo_state = "첨부 사진"
    text_questions = {
        "people": _noul(QUESTIONS["people"]),
        "now": _noul(QUESTIONS["now"]),
        "claim_kind": TEXT_QUESTION,
        "life_threat": _noul(QUESTIONS["life_threat"]),
    }
    photo_questions = {
        "severe": _noul(QUESTIONS["severe"]),
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


def _model_result(evidence, model, rules):
    probs = {key: evidence[key] for key in (*QUESTIONS, *rules)}
    probs["mismatch"] = None
    claim = evidence["claim_kind"]
    photo = evidence["photo_kind"]
    people = probs["people"] >= 0.5
    severe = probs["severe"] >= 0.5
    now = probs["now"] >= 0.5
    details = _rule_results(rules, probs, now)
    reasons = ["진행 중인 사건" if now else "지난 사건으로 분류"]
    urgency = "non_urgent"
    if now:
        urgency = ("normal", "urgent", "top")[int(people) + int(severe)]
        if people:
            reasons.append("신고 글에 인명 위험")
        if severe:
            reasons.append("사진에 심각한 물적 피해")
        if not people and not severe:
            reasons.append("인명 위험·심각한 사진 피해 신호 없음")
        for rule in details:
            if rule["applied"]:
                urgency = "top"
                reasons.append(f"운영 규칙 적용 · {rule['condition']}")

    mismatch = is_mismatch(claim, photo, MISMATCH_THRESHOLD)
    if mismatch:
        reasons.append("글과 사진의 재난 유형이 다를 수 있음")
    non_field_photo = probs["field_photo"] < 0.5
    if non_field_photo:
        reasons.append("첨부 사진에서 재난 현장을 확인하기 어려움")
    uncertain = [
        label for key, label in REVIEW_LABELS.items()
        if MIN_UNCERTAIN <= probs[key] <= MAX_UNCERTAIN
    ]
    if uncertain:
        reasons.append("판단 불확실 · " + ", ".join(uncertain))

    return {
        "urgency": urgency,
        "probs": probs,
        "mismatch": mismatch,
        "non_field_photo": non_field_photo,
        "claim_kind": claim["choice"],
        "claim_prob": claim["prob"],
        "photo_kind": photo["choice"],
        "photo_prob": photo["prob"],
        "review": bool(mismatch or non_field_photo or uncertain),
        "reasons": reasons,
        "mode": "zoro",
        "model": model,
        "rules": details,
    }


def _fallback(text, rules, reason):
    probs = dict.fromkeys((*QUESTIONS, "mismatch", *rules))
    reasons = [reason]
    if _PAST.search(text) and not _ONGOING.search(text):
        urgency = "non_urgent"
        reasons.append("지난 사건 관련 문구 감지")
    elif _PEOPLE_RISK.search(_NO_PEOPLE_RISK.sub(" ", text)):
        urgency = "urgent"
        reasons.append("인명 위험 키워드 감지")
    else:
        urgency = "normal"
        reasons.append("인명 위험 키워드 미검출")
    reasons.append("사진 피해·현장 사진 여부·글과 사진 일치 여부 미확인")
    if rules:
        reasons.append("운영 규칙 미평가")

    return {
        "urgency": urgency,
        "probs": probs,
        "mismatch": None,
        "non_field_photo": None,
        "claim_kind": None,
        "claim_prob": None,
        "photo_kind": None,
        "photo_prob": None,
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
        except (OSError, HTTPException, ValueError, KeyError, TypeError):
            result = _fallback(text, rule_map, "Zoro 응답을 확인할 수 없음 · 키워드 모드")
        else:
            result = _model_result(evidence, model, rule_map)

    result["ms"] = round((time.monotonic() - started) * 1000, 1)
    return result
