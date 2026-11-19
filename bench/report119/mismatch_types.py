"""Modality-specific Zoro questions and conservative fire/non-fire conflict rule."""
import math

TEXT_QUESTION = {
    "type": "choice",
    "instructions": "신고자가 신고 글에서 사건의 원인 또는 종류로 주장하는 재난 하나를 선택하라. 인명 피해와 신고 시점은 재난 종류가 아니다.",
    "criteria": {
        "fire": "산불, 건물·차량 화재 등 불이 난 사건을 주장한다.",
        "storm_flood": "침수, 홍수, 폭우, 태풍, 강풍, 폭풍해일 등 물·바람 재난을 주장한다.",
        "quake_collapse": "지진이나 건물·구조물 붕괴 자체를 사건으로 주장한다.",
        "other": "다른 종류의 사건이거나 글에 재난 종류가 드러나지 않는다.",
    },
}
PHOTO_QUESTION = {
    "type": "choice",
    "instructions": "사진 자체에 보이는 재난의 시각적 증거를 기준으로 하나를 선택하라. 사건 출처나 사진 밖 원인을 추측하지 말고, 구별할 증거가 없으면 unclear를 선택하라.",
    "criteria": {
        "fire_smoke_burn": "화염, 화재에서 솟는 짙은 연기, 검게 탄 잔해나 뚜렷한 소실 흔적이 보인다.",
        "water_storm": "도로·건물에 차오른 물, 범람, 폭풍·강풍에 쓰러진 나무 또는 뜯겨 날아간 지붕 등 물·바람 피해가 보인다.",
        "collapse": "주저앉거나 무너진 건물, 교량 또는 구조물의 붕괴가 뚜렷이 보인다.",
        "unclear": "인물, 복구 작업, 보통 풍경, 일반적인 손상 등만 보이고 어느 재난 종류인지 판별할 근거가 부족하다.",
    },
}


def extract(answer, question):
    label = answer["choice"]
    probabilities = answer["probabilities"]
    if label not in question["criteria"] or set(probabilities) != set(question["criteria"]):
        raise ValueError("Invalid choice response")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Invalid choice probabilities")
    return {"choice": label, "prob": probabilities[label]}


def is_mismatch(text_result, photo_result, threshold):
    text, photo = text_result["choice"], photo_result["choice"]
    conflict = ((text == "fire" and photo in ("water_storm", "collapse"))
                or (text in ("storm_flood", "quake_collapse") and photo == "fire_smoke_burn"))
    return conflict and min(text_result["prob"], photo_result["prob"]) >= threshold
