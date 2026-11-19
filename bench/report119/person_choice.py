"""Report-time person state; preserve the past/current distinction in the other head."""
import math

QUESTION = {
    "type":"choice",
    "instructions": (
        "신고 글에서 사건 당시 사람에게 실제로 생긴 상태를 하나 고르라. 과거 사건도 당시 상태로 분류한다. "
        "명시된 사실만 사용한다. 화재·침수·붕괴의 규모, 말투, 구조 요청 자체로 증상이나 부상을 추측하지 않는다. "
        "여러 상태가 명시되면 생명 위협 증상, 고립, 경상, 없음 순으로 선택한다."
    ),
    "criteria":{
        "life_threat":"의식이 없거나 쓰러져 반응하지 않음, 숨을 잘 쉬지 못함, 사람이 잔해에 깔림, 피를 많이 흘리는 등 생명 위협 증상이 글에 명시됨.",
        "trapped":"사람이 고립되거나 갇혀 대피하지 못함, 또는 안에 있는 사람에게 연락이 닿지 않음. 위 생명 위협 증상은 명시되지 않음.",
        "minor_injury":"가벼운 상처나 부상이 있으나 의식·호흡 이상, 사람 깔림, 심한 출혈이나 고립 언급은 없음.",
        "none":"위험한 사람에 대한 언급이 없거나 모두 대피하여 다친 사람이 없음. 재산 피해만 설명하거나 인명 위험을 부인함.",
    },
}

def validate(probabilities):
    if not isinstance(probabilities,dict) or probabilities.keys()!=QUESTION["criteria"].keys():
        raise ValueError("Missing person-state class")
    if any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1 for p in probabilities.values()):
        raise ValueError("Invalid person-state probability")
    if abs(sum(probabilities.values())-1)>.01:
        raise ValueError("Non-normalized person-state probabilities")
    return probabilities

def choose(probabilities,tau):
    if probabilities["life_threat"]>=tau:
        return "life_threat"
    return max(("trapped","minor_injury","none"),key=probabilities.__getitem__)
