"""Zero-shot Zoro-1 on real 119 call transcripts (AI Hub validation sample, 300 per urgency level).
Usage: ~/venvs/bench/bin/python aihub_eval.py sample900.json out.json   (run from a copy of the demo code)
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests
import person_choice  # the demo's own person-state question
from decision import PERSON_QUESTION

URL = "http://127.0.0.1:30001/v1/systemone"
LARGE = {"type": "choice", "instructions": "이 119 신고 통화는 어떤 종류의 출동이 필요한 신고인가?", "criteria": {
    "구급": "환자 발생: 질병, 통증, 부상, 의식 저하, 심정지, 출산, 중독 등 의료 처치나 병원 이송이 필요하다.",
    "구조": "구조 필요: 갇힘, 고립, 추락, 끼임, 물놀이·산악 사고, 문 개방, 동물 등 구조대 활동이 필요하다.",
    "화재": "화재: 불, 연기, 폭발, 타는 냄새 등 화재 진압이나 확인이 필요하다.",
    "기타": "그 밖의 신고: 단순 문의, 오인 신고, 다른 기관 안내, 비긴급 민원."}}
URGENCY = {"type": "choice", "instructions": "이 신고의 긴급도는?", "criteria": {
    "상": "생명이 위태롭거나 즉시 출동이 필요하다: 심정지, 의식 없음, 호흡 곤란, 심한 출혈, 중증 질환 의심, 건물 화재, 사람이 위험하게 갇힘.",
    "중": "출동이 필요하지만 즉각적인 생명 위협은 아니다: 일반 부상이나 질병, 이송 요청이 필요한 환자, 소규모 화재, 위험이 크지 않은 고립·구조.",
    "하": "긴급성이 낮다: 경미한 증상, 단순 문의나 안내, 오인 신고, 이미 해결된 상황, 비긴급 민원."}}
rows = json.load(open(sys.argv[1]))
s = requests.Session()

def run(r):
    body = {"model": "zoro-1", "state": {"신고 통화 전사": r["dialogue"]},
            "questions": {"large": LARGE, "urgency": URGENCY, "person_state": PERSON_QUESTION}}
    t = time.time()
    for attempt in range(3):
        try:
            a = s.post(URL, json=body, timeout=120).json()["answers"]
            break
        except Exception as e:
            err = e; time.sleep(2)
    else:
        raise err
    return r["id"], {"large": a["large"]["choice"], "urgency": a["urgency"]["choice"], "urgency_p": a["urgency"]["probabilities"],
                     "person_state": a["person_state"]["choice"], "person_p": a["person_state"]["probabilities"], "ms": round((time.time() - t) * 1000)}

out = {}
with ThreadPoolExecutor(4) as ex:
    for n, (i, d) in enumerate(ex.map(run, rows), 1):
        out[i] = d
        if n % 150 == 0: print(n, flush=True)
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False)
print("AIHUB_EVAL_DONE", len(out))
