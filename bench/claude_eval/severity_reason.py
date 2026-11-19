"""Escalate uncertain photos to Zoro's reasoned decision for the 3-level severity choice.

Runs reasoning on every image whose direct P(severe) lies in a wide band, so the band itself can be tuned
on train images afterwards without re-querying. Usage: severity_reason.py severity_variants.json out.json
"""
import base64, io, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

RES, OUT = sys.argv[1], sys.argv[2]
ROOT = "/ephemeral/crisis_bench/crisismmd/"
URL = "http://127.0.0.1:30001/v1/systemone"
MAXPX = 448 * 448
LEVELS = {"type": "choice", "instructions": "사진에 보이는 물적 피해의 정도는?", "criteria": {
    "severe": "심각한 피해: 건물이나 구조물이 무너지거나 크게 파괴되어 쓸 수 없는 상태, 지붕이 사라짐, 도로·다리 유실, 넓은 지역의 잔해.",
    "mild": "경미한 피해: 일부 지붕·벽·창문·울타리 손상, 쓰러진 나무나 약간의 잔해, 구조물은 서 있고 수리 가능한 상태.",
    "none": "피해가 거의 없음: 건물이나 시설에 눈에 띄는 피해가 없거나 아주 사소함."}}
session = requests.Session()


def demo_resize(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    if w * h > MAXPX:
        s = math.sqrt(MAXPX / (w * h))
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=95)
    return out.getvalue()


def reason(path):
    b = base64.b64encode(demo_resize(open(ROOT + path, "rb").read())).decode()
    t = time.time()
    d = session.post(URL, json={"model": "zoro-1", "state": "첨부 사진", "questions": {"v2": LEVELS}, "images": [b],
                                "reasoning": True, "max_reasoning_tokens": 512}, timeout=300).json()
    a = d["answers"]["v2"]
    return path, {"p": a["probabilities"]["severe"], "choice": a["choice"], "source": d.get("answers_source"),
                  "cut": d.get("cut"), "ms": round((time.time() - t) * 1000)}


res = json.load(open(RES))
todo = sorted(p for p, r in res.items() if 0.15 <= r["v2"] <= 0.85)
print("escalation candidates:", len(todo), "of", len(res), flush=True)
t0 = time.time()
out = {}
with ThreadPoolExecutor(8) as ex:
    for n, (p, r) in enumerate(ex.map(reason, todo), 1):
        out[p] = r
        if n % 100 == 0:
            print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
json.dump(out, open(OUT, "w"))
print("REASON_DONE", len(out), f"{time.time() - t0:.0f}s")
