"""Photo-severity variants for Zoro-1 on the 119 scenario images (train + test), one pass, probabilities saved.

Variants (all zero report labels; thresholds are tuned later on train images only):
  v0  current demo question, photo resized like the demo (area <= 448*448)
  v1  decomposed concrete-damage questions (any of them)
  v2  3-level choice with CrisisMMD-style definitions, P(severe)
  v3  current question on the original photo at ~1000x1000 pixel budget
Usage: ~/venvs/zoro/bin/python severity_variants.py /ephemeral/crisis_bench/scenarios.json out.json
"""
import base64, io, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

SCEN, OUT = sys.argv[1], sys.argv[2]
ROOT = "/ephemeral/crisis_bench/crisismmd/"
URL = "http://127.0.0.1:30001/v1/systemone"
MAXPX = 448 * 448

CURRENT = "사진에 심각한 물적 피해가 보이는가? 건물이 불길에 휩싸이거나 다 타버림, 주택가·도로가 사람 허리 이상 깊이로 넓게 침수됨, 건물·구조물 붕괴, 지붕 소실, 도로·다리 유실은 모두 심각한 피해다."
PARTS = {
    "collapse": "사진에 무너지거나 주저앉은 건물·구조물이 보이는가?",
    "roof": "사진에 지붕이 통째로 날아가거나 크게 뜯겨 나간 건물이 보이는가?",
    "road": "사진에 유실되거나 끊기거나 크게 파손된 도로·다리가 보이는가?",
    "vehicle": "사진에 크게 부서지거나 뒤집히거나 잔해에 깔린 차량이 보이는가?",
    "debris": "사진에 건물 잔해가 넓은 지역에 흩어져 있는가?",
}
LEVELS = {"type": "choice", "instructions": "사진에 보이는 물적 피해의 정도는?", "criteria": {
    "severe": "심각한 피해: 건물이 불길에 휩싸이거나 다 타버림, 주택가·도로가 사람 허리 이상 깊이로 넓게 침수됨, 건물·구조물 붕괴, 지붕 소실, 도로·다리 유실, 넓은 지역의 잔해.",
    "mild": "경미한 피해: 일부 지붕·벽·창문·울타리 손상, 쓰러진 나무나 전선, 약간의 잔해나 얕은 물웅덩이, 구조물은 서 있고 수리 가능한 상태.",
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


def ask(b, questions, extra=None):
    body = {"model": "zoro-1", "state": "첨부 사진", "questions": questions, "images": [base64.b64encode(b).decode()]}
    body.update(extra or {})
    for attempt in range(3):
        try:
            return session.post(URL, json=body, timeout=120).json()["answers"]
        except Exception as e:
            err = e
            time.sleep(2)
    raise err


def judge(item):
    path, sev, split = item
    raw = open(ROOT + path, "rb").read()
    small = demo_resize(raw)
    qa = {"v0": {"type": "noul", "instructions": CURRENT}, "v2": LEVELS}
    a = ask(small, qa)
    return path, {"split": split, "gold_severe": sev, "v0": a["v0"]["noul"], "v2": a["v2"]["probabilities"]["severe"],
                  "v2_choice": a["v2"]["choice"]}


scen = json.load(open(SCEN))
images = {}
for s in scen:
    images.setdefault(s["image_path"], (s["image_path"], s["image_severe"], s["split"]))
items = sorted(images.values(), key=lambda x: (x[2] != "train", x[0]))
print("unique images:", sum(i[2] == "train" for i in items), "train,", sum(i[2] == "test" for i in items), "test", flush=True)
t0 = time.time()
res = {}
with ThreadPoolExecutor(6) as ex:
    for n, (p, r) in enumerate(ex.map(judge, items), 1):
        res[p] = r
        if n % 200 == 0:
            print(f"{n}/{len(items)} {time.time() - t0:.0f}s", flush=True)
json.dump(res, open(OUT, "w"))
print("SEVERITY_DONE", len(res), f"{time.time() - t0:.0f}s")
