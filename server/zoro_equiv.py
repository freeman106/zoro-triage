"""Record (or compare against) Zoro answers for a fixed request set, to prove a server change is output-neutral.

Usage: python zoro_equiv.py record out.json [url]   |   python zoro_equiv.py compare ref.json [url]
"""
import base64, glob, json, sys
import requests

mode, path = sys.argv[1], sys.argv[2]
url = (sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:30001") + "/v1/systemone"
imgs = sorted(glob.glob("/ephemeral/crisis_bench/crisismmd/data_image/*/*/*.jpg"))[::400][:24]
Q = {"severe": {"type": "noul", "instructions": "Is there severe structural damage?"},
     "kind": {"type": "choice", "instructions": "Which disaster is visible?",
              "criteria": {"fire": "flames, smoke or burn scars", "water": "flood or storm damage", "collapse": "collapsed buildings", "unclear": "cannot tell"}}}
TQ = {"person": {"type": "choice", "instructions": "신고 글에 나온 사람의 상태는?",
                 "criteria": {"life_threat": "의식 없음·호흡 곤란·깔림·심한 출혈", "trapped": "고립·갇힘", "none": "언급 없음"}}}
texts = ["2층에 아이들과 갇혔어요 물이 계속 차요", "어제 사진인데 보험 문의드려요", "사람이 쓰러져서 숨을 안 쉬어요", "지붕이 날아갔어요 다친 사람은 없어요"]
out = {}
for p in imgs:
    b = base64.b64encode(open(p, "rb").read()).decode()
    a = requests.post(url, json={"model": "z", "state": "attached photo", "questions": Q, "images": [b]}, timeout=60).json()["answers"]
    out[p] = {"severe": a["severe"]["noul"], "kind": a["kind"]["probabilities"]}
for t in texts:
    a = requests.post(url, json={"model": "z", "state": {"report": t}, "questions": TQ}, timeout=60).json()["answers"]
    out[t] = {"person": a["person"]["probabilities"]}
if mode == "record":
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=0)
    print("recorded", len(out))
else:
    ref = json.load(open(path))
    worst = 0.0
    for k, v in out.items():
        for q, val in v.items():
            r = ref[k][q]
            diffs = [abs(val - r)] if isinstance(val, float) else [abs(val[o] - r[o]) for o in val]
            worst = max(worst, *diffs)
    print(f"compared {len(out)} items, max abs probability diff {worst:.6f}")
