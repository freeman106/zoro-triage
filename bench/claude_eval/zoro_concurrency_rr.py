"""Small concurrency probe: is the Zoro server serialising photo requests, and does pre-resizing help?"""
import base64, glob, io, json, statistics, time
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image

import itertools, sys
URLS = [u + "/v1/systemone" for u in (sys.argv[1:] or ["http://127.0.0.1:30001"])]
_rr = itertools.cycle(range(len(URLS)))
sessions = [requests.Session() for _ in URLS]
imgs = sorted(glob.glob("/ephemeral/crisis_bench/crisismmd/data_image/hurricane_harvey/*/*.jpg"))[:16]
raw = [open(p, "rb").read() for p in imgs]

def small(b):
    im = Image.open(io.BytesIO(b)).convert("RGB"); im.thumbnail((448, 448))
    out = io.BytesIO(); im.save(out, "JPEG", quality=85); return out.getvalue()

pre = [small(b) for b in raw]
Q = {k: {"type": "noul", "instructions": t} for k, t in {
    "a": "Is there severe structural damage?", "b": "Is this a real on-scene photo?",
    "c": "Is there an ongoing hazard such as flames or rising water?"}.items()}
s = requests.Session()

def one(b):
    t = time.perf_counter()
    i = next(_rr); r = sessions[i].post(URLS[i], json={"model": "zoro-1", "state": "attached photo", "questions": Q,
                          "images": [base64.b64encode(b).decode()]}, timeout=120).json()
    return (time.perf_counter() - t) * 1000, r.get("timing_ms")

def text_one(_):
    t = time.perf_counter()
    i = next(_rr); sessions[i].post(URLS[i], json={"model": "zoro-1", "state": {"report": "2층에 아이들과 갇혔어요 물이 차요"}, "questions": Q}, timeout=120).json()
    return (time.perf_counter() - t) * 1000, None

print("image sizes KB raw", [len(b) // 1024 for b in raw[:4]], "pre", [len(b) // 1024 for b in pre[:4]])
one(raw[0]); one(pre[0])
for name, data, fn in (("photo original", raw, one), ("photo pre-resized 448", pre, one), ("text only", list(range(16)), text_one)):
    for n in (1, 4, 8, 16):
        batch = data[:n]
        t = time.perf_counter()
        with ThreadPoolExecutor(n) as ex:
            res = list(ex.map(fn, batch))
        wall = (time.perf_counter() - t) * 1000
        lat = [r[0] for r in res]
        print(f"{name:22s} n={n:2d}  wall {wall:6.0f} ms  per-req median {statistics.median(lat):6.0f} ms  throughput {n / wall * 1000:5.1f}/s", flush=True)
print("server timing sample:", res[0][1])
