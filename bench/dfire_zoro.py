"""D-Fire test split: extract images + gold labels, then judge every image with Zoro-1.

Usage: ~/venvs/zoro/bin/python ~/dfire_zoro.py [max_pixels ...]
Writes /ephemeral/crisis_bench/dfire/{images/,gold.json,zoro_<px>.json}
"""
import base64, glob, io, json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import requests
from PIL import Image

ROOT = "/ephemeral/crisis_bench/dfire"
URL = "http://127.0.0.1:30001/v1/systemone"
os.makedirs(f"{ROOT}/images", exist_ok=True)

# ---------------------------------------------------------------- gold labels
if not os.path.exists(f"{ROOT}/gold.json"):
    gold = {}
    for p in sorted(glob.glob(f"{ROOT}/data/test-*.parquet")):
        for row in pq.read_table(p).to_pylist():
            name = row["filename"]
            raw = row["image"]["bytes"]
            img = Image.open(io.BytesIO(raw))
            with open(f"{ROOT}/images/{name}", "wb") as f:
                f.write(raw)
            boxes = [l.split() for l in row["label"].splitlines() if l.strip()]
            area = {0: 0.0, 1: 0.0}
            for b in boxes:
                c, w, h = int(b[0]), float(b[3]), float(b[4])
                area[c] = max(area[c], w * h)
            gold[name] = {"smoke": any(int(b[0]) == 0 for b in boxes), "fire": any(int(b[0]) == 1 for b in boxes),
                          "max_smoke_area": round(area[0], 5), "max_fire_area": round(area[1], 5), "size": img.size}
    json.dump(gold, open(f"{ROOT}/gold.json", "w"))
gold = json.load(open(f"{ROOT}/gold.json"))
cats = {"fire+smoke": 0, "smoke only": 0, "fire only": 0, "none": 0}
for g in gold.values():
    cats["fire+smoke" if g["fire"] and g["smoke"] else "smoke only" if g["smoke"] else "fire only" if g["fire"] else "none"] += 1
print("gold:", len(gold), cats, flush=True)

# ---------------------------------------------------------------- Zoro
QUESTIONS = {
    "smoke": {"type": "noul", "instructions": "Is there visible smoke from a fire anywhere in the image, even a small or distant plume? Clouds, fog, mist, dust and steam are not smoke."},
    "flame": {"type": "noul", "instructions": "Are there visible flames (open fire) anywhere in the image, even small or distant ones? Sunsets, lamps, reflections and orange objects are not flames."},
    "alarm": {"type": "noul", "instructions": "Should a fire-watch camera raise a fire alarm for this image? Yes if flames or smoke from a fire are visible anywhere, however small. No for clouds, fog, sunsets, lights or reflections."},
}
session = requests.Session()


def judge(name, max_px):
    with open(f"{ROOT}/images/{name}", "rb") as f:
        img = base64.b64encode(f.read()).decode()
    body = {"model": "zoro-1", "state": "A still image from a camera that watches for fires (surveillance, wildfire lookout or web photo).",
            "questions": QUESTIONS, "images": [img]}
    if max_px:
        body["max_image_pixels"] = max_px
    t = time.time()
    for attempt in range(3):
        try:
            a = session.post(URL, json=body, timeout=300).json()["answers"]
            return name, {k: round(a[k]["noul"], 4) for k in QUESTIONS}, time.time() - t
        except Exception as e:
            err = e
            time.sleep(2)
    raise err


names = sorted(gold)
for max_px in [int(x) for x in sys.argv[1:]] or [0]:
    out = f"{ROOT}/zoro_{max_px or 'default'}.json"
    judge(names[0], max_px)
    seq = sorted(judge(n, max_px)[2] for n in names[:40])
    t0 = time.time()
    res = {}
    with ThreadPoolExecutor(8) as ex:
        for i, (n, p, _) in enumerate(ex.map(lambda n: judge(n, max_px), names), 1):
            res[n] = p
            if i % 500 == 0:
                print(f"  [{max_px or 'default'}] {i}/{len(names)}  {(time.time() - t0) / i * 1000:.0f} ms/img", flush=True)
    wall = time.time() - t0
    json.dump({"max_px": max_px, "wall": round(wall, 1), "seq_ms_median": round(statistics.median(seq) * 1000), "preds": res}, open(out, "w"))
    print(f"ZORO {max_px or 'default'}: {len(res)} imgs in {wall:.0f}s ({wall / len(res) * 1000:.0f} ms/img at 8 concurrent), one-at-a-time median {statistics.median(seq) * 1000:.0f} ms -> {out}", flush=True)
