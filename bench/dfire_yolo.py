"""Run D-Fire-trained YOLO detectors on the same D-Fire test images. Image-level max confidence per class.

Usage: ~/venvs/bench/bin/python ~/dfire_yolo.py
"""
import json, statistics, time

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import os, torch
if os.environ.get('NO_CUDNN'):
    torch.backends.cudnn.enabled = False  # GPU shared with the Zoro server: too little free memory for cuDNN engines

ROOT = "/ephemeral/crisis_bench/dfire"
gold = json.load(open(f"{ROOT}/gold.json"))
names = sorted(gold)
MODELS = {"yolov8n_dfire": ("rabahdev/fire-smoke-yolov8n", "best.pt"), "yolov8m_dfire_kaggle": ("Namdoji/fire-smoke-yolov8m", None)}
out = {}
for key, (repo, fname) in MODELS.items():
    if fname is None:
        from huggingface_hub import list_repo_files
        fname = next(f for f in list_repo_files(repo) if f.endswith(".pt"))
    model = YOLO(hf_hub_download(repo, fname))
    print(key, fname, model.names, flush=True)
    smoke_ids = [i for i, n in model.names.items() if "smoke" in n.lower()]
    fire_ids = [i for i, n in model.names.items() if "fire" in n.lower() or "flame" in n.lower()]
    for n in names[:5]:
        model.predict(f"{ROOT}/images/{n}", conf=0.01, verbose=False)
    preds, lat = {}, []
    t0 = time.time()
    for n in names:
        t = time.time()
        r = model.predict(f"{ROOT}/images/{n}", conf=0.01, imgsz=640, verbose=False)[0]
        lat.append(time.time() - t)
        cls, conf = r.boxes.cls.tolist(), r.boxes.conf.tolist()
        preds[n] = {"smoke": round(max([c for k, c in zip(cls, conf) if int(k) in smoke_ids], default=0.0), 4),
                    "flame": round(max([c for k, c in zip(cls, conf) if int(k) in fire_ids], default=0.0), 4)}
        preds[n]["alarm"] = max(preds[n]["smoke"], preds[n]["flame"])
    out[key] = {"repo": repo, "file": fname, "median_ms": round(statistics.median(lat) * 1000, 1), "wall": round(time.time() - t0, 1), "preds": preds}
    print(f"YOLO {key}: median {statistics.median(lat) * 1000:.1f} ms/img (batch 1, incl. decode)", flush=True)
json.dump(out, open(f"{ROOT}/yolo.json", "w"))
print("YOLO_DONE")
