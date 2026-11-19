"""Dedicated baselines on the same CrisisMMD test items: CLIP ViT-L/14 features + logistic regression
(image / text / image+text), and TF-IDF + logistic regression on tweet text (full train split).

Usage: ~/venvs/bench/bin/python ~/crisis_probe.py   -> /ephemeral/crisis_bench/crisismmd/probe.json
"""
import json, os, statistics, time

import numpy as np
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import CLIPModel, CLIPProcessor

R = "/ephemeral/crisis_bench/crisismmd"
dev = "cuda"
torch.backends.cudnn.enabled = False  # GPU shared with the Zoro server: too little free memory for cuDNN engines
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(dev).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def rows(task, split):
    txt = open(f"{R}/{task}/{split}.json").read()
    try:
        return json.loads(txt)
    except Exception:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


cache = {}


def vec(o):
    """Newer transformers return a ModelOutput from get_*_features; older ones return the tensor."""
    return o if torch.is_tensor(o) else o.pooler_output


@torch.no_grad()
def feats(items):
    todo = [x for x in items if x["image_id"] not in cache]
    for i in range(0, len(todo), 32):
        b = todo[i:i + 32]
        ims = [Image.open(f"{R}/{x['image_path']}").convert("RGB") for x in b]
        inp = proc(images=ims, text=[x["tweet_text"] for x in b], return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
        fi = vec(model.get_image_features(pixel_values=inp["pixel_values"].half()))
        ft = vec(model.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]))
        fi = torch.nn.functional.normalize(fi.float(), dim=-1).cpu().numpy()
        ft = torch.nn.functional.normalize(ft.float(), dim=-1).cpu().numpy()
        for x, a, t in zip(b, fi, ft):
            cache[x["image_id"]] = (a, t)
    img = np.stack([cache[x["image_id"]][0] for x in items])
    txt = np.stack([cache[x["image_id"]][1] for x in items])
    return img, txt


# latency of a single-image CLIP-L forward (what a deployed image classifier costs per request)
probe_img = Image.open(f"{R}/{rows('damage', 'test')[0]['image_path']}").convert("RGB")
with torch.no_grad():
    for _ in range(5):
        model.get_image_features(pixel_values=proc(images=[probe_img], return_tensors="pt")["pixel_values"].to(dev).half())
    lat = []
    for _ in range(50):
        torch.cuda.synchronize(); t = time.time()
        model.get_image_features(pixel_values=proc(images=[probe_img], return_tensors="pt")["pixel_values"].to(dev).half())
        torch.cuda.synchronize(); lat.append(time.time() - t)
print(f"CLIP-L/14 single image forward median {statistics.median(lat) * 1000:.1f} ms", flush=True)

out = {"clip_ms": round(statistics.median(lat) * 1000, 1)}
for task in ("informative", "humanitarian", "damage"):
    train_all = rows(task, "train")
    train = [x for x in train_all if os.path.exists(f"{R}/{x['image_path']}")]
    test = [x for x in rows(task, "test") if os.path.exists(f"{R}/{x['image_path']}")]
    ytr, yte = [x["label"] for x in train], [x["label"] for x in test]
    itr, ttr = feats(train)
    ite, tte = feats(test)
    res = {}
    for mode, Xtr, Xte in (("image", itr, ite), ("text_clip", ttr, tte), ("both", np.hstack([itr, ttr]), np.hstack([ite, tte]))):
        clf = LogisticRegression(max_iter=3000, C=2.0).fit(Xtr, ytr)
        p = clf.predict(Xte)
        res[mode] = {"acc": accuracy_score(yte, p), "macro_f1": f1_score(yte, p, average="macro"), "weighted_f1": f1_score(yte, p, average="weighted"),
                     "preds": dict(zip([x["image_id"] for x in test], p.tolist())), "n_train": len(train)}
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = tf.fit_transform([x["tweet_text"] for x in train_all])
    clf = LogisticRegression(max_iter=3000, C=4.0).fit(Xtr, [x["label"] for x in train_all])
    p = clf.predict(tf.transform([x["tweet_text"] for x in test]))
    res["text_tfidf"] = {"acc": accuracy_score(yte, p), "macro_f1": f1_score(yte, p, average="macro"), "weighted_f1": f1_score(yte, p, average="weighted"),
                         "preds": dict(zip([x["image_id"] for x in test], p.tolist())), "n_train": len(train_all)}
    out[task] = res
    print(task, {m: (round(r["acc"], 3), round(r["macro_f1"], 3), r["n_train"]) for m, r in res.items()}, flush=True)
json.dump(out, open(f"{R}/probe.json", "w"))
print("CRISIS_PROBE_DONE")
