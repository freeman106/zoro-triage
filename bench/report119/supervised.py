"""Supervised learning curves. Run once per method with an idle GPU."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPModel

from evaluate import ROOT, summary, urgency

B32 = "openai/clip-vit-base-patch32"
MULTI = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
KC = "beomi/KcELECTRA-base-v2022"
GROUPS = ["fire", "storm", "quake"]
EVENTS = {"california_wildfires": "fire", "mexico_earthquake": "quake", "iraq_iran_earthquake": "quake",
          "hurricane_harvey": "storm", "hurricane_irma": "storm", "hurricane_maria": "storm", "srilanka_floods": "storm"}


class Photos:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(B32, dtype=torch.float16).cuda().eval()
        self.proc = AutoProcessor.from_pretrained(B32)

    @torch.inference_mode()
    def __call__(self, rows):
        imgs = []
        for row in rows:
            with Image.open(ROOT / row["image_path"]) as f:
                imgs.append(f.convert("RGB"))
        pixels = self.proc(images=imgs, return_tensors="pt")["pixel_values"].cuda().half()
        feats = self.model.get_image_features(pixel_values=pixels)
        vec = feats if torch.is_tensor(feats) else feats.pooler_output
        return nn.functional.normalize(vec.float(), dim=1).cpu().numpy()


class TextClip:
    def __init__(self):
        self.model = AutoModel.from_pretrained(MULTI).cuda().eval()
        self.tok = AutoTokenizer.from_pretrained(MULTI)
        path = hf_hub_download(MULTI, "2_Dense/model.safetensors", local_files_only=True)
        self.weights = load_file(path)["linear.weight"].cuda()

    @torch.inference_mode()
    def __call__(self, rows):
        x = tokens(self.tok, rows)
        h = self.model(**x).last_hidden_state
        mask = x.attention_mask.unsqueeze(-1)
        mean = (h*mask).sum(1) / mask.sum(1)
        return nn.functional.normalize(nn.functional.linear(mean, self.weights), dim=1).cpu().numpy()


def tokens(tok, rows):
    return tok([r["text"] for r in rows], truncation=True, max_length=128, padding=True, return_tensors="pt").to("cuda")


def batched(fn, rows, size=24):
    return np.concatenate([fn(rows[i:i+size]) for i in range(0,len(rows),size)])


def fit_kc(rows):
    tok = AutoTokenizer.from_pretrained(KC)
    model = AutoModel.from_pretrained(KC).cuda().train()
    head = nn.Linear(model.config.hidden_size, 7).cuda()
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": 3e-5},
                             {"params": head.parameters(), "lr": 5e-4}])
    rng = random.Random(42)
    def logits(batch):
        return head(model(**tokens(tok, batch)).last_hidden_state[:,0]).split([2,2,3], dim=1)
    for _ in range(max(4, (60 + (len(rows)+15)//16 - 1)//((len(rows)+15)//16))):
        order = rng.sample(rows, len(rows))
        for i in range(0,len(rows),16):
            b = order[i:i+16]
            ys = [[int(r["people"] != "none") for r in b], [int(r["time"] == "now") for r in b],
                  [GROUPS.index(r["claim_group"]) for r in b]]
            loss = sum(nn.functional.cross_entropy(out, torch.tensor(y, device="cuda")) for out,y in zip(logits(b),ys))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    @torch.inference_mode()
    def predict(batch):
        return torch.stack([out.argmax(1) for out in logits(batch)], dim=1).cpu().numpy()
    return predict


def measure(fn, test):
    fn(test[0])
    pred = {}
    times = []
    for r in test:
        t = time.perf_counter()
        level, mismatch = fn(r)
        times.append((time.perf_counter()-t)*1000)
        pred[r["id"]] = {"urgency": str(level), "mismatch": bool(mismatch)}
    return pred, statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["mlp","kc"])
    mode = p.parse_args().mode
    rows = json.loads(Path("scenarios.json").read_text())
    train, test = [[r for r in rows if r["split"] == split] for split in ("train","test")]
    assert len(train) == 1500 and len(test) == 400 and all(r["text"] for r in rows)
    order = random.Random(42).sample(train, len(train))
    photos = Photos()
    if mode == "mlp":
        text_clip = TextClip()
        def joint(b):
            return np.hstack([photos(b), text_clip(b)])
        features = batched(joint, order)
    else:
        raw = (ROOT/"damage/train.json").read_text()
        try:
            real = json.loads(raw)
        except ValueError:
            real = list(map(json.loads, raw.splitlines()))
        real = [r for r in real if (ROOT/r["image_path"]).is_file()]
        assert not {r["image_id"] for r in real} & {r["image_id"] for r in test}
        features = batched(photos, real)
        severe = LogisticRegression(C=4, max_iter=1000).fit(features,[r["label"]=="severe_damage" for r in real])
        group = LogisticRegression(C=4, max_iter=1000).fit(features,[EVENTS[r["event_name"]] for r in real])
        print("CrisisMMD supervised photographs",len(real),flush=True)

    for n in (50,200,1500):
        torch.manual_seed(42)
        b = order[:n]
        if mode == "mlp":
            valid = [i for i,r in enumerate(b) if not r["gold_mismatch"]]
            def mlp():
                return MLPClassifier(hidden_layer_sizes=(128,),solver="lbfgs",alpha=0.1,max_iter=300,random_state=42)
            level = mlp().fit(features[valid], [b[i]["gold_urgency"] for i in valid])
            mismatch = mlp().fit(features[:n], [r["gold_mismatch"] for r in b])
            def predict(r):
                x = joint([r])
                return level.predict(x)[0], mismatch.predict(x)[0]
        else:
            kc = fit_kc(b)
            def predict(r):
                risk, now, claim = kc([r])[0]
                x = photos([r])
                return urgency(risk, severe.predict(x)[0], now), GROUPS[claim] != group.predict(x)[0]
        preds, ms = measure(predict,test)
        result = dict(method=mode, n_train=n, median_ms=ms, **summary(test,preds), train_ids=[r["id"] for r in b], preds=preds)
        print(mode,n,summary(test,preds),"median_ms",ms,flush=True)
        Path(f"supervised119-{mode}-{n}.json").write_text(json.dumps(result))
        if mode == "kc":
            del kc
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
