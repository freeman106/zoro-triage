"""Calibrate on train, freeze the threshold, then score the complete test set once."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import random
from urllib.request import Request, urlopen

from mismatch_types import TEXT_QUESTION, PHOTO_QUESTION, extract, is_mismatch

SEED = 119
TRAIN_N = 500
GRID = [n / 100 for n in range(25, 100, 5)]
OUT = Path("mismatch119-v1")
URL = "http://127.0.0.1:30001/v1/systemone"


def score(pred, gold):
    tp = sum(p and g for p,g in zip(pred,gold))
    fp = sum(pred) - tp
    fn = sum(gold) - tp
    return dict(tp=tp, fp=fp, fn=fn, tn=len(gold)-tp-fp-fn,
                precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0,
                f05=5*tp/(5*tp+4*fp+fn) if tp+fp+fn else 0)


def call(key, question, state, image=None):
    body = {"model":"zoro-1", "state":state, "questions":{key:question}}
    if image:
        body["images"] = [image]
    req = Request(URL,json.dumps(body,ensure_ascii=False).encode(),{"Content-Type":"application/json"})
    with urlopen(req,timeout=30) as res:
        result = json.load(res)
    return extract(result["answers"][key],question),result["model"]


def predict(row):
    image = base64.b64encode((Path("crisismmd")/row["image_path"]).read_bytes()).decode()
    text, model = call("claim_kind",TEXT_QUESTION,{"신고":row["text"]})
    photo, image_model = call("photo_kind",PHOTO_QUESTION,"첨부 사진",image)
    if model != image_model:
        raise ValueError("Model changed")
    return dict(id=row["id"],text=text,photo=photo,model=model)


def save(path, data):
    with path.open("x") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("split",choices=["train","test"])
    split=arg.parse_args().split
    data = Path("scenarios.json")
    signature = {p.name:sha256(p.read_bytes()).hexdigest() for p in (data,Path(__file__),Path(__file__).with_name("mismatch_types.py"))}
    rows=json.loads(data.read_text())
    test=[r for r in rows if r["split"]=="test"]
    posts={r["image_id"].rsplit("_",1)[0] for r in test}
    pool=sorted((r for r in rows if r["split"]=="train" and r["image_id"].rsplit("_",1)[0] not in posts),key=lambda r:r["id"])
    OUT.mkdir(exist_ok=True)
    config=OUT/"mismatch_config.json"
    if split=="train":
        selected=random.Random(SEED).sample(pool,TRAIN_N)
        if config.exists():
            raise RuntimeError("Calibration already frozen")
    else:
        freeze=json.loads(config.read_text())
        if freeze["signature"]!=signature:
            raise RuntimeError("Frozen inputs have changed")
        selected=test
        assert len(selected)==len({r["id"] for r in selected})==400
    destination=OUT/(split+"-predictions.jsonl")
    # Inference is resumable per ID. No test labels or scores are emitted during inference.
    raw={}
    if destination.exists():
        raw={v["id"]:v for v in map(json.loads,destination.read_text().splitlines())}
    remaining=[r for r in selected if r["id"] not in raw]
    with ThreadPoolExecutor(4) as executor, destination.open("a") as f:
        for record in executor.map(predict,remaining):
            raw[record["id"]]=record
            f.write(json.dumps(record)+"\n")
            f.flush()
            if len(raw)%100==0:
                print(split,len(raw),flush=True)
    assert set(raw)=={r["id"] for r in selected} and len({v["model"] for v in raw.values()})==1
    def metrics(t):
        return score([is_mismatch(raw[r["id"]]["text"],raw[r["id"]]["photo"],t) for r in selected],
                     [r["gold_mismatch"] for r in selected])
    if split=="train":
        curve=[dict(threshold=t,**metrics(t)) for t in GRID]
        best=max(curve,key=lambda x:(x["f05"],x["recall"],x["threshold"]))
        save(config,dict(threshold=best["threshold"],seed=SEED,n=TRAIN_N,signature=signature,grid=curve,
                         model=next(iter(raw.values()))["model"],gate="precision > 80/121 and F0.5 > legacy F0.5"))
        print("FROZEN",best)
    else:
        output=OUT/"final-test.json"
        if output.exists():
            raise RuntimeError("Test already scored")
        assert freeze["model"]==next(iter(raw.values()))["model"]
        gold=[r["gold_mismatch"] for r in selected]
        old_path,=Path.cwd().glob("pred119-parts-*.jsonl")
        old={r["id"]:r["mismatch"] for r in map(json.loads,old_path.read_text().splitlines())}
        old_metrics=score([old[r["id"]] for r in selected],gold)
        shortcut=score([r["claim_group"]=="fire" for r in selected],gold)
        new=metrics(freeze["threshold"])
        result=dict(n=len(selected),threshold=freeze["threshold"],legacy=old_metrics,claim_only=shortcut,
                    separated=new,adopt=new["precision"]>shortcut["precision"] and new["f05"]>old_metrics["f05"])
        save(output,result)
        print("FINAL",result)

if __name__=="__main__":
    main()
