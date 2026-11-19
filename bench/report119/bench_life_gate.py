"""Freeze a train-only threshold then score one complete holdout, never tune on test."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

PROMPT = "글에 의식 없음, 호흡 곤란, 깔림, 심한 출혈처럼 지금 목숨이 위태로운 사람이 명시되어 있는가? 고립·갇힘·대피 불가·연락 두절만으로는 아니오. 글에 없는 내용은 추측하지 마."
POSITIVE = {"사람이 쓰러져서 의식이 없음", "연기를 마신 사람이 숨을 잘 못 쉼", "건물 잔해에 깔린 사람이 있음", "가족 한 명이 다리를 다쳐 피를 흘림"}
N = 700
SEED = 119
CUTS = [i/100 for i in range(101)]


def top(probs, tau):
    return probs["now"] >= .5 and (probs["life_threat"] >= tau or
                                   (probs["people"] >= .5 and probs["ongoing_hazard"] >= .5))


def evaluate(rows, decisions, tau):
    tp = fp = fn = total_top = 0
    for row in rows:
        p = decisions[row["id"]]["probs"]
        pred, gold = p["life_threat"] >= tau, row["people_detail"] in POSITIVE
        tp += pred and gold
        fp += pred and not gold
        fn += not pred and gold
        total_top += top(p,tau)
    return dict(threshold=tau, n=len(rows), tp=tp, fp=fp, fn=fn,
                precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn),
                top=total_top, top_rate=total_top/len(rows))


def read_records(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {r["id"]:r for r in rows}
    assert len(result)==len(rows)
    return result


def save(path, value):
    with path.open("x") as f:
        json.dump(value,f,indent=2,ensure_ascii=False)


def missed(ids, levels):
    count=sum(levels[i]!="top" for i in ids)
    return dict(n=len(ids),missed=count,rate=count/len(ids))


def run(args):
    sys.path.insert(0,str(args.demo.resolve()))
    import decision
    decision.QUESTIONS["life_threat"] = PROMPT
    out=args.out
    train=args.phase=="train"
    data=json.loads(args.data.read_text())
    test=[r for r in data if r["split"]=="test"]
    frozen_file=out/"frozen.json"
    output=out/("frozen.json" if train else "holdout.json")
    if output.exists():
        raise SystemExit(f"Already finalized: {output}")
    manifest={str(p.name):sha256(p.read_bytes()).hexdigest() for p in
              [Path(__file__),args.data,*(args.demo/f for f in ("decision.py","scene_question.py","mismatch_types.py","mismatch_config.json"))]}
    if train:
        posts={r["image_id"].rsplit("_",1)[0] for r in test}
        pool=sorted((r for r in data if r["split"]=="train" and r["image_id"].rsplit("_",1)[0] not in posts),key=lambda r:r["id"])
        selected=random.Random(SEED).sample(pool,N)
    else:
        frozen=json.loads(frozen_file.read_text())
        assert manifest==frozen["manifest"]
        selected=test
        assert len(selected)==400
    metadata=out/"inputs.json"
    if not metadata.exists():
        save(metadata,manifest)
    assert json.loads(metadata.read_text())==manifest
    path=out/(args.phase+"-predictions.jsonl")
    results=read_records(path) if path.exists() else {}
    def predict(row):
        photo=(args.images/row["image_path"]).read_bytes()
        result=decision.decide(row["text"],photo,[])
        if result["mode"]!="zoro":
            raise RuntimeError("Model inference failed: "+row["id"])
        return dict(id=row["id"],probs=result["probs"],model=result["model"])
    with ThreadPoolExecutor(3) as pool,path.open("a") as f:
        for record in pool.map(predict,[r for r in selected if r["id"] not in results]):
            results[record["id"]]=record
            f.write(json.dumps(record)+"\n")
            f.flush()
            if len(results)%100==0: print(args.phase,len(results),flush=True)
    assert set(results)=={r["id"] for r in selected} and len({r["model"] for r in results.values()})==1
    model=next(iter(results.values()))["model"]
    if train:
        curve=[evaluate(selected,results,t) for t in CUTS]
        def preference(m):
            return (round(min(m["recall"]-.90,.40-m["top_rate"]),10),m["precision"],m["recall"],m["threshold"])
        best=max(curve,key=preference)
        save(output,dict(prompt=PROMPT,gate={"min_recall":.90,"max_top_rate":.40},model=model,
                         selection="maximize min(recall-.90,.40-top_rate), then precision, recall, threshold",
                         manifest=manifest,seed=SEED,selected=best,grid=curve))
        print("FROZEN",best,flush=True)
        return
    assert frozen["model"]==model
    tau=frozen["selected"]["threshold"]
    candidate=evaluate(test,results,tau)
    keep=candidate["tp"]*10>=9*(candidate["tp"]+candidate["fn"]) and candidate["top"]*5<=2*len(test)
    old=read_records(args.before)
    assert set(old)==set(results) and all(v["mode"]=="zoro" and v["model"]==model for v in old.values())
    old_scores=evaluate(test,old,.5)
    old_levels={i:("top" if top(r["probs"],.5) else "lower") for i,r in old.items()}
    updated={i:("top" if top(r["probs"],tau) else "lower") for i,r in results.items()}
    historical=read_records(args.legacy)
    old83=[r["id"] for r in test if not r["gold_mismatch"] and r["gold_urgency"]=="top"]
    # Under the new rule these positives are known independently of photo ongoing-hazard labels.
    definite=[r["id"] for r in test if not r["gold_mismatch"] and r["time"]=="now" and r["people_detail"] in POSITIVE]
    def miss_table(ids):
        return {name:missed(ids,levels) for name,levels in (
            ("historic", {i:r["urgency"] for i,r in historical.items()}),("previous_live",old_levels),
            ("candidate",updated),("selected_live",updated if keep else old_levels))}
    buckets={}
    for detail in sorted({r["people_detail"] for r in test}):
        ids=[r["id"] for r in test if r["people_detail"]==detail]
        buckets[detail]={"n":len(ids),"before":sum(old[i]["probs"]["life_threat"]>=.5 for i in ids),
                         "candidate":sum(results[i]["probs"]["life_threat"]>=tau for i in ids)}
    final=dict(old=old_scores,candidate=candidate,adopt=keep,old_gold_top=miss_table(old83),new_definite_top=miss_table(definite),
               groups=buckets)
    save(output,final)
    print("HOLDOUT",json.dumps({k:v for k,v in final.items() if k!="groups"}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["train","test"])
    for arg in ("demo","data","images","out","before","legacy"):
        p.add_argument("--"+arg,type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/"lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
