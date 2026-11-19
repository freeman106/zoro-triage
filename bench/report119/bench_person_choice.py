"""Fit once on frozen train IDs, then score a single locked 400-report holdout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sys
from urllib.request import Request,urlopen

from person_choice import QUESTION,choose,validate
from bench_life_gate import POSITIVE,read_records
from collections import Counter

def write_new(path,data):
    with path.open("x") as out:
        json.dump(data,out,indent=2,ensure_ascii=False)

def measure(rows,pred,photo,tau):
    tp=fp=fn=tops=right=0
    counts=Counter()
    for r in rows:
        k=r["id"]
        state=choose(pred[k]["probs"],tau)
        life=state=="life_threat"
        gold=r["people_detail"] in POSITIVE
        gold_state="life_threat" if gold else {"at_risk":"trapped","none":"none","injured":"minor_injury"}[r["people"]]
        right+=state==gold_state
        counts[state]+=1
        tp+=life and gold
        fp+=life and not gold
        fn+=not life and gold
        tops+=pred[k]["now"]>=.5 and (life or (state=="trapped" and photo[k]["probs"]["ongoing_hazard"]>=.5))
    return dict(n=len(rows),threshold=tau,tp=tp,fp=fp,fn=fn,recall=tp/(tp+fn),precision=tp/(tp+fp) if tp+fp else 0,
                top=tops,top_rate=tops/len(rows),correct_state=right,counts=dict(counts))

def run(args):
    out=args.out
    sys.path.insert(0,str(args.demo.resolve()))
    import decision
    rows=json.loads(args.data.read_text())
    train=args.phase=="train"
    photo_path=args.archive/(args.phase+"-predictions.jsonl")
    photo=read_records(photo_path)
    sig_paths=[Path(__file__),Path(__file__).with_name("person_choice.py"),Path(__file__).with_name("bench_life_gate.py"),
               args.data,args.demo/"decision.py",args.demo/"mismatch_types.py",
               args.archive/"train-predictions.jsonl",args.archive/"test-predictions.jsonl"]
    sig={p.name:sha256(p.read_bytes()).hexdigest() for p in sig_paths}
    dest=out/("frozen.json" if train else "holdout.json")
    if dest.exists():
        raise RuntimeError("Stage already finalized")
    meta=out/"inputs.json"
    if not meta.exists():
        write_new(meta,sig)
    assert json.loads(meta.read_text())==sig
    frozen=None if train else json.loads((out/"frozen.json").read_text())
    if frozen:
        assert frozen["signature"]==sig
    selected=[r for r in rows if r["split"]==args.phase and r["id"] in photo]
    ids={r["id"] for r in selected}
    assert ids==set(photo) and len(selected)==len(ids)==(700 if train else 400)
    assert len({v["model"] for v in photo.values()})==1
    model=next(iter(photo.values()))["model"]
    if frozen: assert model==frozen["model"]
    questions={"person_state":QUESTION,"now":decision._noul(decision.QUESTIONS["now"]),"claim_kind":decision.TEXT_QUESTION}
    def predict(r):
        body={"model":"zoro-1","state":{"신고":r["text"]},"questions":questions}
        req=Request("http://127.0.0.1:30001/v1/systemone",json.dumps(body,ensure_ascii=False).encode(),{"Content-Type":"application/json"})
        with urlopen(req,timeout=12) as response: res=json.load(response)
        a=res["answers"]
        now=a["now"]["noul"]
        assert res["model"]==model and type(now) in (float,int) and 0<=now<=1
        probs=validate(a["person_state"]["probabilities"])
        return dict(id=r["id"],probs=probs,now=now,claim=a["claim_kind"]["choice"],raw_choice=a["person_state"]["choice"])
    path=out/(args.phase+".jsonl")
    pred=read_records(path) if path.exists() else {}
    with ThreadPoolExecutor(4) as pool,path.open("a") as f:
        for v in pool.map(predict,[r for r in selected if r["id"] not in pred]):
            pred[v["id"]]=v
            f.write(json.dumps(v)+"\n")
            f.flush()
            if len(pred)%100==0:print(args.phase,len(pred),flush=True)
    assert set(pred)==ids
    if train:
        curve=[measure(selected,pred,photo,i/100) for i in range(101)]
        def key(m):
            return (round(min(m["recall"]-.90,.40-m["top_rate"]),10),m["precision"],m["recall"],m["threshold"])
        best=max(curve,key=key)
        write_new(dest,dict(gate="recall>=0.90 AND top_rate<=0.40",selection="max min(recall-.90,.40-top_rate); ties precision,recall,threshold",
                            selected=best,grid=curve,signature=sig,model=model))
        print("FROZEN",best,flush=True)
    else:
        m=measure(selected,pred,photo,frozen["selected"]["threshold"])
        m["adopt"]=10*m["tp"]>=9*(m["tp"]+m["fn"]) and 5*m["top"]<=2*m["n"]
        write_new(dest,m)
        print("HOLDOUT",m,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["train","test"])
    for name in ("demo","data","archive","out"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    with (args.out/"lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
