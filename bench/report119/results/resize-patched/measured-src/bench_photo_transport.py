"""Paired full-engine sweep of original vs area-capped JPEG input on the patched server."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median
import sys

from bench_bundle import load_module, single
from bench_life_gate import POSITIVE

def values(d):
    p=d["probs"]
    return {"people":d["person_state"]!="none","life":d["person_state"]=="life_threat",
            "person_state":d["person_state"],"now":p["now"]>=.5,"severe":p["severe"]>=.5,
            "field_photo":p["field_photo"]>=.5,"ongoing_hazard":p["ongoing_hazard"]>=.5,
            **{k:d[k] for k in ("claim_kind","photo_kind","mismatch","urgency","review")}}

def scores(gold,old,new):
    changed=sum(a!=b for a,b in zip(old,new))
    answer={"n":len(old),"changed":changed}
    if gold is not None:
        for name,pred in (("before",old),("after",new)):
            answer[name]=sum(x==y for x,y in zip(pred,gold))
    return answer

def run(a):
    sys.path.insert(0,str(a.demo.resolve()))
    before=load_module(a.demo/"before.py","before")
    after=load_module(a.demo/"decision.py","after")
    inputs=json.loads(a.data.read_text())
    assert len(inputs)==len({r['id'] for r in inputs})==400
    sig={p.name:sha256(p.read_bytes()).hexdigest() for p in [a.data,Path(__file__),*(a.demo/name for name in (
        "before.py","decision.py","person_choice.py","person_threshold.json","mismatch_types.py","scene_question.py","mismatch_config.json"))]}
    path=a.out/"inputs.json"
    a.out.mkdir(parents=True,exist_ok=True)
    if path.exists():assert json.loads(path.read_text())==sig
    else:path.write_text(json.dumps(sig,indent=2))
    output=a.out/"paired.jsonl"
    records={v["id"]:v for v in map(json.loads,output.read_text().splitlines())} if output.exists() else {}
    order=[True,False]*200
    random.Random(119).shuffle(order)
    for m in (before,after):
        assert single(inputs[0],a.images,m)["mode"]=="zoro"
    with output.open("a") as f:
        for r,old_first in zip(inputs,order):
            if r["id"] in records:continue
            item={"id":r["id"]}
            raw=(a.images/r["image_path"]).read_bytes()
            reduced=after.prepare_photo(raw)
            item.update(raw_bytes=len(raw),transmitted_bytes=len(reduced),resized=raw!=reduced)
            for key,m in ([("before",before),("after",after)] if old_first else [("after",after),("before",before)]):
                d=single(r,a.images,m)
                if d["mode"]!="zoro":raise RuntimeError((r["id"],key,d["reasons"]))
                item[key]=dict(values(d),ms=d["wall_ms"],model=d["model"])
            assert item["before"]["model"]==item["after"]["model"]
            records[r["id"]]=item
            f.write(json.dumps(item)+"\n");f.flush()
            if len(records)%50==0:print("paired",len(records),flush=True)
    assert len(records)==400
    raw_rows=list(records.values())
    kind={"fire":"fire","storm":"storm_flood","quake":"quake_collapse"}
    visual={"fire":"fire_smoke_burn","storm":"water_storm","quake":"collapse"}
    attrs={}
    for key in ("people","life","person_state","now","severe","field_photo","ongoing_hazard","claim_kind","photo_kind","mismatch","urgency","review"):
        gold=[];old=[];new=[]
        for r in inputs:
            if key=="urgency" and r["gold_mismatch"]:continue
            life=r["people_detail"] in POSITIVE
            label={"people":r["people"]!="none","life":life,"person_state":"life_threat" if life else {"at_risk":"trapped","none":"none","injured":"minor_injury"}[r["people"]],
                   "now":r["time"]=="now","severe":r["image_severe"],"claim_kind":kind[r["claim_group"]],
                   "photo_kind":visual[r["image_group"]],"mismatch":r["gold_mismatch"],"urgency":r["gold_urgency"]}.get(key)
            row=records[r["id"]];old.append(row["before"][key]);new.append(row["after"][key]);gold.append(label)
        attrs[key]=scores(None if gold[0] is None else gold,old,new)
    result=dict(n=400,metrics=attrs,ms={k:median(r[k]["ms"] for r in raw_rows) for k in ("before","after")},
                resized=sum(r["resized"] for r in raw_rows),size={k:sum(r[k] for r in raw_rows) for k in ("raw_bytes","transmitted_bytes")})
    (a.out/"summary.json").write_text(json.dumps(result,indent=2))
    print("FINAL",result,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for opt in ("demo","images","data","out"):p.add_argument("--"+opt,type=Path,required=True)
    run(p.parse_args())
