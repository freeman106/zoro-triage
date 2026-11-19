"""Identify the new-policy top miss rate where possible and bound unlabelled visual hazards."""
import argparse
import json
from pathlib import Path
from bench_life_gate import POSITIVE, read_records, top


def rates(rows, old_photo, predicted):
    # A true current life threat is definitely top. Other current people at risk
    # are top only if the unlabelled, ongoing visual hazard is present.
    certain={r["id"] for r in rows if r["time"]=="now" and r["people_detail"] in POSITIVE}
    possible={r["id"] for r in rows if r["time"]=="now" and r["people"]!="none"}-certain
    proxy=certain|{i for i in possible if old_photo[i]["probs"]["ongoing_hazard"]>=.5}
    known_misses=len(certain-predicted)
    unidentified_misses=len(possible-predicted)
    return {"definite_n":len(certain),"definite_missed":known_misses,"possible_n":len(possible),
            "miss_rate_min":known_misses/(len(certain)+len(possible&predicted)),
            "miss_rate_max":(known_misses+unidentified_misses)/(len(certain)+unidentified_misses),
            "proxy_n":len(proxy),"proxy_missed":len(proxy-predicted),"proxy_rate":len(proxy-predicted)/len(proxy)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test",type=Path)
    parser.add_argument("previous",type=Path)
    parser.add_argument("legacy",type=Path)
    parser.add_argument("directory",type=Path)
    a=parser.parse_args()
    d=a.directory
    rows=[r for r in json.loads(a.test.read_text()) if not r["gold_mismatch"]]
    assert len(rows)==303
    baseline=read_records(a.previous)
    historic={i for i,r in read_records(a.legacy).items() if r["urgency"]=="top"}
    previous={i for i,r in baseline.items() if top(r["probs"],.5)}
    decision=json.loads((d/"holdout.json").read_text())
    cut=decision["candidate"]["threshold"]
    proposed={i for i,r in read_records(d/"test-predictions.jsonl").items() if top(r["probs"],cut)}
    final=proposed if decision["adopt"] else previous
    result={k:rates(rows,baseline,v) for k,v in [("historic",historic),("previous_live",previous),("candidate",proposed),("final_live",final)]}
    (d/"miss-bounds.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=="__main__":
    main()
