"""Isolate policy changes by replaying identical model outputs without another call."""
from collections import Counter
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"demo119"))
import decision

def main():
    folder=Path(__file__).resolve().parent/"results/bundle-v1"
    old={r["id"]:r for r in map(json.loads,(folder/"before.jsonl").read_text().splitlines())}
    rows=list(map(json.loads,(folder/"after.jsonl").read_text().splitlines()))
    assert len(old)==len(rows)==400 and set(old)=={r["id"] for r in rows}
    changes, overall = Counter(), Counter()
    review_added=0
    conflicts=0
    with (folder/"final-policy.jsonl").open("w") as out:
        for row in rows:
            assert row["mode"]=="zoro"
            evidence={key:row["probs"][key] for key in decision.QUESTIONS}
            evidence.update(claim_kind={"choice":row["claim_kind"],"prob":row["claim_prob"]},
                            photo_kind={"choice":row["photo_kind"],"prob":row["photo_prob"]})
            result=decision._model_result(evidence,row["model"],{})
            key=row["id"]
            changes[row["urgency"],result["urgency"]]+=1
            overall[old[key]["urgency"],result["urgency"]]+=1
            conflicts+=evidence["now"]<.5 and evidence["ongoing_hazard"]>=.5
            review_added+=result["review"] and not row["review"]
            out.write(json.dumps(dict(id=key,urgency=result["urgency"],review=result["review"]))+"\n")
    def counts(matrix):
        return dict(changed=sum(n for (a,b),n in matrix.items() if a!=b),
                    matrix={a:{b:matrix[a,b] for b in levels} for a in levels})
    levels=["top","urgent","normal","non_urgent"]
    result=dict(n=len(rows),same_evidence=counts(changes),original_to_final=counts(overall),
                before_policy={k:sum(v for (a,b),v in changes.items() if a==k) for k in levels},
                final_policy={k:sum(v for (a,b),v in changes.items() if b==k) for k in levels},
                past_with_hazard=conflicts,review_added=review_added)
    (folder/"policy-change.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=="__main__":
    main()
