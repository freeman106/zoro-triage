"""Recompute all reported scores from the per-report predictions."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from evaluate import summary, urgency

def nested_boolean(raw, key):
    # Accept one explicit answer, or a lone boolean substituted for "type".
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[raw.index("{"):])
    except ValueError:
        return None
    expected = {"people", "now"} if key in ("people", "now") else {key}
    if not isinstance(obj, dict) or not set(obj) <= expected:
        return None
    value = obj.get(key)
    if not isinstance(value, dict):
        return None
    if set(value) == {"type"} and isinstance(value["type"], bool):
        return value["type"]
    fields = value.keys() & {"answer", "value", "result", "noul"}
    if len(fields) != 1 or any(isinstance(v, bool) for k, v in value.items() if k not in fields):
        return None
    answer = value[next(iter(fields))]
    return answer if isinstance(answer, bool) else None


def component_coverage(test, preds):
    return {key: sum(preds[r["id"]][key] is not None for r in test)
            for key in ("people", "now", "severe", "mismatch")}


def format_sensitivity(test, preds):
    relaxed = {}
    for key, pred in preds.items():
        row = pred.copy()
        for field, raw in row.get("invalid_raw", {}).items():
            row[field] = nested_boolean(raw, field)
        flags = [row[k] for k in ("people", "severe", "now")]
        if None not in flags:
            row["urgency"] = urgency(*flags)
        relaxed[key] = row
    return dict(summary(test, relaxed), valid_fields=component_coverage(test, relaxed),
                valid_urgency=sum(relaxed[r["id"]]["urgency"] != "invalid" for r in test if not r["gold_mismatch"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-signature", help="Use this run for direct, parts and Qwen direct")
    parser.add_argument("--direct-times", type=Path, help="Isolated Zoro direct latency sample for the selected test set")
    parser.add_argument("--legacy", action="store_true", help="Summarize the three original zero-shot settings")
    args = parser.parse_args()
    data=json.loads(Path("scenarios.json").read_text())
    test=[r for r in data if r["split"]=="test"]
    train=[r for r in data if r["split"]=="train"]
    names={r["id"] for r in test}
    result={}
    signature = hashlib.sha256(json.dumps(test, sort_keys=True).encode() +
                               Path(__file__).with_name("evaluate.py").read_bytes()).hexdigest()[:12]
    reference = args.reference_signature or signature
    modes = ["direct", "parts", "qwen"]
    if not args.legacy:
        modes.append("qwen_parts")
    for mode in modes:
        run = signature if mode == "qwen_parts" else reference
        path = Path(f"pred119-{mode}-{run}.jsonl")
        values=list(map(json.loads,path.read_text().splitlines()))
        preds={a["id"]:a for a in values}
        assert len(values)==400 and set(preds)==names
        times = [a["ms"] for a in values]
        if mode == "direct" and args.direct_times:
            times = json.loads(args.direct_times.read_text())
        assert len(times) == len(test) and all(t > 0 for t in times)
        result[mode]=dict(summary(test,preds),median_ms=median(times))
        if mode in ("parts","qwen_parts"):
            for k in ("severe","people","now"):
                truth={"severe":[r["image_severe"] for r in test],"people":[r["people"]!="none" for r in test],
                       "now":[r["time"]=="now" for r in test]}[k]
                result[mode][k+"_acc"]=sum(preds[r["id"]][k]==y for r,y in zip(test,truth))/len(test)
        if mode == "qwen_parts":
            result[mode]["valid_fields"] = component_coverage(test, preds)
            result[mode]["valid_urgency"] = sum(preds[r["id"]]["urgency"] != "invalid" for r in test if not r["gold_mismatch"])
            result["qwen_parts_nested_boolean"] = format_sensitivity(test, preds)
    for mode in ("mlp","kc"):
        for n in (50,200,1500):
            d=json.loads(Path(f"supervised119-{mode}-{n}.json").read_text())
            assert set(d["preds"])==names and len(set(d["train_ids"]))==n
            result[f"{mode}-{n}"]=dict(summary(test,d["preds"]),median_ms=d["median_ms"])
    groups={r["claim_group"] for r in train}
    majority={g:sum(r["gold_mismatch"] for r in train if r["claim_group"]==g)>sum(r["claim_group"]==g for r in train)/2 for g in groups}
    preds={r["id"]:{"urgency":"normal","mismatch":majority[r["claim_group"]]} for r in test}
    diag=summary(test,preds)
    result["claim_slot_only"]=dict(rule=majority,precision=diag["precision"],recall=diag["recall"])
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    text_only = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2,4), min_df=2),
                              LogisticRegression(C=4, max_iter=500))
    text_only.fit([r["text"] for r in train],[r["gold_mismatch"] for r in train])
    flags = text_only.predict([r["text"] for r in test])
    for r,flag in zip(test,flags):
        preds[r["id"]]["mismatch"] = bool(flag)
    diag = summary(test,preds)
    result["text_only_mismatch_1500"] = dict(precision=diag["precision"],recall=diag["recall"])
    print(json.dumps(result,ensure_ascii=False,indent=2))
    Path("report119-results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
