"""Choose the escalation band and threshold on train images only; score test once."""
import json, statistics, sys

RES, REA, TEST, PAIRED = sys.argv[1:5]
res, rea = json.load(open(RES)), json.load(open(REA))
gold = {s["id"]: s for s in json.load(open(TEST))}
paired = {r["id"]: r["after"] for r in map(json.loads, open(PAIRED))}
cons = [i for i in gold if not gold[i]["gold_mismatch"]]
src = [r["source"] for r in rea.values()]
print("reasoned answers:", sum(s == "reasoned" for s in src), "/", len(src), " cut:", sum(bool(r["cut"]) for r in rea.values()),
      " median ms:", statistics.median(r["ms"] for r in rea.values()))


def p_final(path, lo, hi, mode):
    d = res[path]["v2"]
    if lo <= d <= hi and path in rea:
        r = rea[path]["p"]
        return r if mode == "replace" else (d + r) / 2
    return d


def old_policy(now, people, severe):
    return "non_urgent" if not now else ("normal", "urgent", "top")[int(people) + int(severe)]


train = [p for p, r in res.items() if r["split"] == "train"]
best = None
for mode in ("replace", "average"):
    for lo in (0.15, 0.2, 0.25, 0.3, 0.35):
        for hi in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
            for thr in [x / 100 for x in range(20, 81, 2)]:
                acc = sum((p_final(p, lo, hi, mode) >= thr) == res[p]["gold_severe"] for p in train) / len(train)
                if best is None or acc > best[0]:
                    best = (acc, mode, lo, hi, thr)
acc, mode, lo, hi, thr = best
test = [p for p, r in res.items() if r["split"] == "test"]
esc = sum(lo <= res[p]["v2"] <= hi for p in test) / len(test)
img = sum((p_final(p, lo, hi, mode) >= thr) == res[p]["gold_severe"] for p in test) / len(test)
urg = sum(old_policy(paired[i]["now"], paired[i]["people"], p_final(gold[i]["image_path"], lo, hi, mode) >= thr) == gold[i]["gold_urgency"] for i in cons) / len(cons)
print(f"chosen on train: mode={mode} band=[{lo},{hi}] thr={thr}  train acc {acc:.3f}")
print(f"test: photo acc {img:.3f}  escalated share {esc:.1%}  urgency {urg:.3f}   (direct 3-level: 0.690 / 0.667, KcELECTRA 0.680)")
