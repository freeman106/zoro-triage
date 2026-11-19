"""Tune each photo-severity variant's threshold on train images only, then score test images once and
recompute old-policy urgency on the 303 consistent test reports (same gold as the KcELECTRA 68.0% figure)."""
import json, statistics, sys

RES, TEST, PAIRED = sys.argv[1:4]
res = json.load(open(RES))
gold = {s["id"]: s for s in json.load(open(TEST))}
paired = {r["id"]: r["after"] for r in map(json.loads, open(PAIRED))}
PARTS = ["collapse", "roof", "road", "vehicle", "debris"]

score = {
    "v0 current": lambda r: r["v0"],
    "v1 any part": lambda r: max(r[k] for k in PARTS),
    "v1 mean part": lambda r: statistics.mean(r[k] for k in PARTS),
    "v2 3-level": lambda r: r["v2"],
    "v3 hi-res": lambda r: r["v3"],
    "avg v0+v2+v3": lambda r: (r["v0"] + r["v2"] + r["v3"]) / 3,
    "avg all": lambda r: (r["v0"] + r["v2"] + r["v3"] + max(r[k] for k in PARTS)) / 4,
}
train = [r for r in res.values() if r["split"] == "train"]
test_img = {p: r for p, r in res.items() if r["split"] == "test"}
cons = [i for i in gold if not gold[i]["gold_mismatch"]]


def old_policy(now, people, severe):
    if not now:
        return "non_urgent"
    return ("normal", "urgent", "top")[int(people) + int(severe)]


base = sum(old_policy(paired[i]["now"], paired[i]["people"], paired[i]["severe"]) == gold[i]["gold_urgency"] for i in cons) / len(cons)
print(f"baseline (demo as deployed): urgency {base:.3f}   [KcELECTRA+CLIP, 50 reports + 2,468 photo labels: 0.680]")
print(f"{'variant':16s} {'thr':>5s} {'train acc':>9s} {'test img acc':>12s} {'test scen acc':>13s} {'urgency':>8s}")
for name, f in score.items():
    best = max((sum((f(r) >= t) == r["gold_severe"] for r in train) / len(train), t) for t in [x / 100 for x in range(5, 96)])
    tr_acc, thr = best
    img_acc = sum((f(r) >= thr) == r["gold_severe"] for r in test_img.values()) / len(test_img)
    scen_acc = sum((f(res[gold[i]["image_path"]]) >= thr) == gold[i]["image_severe"] for i in gold) / len(gold)
    urg = sum(old_policy(paired[i]["now"], paired[i]["people"], f(res[gold[i]["image_path"]]) >= thr) == gold[i]["gold_urgency"] for i in cons) / len(cons)
    print(f"{name:16s} {thr:5.2f} {tr_acc:9.3f} {img_acc:12.3f} {scen_acc:13.3f} {urg:8.3f}")
