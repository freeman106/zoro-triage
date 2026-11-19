"""Compare Zoro-1 and D-Fire-trained YOLO detectors at image level on the D-Fire test split."""
import glob, json, os

ROOT = "/ephemeral/crisis_bench/dfire"
gold = json.load(open(f"{ROOT}/gold.json"))
names = sorted(gold)
systems = {}
for p in sorted(glob.glob(f"{ROOT}/zoro_*.json")):
    d = json.load(open(p))
    systems[f"Zoro ({'448px' if not d['max_px'] else '~1000px'})"] = (d["preds"], f"{d['seq_ms_median']} ms")
if os.path.exists(f"{ROOT}/yolo.json"):
    for k, d in json.load(open(f"{ROOT}/yolo.json")).items():
        systems[f"YOLO {k}"] = (d["preds"], f"{d['median_ms']} ms")


def cat(g):
    return "fire+smoke" if g["fire"] and g["smoke"] else "smoke only" if g["smoke"] else "fire only" if g["fire"] else "none"


def auc(pos, neg):
    import bisect
    neg = sorted(neg)
    s = 0.0
    for p in pos:
        lo, hi = bisect.bisect_left(neg, p), bisect.bisect_right(neg, p)
        s += lo + 0.5 * (hi - lo)
    return s / (len(pos) * len(neg))


cats = ["fire+smoke", "smoke only", "fire only", "none"]
by = {c: [n for n in names if cat(gold[n]) == c] for c in cats}
small = [n for n in by["smoke only"] if gold[n]["max_smoke_area"] < 0.01]
large = [n for n in by["smoke only"] if gold[n]["max_smoke_area"] >= 0.01]
print(f"test images: {len(names)}  " + "  ".join(f"{c} {len(by[c])}" for c in cats) + f"  (smoke-only small<1% area {len(small)}, large {len(large)})")

summary = {}
for sys_name, (pred, ms) in systems.items():
    s = lambda n: pred[n]["alarm"]
    pos = [s(n) for n in names if cat(gold[n]) != "none"]
    neg = [s(n) for n in by["none"]]
    a = auc(pos, neg)
    print(f"\n== {sys_name}   per-image latency {ms}   alarm AUC {a:.3f}")
    # operating points: own default threshold, and threshold fixed so that 5% / 2% of no-fire images alarm
    neg_sorted = sorted(neg, reverse=True)
    points = {"default thr": 0.5 if sys_name.startswith("Zoro") else 0.25}
    for fpr in (0.05, 0.02):
        points[f"FPR {int(fpr * 100)}%"] = neg_sorted[int(len(neg) * fpr)] + 1e-9
    rows = {}
    for label, thr in points.items():
        rec = {c: sum(s(n) >= thr for n in by[c]) / len(by[c]) for c in cats}
        rec["small smoke"] = sum(s(n) >= thr for n in small) / len(small)
        rec["large smoke"] = sum(s(n) >= thr for n in large) / len(large)
        allpos = [n for n in names if cat(gold[n]) != "none"]
        tp = sum(s(n) >= thr for n in allpos)
        fp = sum(s(n) >= thr for n in by["none"])
        prec = tp / max(1, tp + fp)
        recall = tp / len(allpos)
        f1 = 2 * prec * recall / max(1e-9, prec + recall)
        rows[label] = {"thr": round(thr, 4), "recall_all": recall, "precision": prec, "f1": f1, **rec}
        print(f"  {label:12s} thr {thr:.3f} | recall all {recall:.3f}  fire+smoke {rec['fire+smoke']:.3f}  smoke-only {rec['smoke only']:.3f}"
              f" (small {rec['small smoke']:.3f} / large {rec['large smoke']:.3f})  fire-only {rec['fire only']:.3f} | false alarm {rec['none']:.3f}  F1 {f1:.3f}")
    summary[sys_name] = {"auc": a, "latency": ms, "points": rows}
json.dump(summary, open(f"{ROOT}/summary.json", "w"), indent=1)
