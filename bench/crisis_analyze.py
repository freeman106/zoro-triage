"""Zoro-1 vs dedicated baselines on the same CrisisMMD test items."""
import json, os
from sklearn.metrics import accuracy_score, f1_score

R = "/ephemeral/crisis_bench/crisismmd"
z = json.load(open(f"{R}/zoro.json"))
p = json.load(open(f"{R}/probe.json")) if os.path.exists(f"{R}/probe.json") else {}
print(f"CLIP-L/14 image forward: {p.get('clip_ms', '?')} ms/image\n")
for task in ("informative", "humanitarian", "damage"):
    print(f"== {task}")
    print(f"   {'system':38s} {'acc':>6s} {'macroF1':>8s} {'wF1':>6s}  latency")
    for mode in ("image", "text", "both"):
        k = f"{task}/{mode}"
        if k not in z:
            continue
        d = z[k]["preds"]
        ids = sorted(d)
        g, pr = [d[i]["gold"] for i in ids], [d[i]["pred"] for i in ids]
        print(f"   {'Zoro ' + mode:38s} {accuracy_score(g, pr):6.3f} {f1_score(g, pr, average='macro'):8.3f} {f1_score(g, pr, average='weighted'):6.3f}  {z[k]['seq_ms_median']} ms one-at-a-time, {z[k]['wall'] / len(ids) * 1000:.0f} ms/item at 8 concurrent")
    for mode, r in p.get(task, {}).items():
        print(f"   {'dedicated ' + mode + ' (train ' + str(r['n_train']) + ')':38s} {r['acc']:6.3f} {r['macro_f1']:8.3f} {r['weighted_f1']:6.3f}")
    print()
