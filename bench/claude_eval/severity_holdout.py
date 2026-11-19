"""Fresh holdout: CrisisMMD damage-test photos never used in the 119 scenarios. Old vs revised definitions, thresholds fixed from train."""
import importlib.util, json, sys
from concurrent.futures import ThreadPoolExecutor

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    src = open(path).read().split("\nscen = json.load")[0]   # reuse definitions only, skip the main run
    src = src.replace("SCEN, OUT = sys.argv[1], sys.argv[2]", "SCEN = OUT = None")
    mod = type(sys)(name); exec(compile(src, path, "exec"), mod.__dict__); return mod

old = load("/home/ubuntu/severity_variants.py", "old")
new = load("/home/ubuntu/severity_variants_v2def.py", "new")
rows = [json.loads(l) for l in open("/ephemeral/crisis_bench/crisismmd/damage/test.json")] if open("/ephemeral/crisis_bench/crisismmd/damage/test.json").read(1) != "[" else json.load(open("/ephemeral/crisis_bench/crisismmd/damage/test.json"))
used = {s["image_path"] for s in json.load(open("/ephemeral/crisis_bench/scenarios.json"))}
hold = [r for r in rows if r["image_path"] not in used]
print("holdout images:", len(hold), flush=True)
THR = {("old", "v0"): 0.62, ("old", "v2"): 0.47, ("new", "v0"): 0.62, ("new", "v2"): 0.60}

def run(r):
    raw = open(old.ROOT + r["image_path"], "rb").read()
    small = old.demo_resize(raw)
    out = {"gold": r["label"] == "severe_damage"}
    for tag, m in (("old", old), ("new", new)):
        a = m.ask(small, {"v0": {"type": "noul", "instructions": m.CURRENT}, "v2": m.LEVELS})
        out[tag] = {"v0": a["v0"]["noul"], "v2": a["v2"]["probabilities"]["severe"]}
    return out

with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(run, hold))
for (tag, v), thr in THR.items():
    acc = sum((x[tag][v] >= thr) == x["gold"] for x in res) / len(res)
    print(f"{tag} {v} thr {thr}: holdout accuracy {acc:.3f}  (n={len(res)})")
json.dump(res, open("/home/ubuntu/severity_holdout.json", "w"))
