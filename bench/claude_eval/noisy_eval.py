"""Run the demo's own decision code on clean vs messy test reports (same photos, same gold).

Usage (on the instance, from a copy of the demo code):
  ~/venvs/bench/bin/python noisy_eval.py test119.json test119_noisy.json out.json
"""
import json, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor

import decision

decision.DECISION_TIMEOUT = 15.0          # measure judgments, not the demo's 2 s fallback budget
ROOT = "/ephemeral/crisis_bench/crisismmd/"
clean = {r["id"]: r for r in json.load(open(sys.argv[1]))}
noisy = {r["id"]: r for r in json.load(open(sys.argv[2]))}


def run(job):
    tag, r = job
    img = open(ROOT + r["image_path"], "rb").read()
    t = time.time()
    d = decision.decide(r["text"], img, [])
    return tag, r["id"], {"mode": d.get("mode"), "urgency": d["urgency"], "person_state": d.get("person_state"),
                          "probs": d.get("probs"), "claim_kind": d.get("claim_kind"), "ms": round((time.time() - t) * 1000)}


jobs = [("clean", r) for r in clean.values()] + [("noisy", r) for r in noisy.values()]
out = {"clean": {}, "noisy": {}}
with ThreadPoolExecutor(4) as ex:
    for n, (tag, i, d) in enumerate(ex.map(run, jobs), 1):
        out[tag][i] = d
        if n % 100 == 0:
            print(n, "/", len(jobs), flush=True)
json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False)
print("NOISY_EVAL_DONE", {k: statistics.median(v["ms"] for v in out[k].values()) for k in out})
