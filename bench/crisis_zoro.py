"""CrisisMMD (HF v2 split) test sets judged by Zoro-1: image only / text only / image + text.

Usage: ~/venvs/zoro/bin/python ~/crisis_zoro.py   -> /ephemeral/crisis_bench/crisismmd/zoro.json
"""
import base64, json, os, statistics, time
from concurrent.futures import ThreadPoolExecutor

import requests

R = "/ephemeral/crisis_bench/crisismmd"
URL = "http://127.0.0.1:30001/v1/systemone"
EVENT = {"hurricane_harvey": "Hurricane Harvey (2017)", "hurricane_irma": "Hurricane Irma (2017)", "hurricane_maria": "Hurricane Maria (2017)",
         "california_wildfires": "California wildfires (2017)", "mexico_earthquake": "Mexico earthquake (2017)",
         "iraq_iran_earthquake": "Iraq-Iran earthquake (2017)", "srilanka_floods": "Sri Lanka floods (2017)"}

TASKS = {
    "informative": {"type": "choice", "instructions": "Is this post useful for humanitarian aid during the disaster?", "criteria": {
        "informative": "Useful for humanitarian aid: shows or reports damage, affected, injured or missing people, rescue, volunteering or donations, warnings, or other concrete situational information about the disaster.",
        "not_informative": "Not useful for humanitarian aid: unrelated to the disaster, ads, memes, jokes, selfies, generic prayers or opinions without situational information."}},
    "humanitarian": {"type": "choice", "instructions": "Which humanitarian category does this post belong to?", "criteria": {
        "affected_individuals": "People affected by the disaster: evacuated, displaced, stranded, sheltering, without food, water or power.",
        "infrastructure_and_utility_damage": "Damaged or destroyed buildings, houses, roads, bridges, power lines or other infrastructure and utilities.",
        "injured_or_dead_people": "People injured or killed by the disaster.",
        "missing_or_found_people": "People reported missing or found.",
        "rescue_volunteering_or_donation_effort": "Rescue operations, first responders, volunteers, relief supplies, fundraising or donation drives.",
        "vehicle_damage": "Damaged, destroyed or flooded vehicles.",
        "other_relevant_information": "Other disaster-related information that fits none of the above: maps, forecasts, satellite images, flood or fire scenes without clear damage, official updates.",
        "not_humanitarian": "Not useful for humanitarian aid: unrelated content, ads, memes, jokes, selfies, generic prayers or opinions."}},
    "damage": {"type": "choice", "instructions": "How severe is the damage visible in the image?", "criteria": {
        "severe_damage": "Severe damage: substantial destruction, collapsed or roofless buildings, washed-out roads or bridges, structures that are no longer usable.",
        "mild_damage": "Mild damage: partial damage such as some roof, wall, window or fence damage, fallen trees or debris, structures that still stand and can be repaired.",
        "little_or_no_damage": "Little or no damage: no visible damage to buildings or infrastructure, or only trivial damage."}},
}
MODES = {"informative": ("image", "text", "both"), "humanitarian": ("image", "text", "both"), "damage": ("image", "both")}
session = requests.Session()


def rows(task, split):
    txt = open(f"{R}/{task}/{split}.json").read()
    try:
        return json.loads(txt)
    except Exception:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def judge(job):
    task, mode, x = job
    ev = EVENT.get(x["event_name"], x["event_name"])
    body = {"model": "zoro-1", "questions": {"q": TASKS[task]}}
    if mode == "image":
        body["state"] = f"An image posted on social media during the {ev}."
    else:
        body["state"] = {"context": f"A tweet posted during the {ev}.", "tweet_text": x["tweet_text"]}
        if mode == "both":
            body["state"]["note"] = "The attached image was posted with this tweet."
    if mode != "text":
        with open(f"{R}/{x['image_path']}", "rb") as f:
            body["images"] = [base64.b64encode(f.read()).decode()]
    t = time.time()
    for attempt in range(3):
        try:
            a = session.post(URL, json=body, timeout=300).json()["answers"]["q"]
            return job, a["choice"], {k: round(v, 4) for k, v in a["probabilities"].items()}, time.time() - t
        except Exception as e:
            err = e
            time.sleep(2)
    raise err


out = {}
for task in TASKS:
    test = [x for x in rows(task, "test") if os.path.exists(f"{R}/{x['image_path']}")]
    print(f"{task}: {len(test)} test items with image on disk", flush=True)
    for mode in MODES[task]:
        jobs = [(task, mode, x) for x in test]
        judge(jobs[0])
        seq = sorted(judge(j)[3] for j in jobs[:30])
        t0 = time.time()
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(judge, jobs))
        wall = time.time() - t0
        acc = sum(c == j[2]["label"] for j, c, _, _ in res) / len(res)
        out[f"{task}/{mode}"] = {"wall": round(wall, 1), "seq_ms_median": round(statistics.median(seq) * 1000),
                                 "preds": {j[2]["image_id"]: {"gold": j[2]["label"], "pred": c, "probs": p} for j, c, p, _ in res}}
        print(f"  {task}/{mode}: acc {acc:.3f}  {wall / len(res) * 1000:.0f} ms/item at 8 concurrent, one-at-a-time median {statistics.median(seq) * 1000:.0f} ms", flush=True)
        json.dump(out, open(f"{R}/zoro.json", "w"))
print("CRISIS_ZORO_DONE")
