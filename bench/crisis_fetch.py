"""Download only the CrisisMMD images needed for the benchmark (HF split files)."""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

ROOT = "/ephemeral/crisis_bench/crisismmd"
REPO = "QCRI/CrisisMMD"
TRAIN_PER_TASK = int(sys.argv[1]) if len(sys.argv) > 1 else 4000


def rows(task, split):
    p = hf_hub_download(REPO, f"{task}/{split}.json", repo_type="dataset", local_dir=ROOT)
    txt = open(p).read()
    try:
        return json.loads(txt)
    except Exception:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


need = set()
for task in ("informative", "humanitarian", "damage"):
    for split in ("train", "dev", "test"):
        r = rows(task, split)
        if split == "train":
            random.Random(0).shuffle(r)
            r = r[:TRAIN_PER_TASK]
        if split == "dev":
            continue
        need.update(x["image_path"] for x in r)
need = sorted(p for p in need if not os.path.exists(os.path.join(ROOT, p)))
print("to download:", len(need), flush=True)

done = 0
t0 = time.time()


def get(p):
    for i in range(5):
        try:
            hf_hub_download(REPO, p, repo_type="dataset", local_dir=ROOT)
            return True
        except Exception as e:
            time.sleep(2 * (i + 1))
    print("FAIL", p, flush=True)
    return False


with ThreadPoolExecutor(32) as ex:
    for ok in ex.map(get, need):
        done += 1
        if done % 250 == 0:
            el = time.time() - t0
            print(f"{done}/{len(need)}  {done / el:.1f}/s  eta {(len(need) - done) / (done / el) / 60:.1f} min", flush=True)
print("CRISIS_FETCH_DONE", round(time.time() - t0), "s")
