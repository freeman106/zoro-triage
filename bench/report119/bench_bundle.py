"""Compare two frozen decision modules using the demo inbox and one shared Zoro server."""
import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import random
from statistics import median
import sys
from tempfile import TemporaryDirectory
import time


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def single(row, image_root, module):
    start = time.perf_counter()
    image = (image_root / row["image_path"]).read_bytes()
    result = module.decide(row["text"], image, [])
    return dict(result, id=row["id"], wall_ms=round(1000*(time.perf_counter()-start), 3))


def burst(store, module, manifest, image_root):
    store.decide = module.decide
    with TemporaryDirectory() as home:
        inbox = store.Inbox(Path(home), image_root, manifest, workers=4)
        try:
            ids = inbox.add_examples()
            with inbox.changed:
                complete = inbox.changed.wait_for(lambda: inbox.burst["done"] == 100, timeout=180)
                if not complete or len(set(ids)) != 100:
                    raise RuntimeError("Incomplete burst")
                stats = dict(inbox.burst)
                stats["zoro"] = sum(inbox.cards[k]["decision"]["mode"] == "zoro" for k in ids)
            return stats
        finally:
            inbox.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, required=True, help="Directory with frozen before.py, after.py and demo modules")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.demo.resolve()))
    import store
    variants = {name: load_module(args.demo/(name+".py"), name) for name in ("before","after")}
    manifest = args.demo/"fixtures/reports.json"
    rows = json.loads(args.data.read_text())
    assert len(rows) == len({r["id"] for r in rows}) == 400 and all(r["split"]=="test" for r in rows)
    paths = [args.data,manifest,Path(__file__),*(args.demo/file for file in (
        "before.py","after.py","store.py","scene_question.py","mismatch_types.py","mismatch_config.json"))]
    (args.output/"provenance.json").write_text(json.dumps({p.name:sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    for _ in range(3):
        for module in variants.values():
            assert single(rows[0], args.images, module)["mode"] == "zoro"
    first = ["before","after"]*200
    random.Random(119).shuffle(first)
    modes = {name: Counter() for name in variants}
    with ExitStack() as stack:
        writers = {name:stack.enter_context((args.output/(name+".jsonl")).open("x")) for name in variants}
        for i,(row,lead) in enumerate(zip(rows,first),1):
            for name in (lead, "after" if lead=="before" else "before"):
                result = single(row,args.images,variants[name])
                writers[name].write(json.dumps(result,ensure_ascii=False)+"\n")
                writers[name].flush()
                modes[name][result["mode"]] += 1
            if i%50==0:
                print("paired",i,modes,flush=True)
    times = {name:[] for name in variants}
    for name in ("before","after","after","before","before","after"):
        run = burst(store,variants[name],manifest,args.images)
        times[name].append(run)
        print("burst",name,run["elapsed_ms"],"ms",run["fallback"],"fallback",flush=True)
    for name,runs in times.items():
        data = dict(elapsed_ms=median(r["elapsed_ms"] for r in runs),done=100,total=100,
                    fallback=sum(r["fallback"] for r in runs),runs=runs)
        (args.output/(name+"-burst.json")).write_text(json.dumps(data,indent=2))
    print("FINISHED",flush=True)


if __name__=="__main__":
    main()
