"""Measure frozen report119 burst throughput at 4, 8 and 16 inbox workers."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


ORDERS = ((4, 8, 16), (8, 16, 4), (16, 4, 8))
FIXTURE_COUNT = 100
WARMUP_WORKERS = 4
WARMUP_COUNT = 8
MODEL_POOL_SIZE = 12
TIMEOUT_REASON = "Zoro 응답 시간 초과 · 키워드 모드"
FROZEN_FILES = ("decision.py", "store.py", "mismatch_types.py", "mismatch_config.json",
                "scene_question.py", "fixtures/reports.json")


def measure(args):
    if args.group == "patched":
        os.environ["DEMO_WORKERS"] = str(args.worker)
    sys.path.insert(0, str(args.demo))
    import decision
    from store import Inbox

    model_pool_size = decision._POOL._max_workers
    if args.group == "patched" and model_pool_size < 2 * args.worker:
        raise ValueError("Insufficient model transport concurrency")
    if args.group != "patched" and model_pool_size != MODEL_POOL_SIZE:
        raise ValueError(f"Expected live model pool size {MODEL_POOL_SIZE}, got {model_pool_size}")

    with TemporaryDirectory(prefix="report119-workers-") as directory:
        home = Path(directory)
        manifest = args.demo / "fixtures/reports.json"
        if args.warmup:
            rows = json.loads(manifest.read_text(encoding="utf-8"))
            manifest = home / "warmup-fixtures.json"
            manifest.write_text(json.dumps(rows[:WARMUP_COUNT]), encoding="utf-8")
        inbox = Inbox(home, args.images, manifest, args.worker)
        try:
            expected = WARMUP_COUNT if args.warmup else FIXTURE_COUNT
            if inbox.cards or inbox.rules or len(inbox.fixtures) != expected:
                raise ValueError("Expected an empty inbox and the curated fixtures")
            ids = inbox.add_examples()
            with inbox.changed:
                if not inbox.changed.wait_for(lambda: inbox.burst["done"] == expected, timeout=180):
                    raise TimeoutError(f"Inbox completed {inbox.burst['done']}/{expected} decisions")
                state = json.loads(inbox.state())
                # A background retry can replace card.decision while this burst runs.
                first = {}
                for payload, in inbox.db.execute("SELECT payload FROM journal WHERE stream='decisions' ORDER BY seq"):
                    event = json.loads(payload)
                    if event["event"] == "decision":
                        first.setdefault(event["id"], event["decision"])
            if state["rules"] or state["burst"]["revisions"] != [0]:
                raise ValueError("The run included operator rules")
            cards = {card["id"]: card for card in state["cards"]}
            records = []
            for key in ids:
                card = cards[key]
                result = first[key]
                if result["mode"] not in {"zoro", "fallback"}:
                    raise ValueError(f"Unexpected decision mode for {card['scenario_id']}")
                records.append({
                    "scenario_id": card["scenario_id"],
                    "ms": result["ms"],
                    "mode": result["mode"],
                    "model": result.get("model"),
                    "policy_version": result["policy_version"],
                    "fallback_reason": result["reasons"][0] if result["mode"] == "fallback" else None,
                })
            if len({record["policy_version"] for record in records}) != 1:
                raise ValueError("Decisions used different policies")
            fallback = sum(record["mode"] == "fallback" for record in records)
            deadline_timeout = sum(record["fallback_reason"] == TIMEOUT_REASON for record in records)
            if fallback != state["burst"]["fallback"]:
                raise ValueError("Decision fallback count differs from the server burst count")
            recovered = None
            if args.group == "patched" and fallback:
                with inbox.changed:
                    def finished():
                        return all((inbox.cards[key].get("recovery") or {}).get("status") not in
                                   ("scheduled", "retrying") for key in ids)
                    if not inbox.changed.wait_for(finished, timeout=180):
                        raise TimeoutError("Recovery did not settle")
                    recovered = sum((inbox.cards[key].get("recovery") or {}).get("status") == "recovered" for key in ids)
            return {
                "retry_recovered": recovered,
                "workers": args.worker,
                "model_pool": model_pool_size,
                "count": len(records),
                "elapsed_ms": state["burst"]["elapsed_ms"],
                "median_decision_ms": median(record["ms"] for record in records),
                "fallback": fallback,
                "deadline_timeout": deadline_timeout,
                "other_fallback": fallback - deadline_timeout,
                "decisions": records,
            }
        finally:
            inbox.close()
            decision._POOL.shutdown(wait=True, cancel_futures=True)


def child_run(args, workers, warmup=False):
    command = [sys.executable, str(Path(__file__).resolve()),
               "--demo", str(args.demo), "--images", str(args.images),
               "--worker", str(workers), "--group",args.group]
    if warmup:
        command.append("--warmup")
    output = subprocess.check_output(command, text=True, timeout=240)
    time.sleep(args.drain_seconds)
    return json.loads(output)


def summarize(runs, orders):
    summaries = {}
    for workers in orders[0]:
        group = [run for run in runs if run["workers"] == workers]
        summaries[workers] = {
            "trials": len(group),
            "count": sum(run["count"] for run in group),
            "median_elapsed_ms": median(run["elapsed_ms"] for run in group),
            "median_decision_ms": median(record["ms"] for run in group for record in run["decisions"]),
            "fallback": sum(run["fallback"] for run in group),
            "deadline_timeout": sum(run["deadline_timeout"] for run in group),
            "other_fallback": sum(run["other_fallback"] for run in group),
        }
    eligible = [worker for worker, summary in summaries.items()
                if summary["trials"] == len(orders) and summary["fallback"] == 0]
    chosen = min(eligible, key=lambda worker: (summaries[worker]["median_elapsed_ms"], worker), default=None)
    return summaries, chosen


def sweep(args):
    orders = {"requested": ORDERS, "recovery": ((2,3),(3,2),(2,3)), "serial": ((1,),(1,),(1,)),
              "patched": ((3,4,6,8),(8,6,4,3),(6,3,8,4))}[args.group]
    files = (*FROZEN_FILES, *(name for name in ("person_choice.py", "person_threshold.json") if (args.demo / name).is_file()))
    manifest = {name: sha256((args.demo / name).read_bytes()).hexdigest() for name in files}
    rows = json.loads((args.demo / "fixtures/reports.json").read_text(encoding="utf-8"))
    if len(rows) != FIXTURE_COUNT or len({row["id"] for row in rows}) != FIXTURE_COUNT:
        raise ValueError("Expected 100 unique curated scenarios")
    for row in rows:
        if not (args.images / row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])

    runs = []
    warmups = []
    for trial, order in enumerate(orders, 1):
        for workers in order:
            warmup = child_run(args, WARMUP_WORKERS, warmup=True)
            warmups.append(dict(trial=trial, before_workers=workers, **warmup))
            run = child_run(args, workers)
            runs.append(dict(trial=trial, **run))
            print(f"Trial {trial}, workers {workers}: {run['elapsed_ms']} ms, "
                  f"{run['fallback']} fallbacks", file=sys.stderr, flush=True)
    if manifest != {name: sha256((args.demo / name).read_bytes()).hexdigest() for name in files}:
        raise ValueError("Frozen inputs changed during the sweep")

    summaries, chosen = summarize(runs, orders)
    report = {
        "input_sha256": manifest,
        "model_pool_size": {r["workers"]:r["model_pool"] for r in runs},
        "orders": orders,
        "operator_rules": [],
        "drain_seconds": args.drain_seconds,
        "warmup_workers": WARMUP_WORKERS,
        "warmup_count": WARMUP_COUNT,
        "timeout_definition": "Engine timeout fallback (shared decision deadline or socket timeout)",
        "warmups": warmups,
        "runs": runs,
        "summary": summaries,
        "chosen_workers": chosen,
    }
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"chosen_workers": chosen, "summary": summaries}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True, help="Frozen demo119 directory")
    parser.add_argument("--images", type=Path, required=True, help="Root of the CrisisMMD images")
    parser.add_argument("--out", type=Path, help="Destination for the JSON report")
    parser.add_argument("--drain-seconds", type=float, default=1.0,
                        help="Idle time after each benchmark subprocess exits")
    parser.add_argument("--group", choices=["requested","recovery","serial","patched"], default="requested")
    parser.add_argument("--worker", type=int, choices=(1,2,3,4,6,8,16), help=argparse.SUPPRESS)
    parser.add_argument("--warmup", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.demo = args.demo.resolve()
    args.images = args.images.resolve()
    if args.worker is not None:
        print(json.dumps(measure(args), ensure_ascii=False))
        return
    if args.out is None:
        parser.error("--out is required")
    if args.out.exists():
        parser.error(f"--out already exists: {args.out}")
    if args.drain_seconds < 0:
        parser.error("--drain-seconds cannot be negative")
    sweep(args)


if __name__ == "__main__":
    main()
