"""Measure first-pass review badges and latency for the curated 100-report burst."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory


def measure(demo, images):
    sys.path.insert(0, str(demo))
    from store import Inbox

    workers = int(os.environ.get("DEMO_WORKERS", "4"))
    manifest = demo / "fixtures/reports.json"
    with TemporaryDirectory(prefix="review-burst-") as home:
        inbox = Inbox(home, images, manifest, workers)
        try:
            ids = inbox.add_examples()
            with inbox.changed:
                if len(ids) != 100 or not inbox.changed.wait_for(lambda: inbox.burst["done"] == 100, 180):
                    raise RuntimeError("The curated burst did not finish 100 reports")
                burst = inbox.burst.copy()
                decisions = {}
                for (payload,) in inbox.db.execute("SELECT payload FROM journal WHERE stream='decisions' ORDER BY seq"):
                    event = json.loads(payload)
                    if event["event"] == "decision":
                        decisions.setdefault(event["id"], event["decision"])
                records = []
                for key in ids:
                    card, result = inbox.cards[key], decisions[key]
                    records.append({
                        "scenario": card["scenario_id"], "urgency": result["urgency"], "review": result["review"],
                        "uncertainty_reason": any(s.startswith("판단 불확실") for s in result["reasons"]),
                        "decision_ms": result["ms"], "mode": result["mode"],
                        "first_decision_with_wait_ms": card.get("first_decision_with_wait_ms"),
                    })
            if len({r["scenario"] for r in records}) != 100 or burst["revisions"] != [0]:
                raise RuntimeError("Expected 100 unique scenarios without operator rules")
            total = burst["elapsed_ms"]
            return {
                "workers": workers, "reports": len(records), "fixture_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "policy_version": decisions[ids[0]]["policy_version"],
                "review_badges": sum(r["review"] for r in records),
                "uncertainty_reasons": sum(r["uncertainty_reason"] for r in records),
                "first_pass_fallback": burst["fallback"], "total_ms": total,
                "throughput_per_second": round(100_000 / total, 3), "mean_completion_interval_ms": total / 100,
                "median_decision_ms": median(r["decision_ms"] for r in records), "records": records,
            }
        finally:
            inbox.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", type=Path)
    parser.add_argument("--images", type=Path, default=Path("/ephemeral/crisis_bench/crisismmd"))
    args = parser.parse_args()
    print(json.dumps(measure(args.demo.resolve(), args.images), ensure_ascii=False, indent=2))
