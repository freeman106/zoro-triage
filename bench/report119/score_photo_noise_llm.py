"""Score categorical VLM controls against the exact §21 photo-noise inputs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from score_photo_noise import PROFILES, index, pairs, rate


def score(records, clean):
    severity_hits = sum(row["severe_prediction"] is row["gold_severe"] for row in records.values())
    severe_true = sum(row["gold_severe"] for row in records.values())
    severe_tp = sum(row["gold_severe"] and row["severe_prediction"] is True for row in records.values())
    mismatch_true = sum(row["gold_mismatch"] for row in records.values())
    mismatch_tp = sum(row["gold_mismatch"] and row["mismatch"] is True for row in records.values())
    mismatch_predicted = sum(row["mismatch"] is True for row in records.values())
    severity_pairs = [key for key, row in records.items()
                      if row["severe_prediction"] is not None and clean[key]["severe_prediction"] is not None]
    urgency_pairs = [key for key, row in records.items()
                     if row["urgency"] is not None and clean[key]["urgency"] is not None]
    clean_top = [key for key in urgency_pairs if clean[key]["urgency"] == "top"]
    reviews = [row["reason_review"] for row in records.values() if row.get("reason_review") is not None]
    complete = [review for review in reviews if review["status"] == "done"]
    return {
        "severity_accuracy": rate(severity_hits, len(records)),
        "severe_recall": rate(severe_tp, severe_true),
        "severity_flips_from_clean": rate(
            sum(records[key]["severe_prediction"] != clean[key]["severe_prediction"] for key in severity_pairs),
            len(severity_pairs),
        ),
        "mismatch_precision": rate(mismatch_tp, mismatch_predicted),
        "mismatch_recall": rate(mismatch_tp, mismatch_true),
        "urgency_changes_from_clean": rate(
            sum(records[key]["urgency"] != clean[key]["urgency"] for key in urgency_pairs), len(urgency_pairs),
        ),
        "clean_top_lost": rate(sum(records[key]["urgency"] != "top" for key in clean_top), len(clean_top)),
        "invalid_reports": sum(bool(row["invalid_fields"]) for row in records.values()),
        "transport_error_reports": sum(row["transport_error"] for row in records.values()),
        "invalid_field_counts": dict(Counter(field for row in records.values() for field in row["invalid_fields"])),
        "invalid_urgency": sum(row["urgency"] is None for row in records.values()),
        "hard_reviews": sum(row["hard_review"] for row in records.values()),
        "confirmation_completed": rate(len(complete), len(reviews)),
        "confirmation_topics": dict(Counter(review["topic"] for review in complete)),
    }


def confirmation_agreement(clean, variant):
    attempted = [
        (clean[key]["reason_review"], row["reason_review"])
        for key, row in variant.items()
        if clean[key].get("reason_review") is not None and row.get("reason_review") is not None
    ]
    completed = [(a, b) for a, b in attempted if a["status"] == b["status"] == "done"]
    return {
        "attempted_in_both": len(attempted),
        "selected_topic_agreement": rate(sum(a["topic"] == b["topic"] for a, b in completed), len(completed)),
    }


def fire_nonfire_mismatch(records):
    positives = [row for row in records.values()
                 if (row["claim_group"] == "fire") != (row["photo_group"] == "fire")]
    correct = sum(row["gold_mismatch"] for row in positives)
    return {
        "precision": rate(correct, len(positives)),
        "recall": rate(correct, sum(row["gold_mismatch"] for row in records.values())),
    }


def compare(reference, controls):
    reference_run = reference["runs"]["dev"]
    metrics = {}
    agreements = {}
    for name, control in controls.items():
        if control["dataset_sha256"] != reference_run["dataset_sha256"]:
            raise ValueError(f"{name} did not use the reference dataset")
        clean = index(control, "clean")
        metrics[name] = {}
        agreements[name] = {}
        for profile in PROFILES:
            rows = index(control, profile)
            if pairs(rows) != pairs(index(reference_run, profile)):
                raise ValueError(f"Report IDs, image bytes or labels differ for {name} / {profile}")
            metrics[name][profile] = score(rows, clean)
            if name == "kc":
                metrics[name][profile]["mismatch_fire_nonfire"] = fire_nonfire_mismatch(rows)
            if profile != "clean":
                agreements[name][profile] = confirmation_agreement(clean, rows)
    return metrics, agreements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("zoro", "qwen", "internvl", "kc", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--json-qwen", type=Path)
    parser.add_argument("--json-internvl", type=Path)
    args = parser.parse_args()
    if bool(args.json_qwen) != bool(args.json_internvl):
        parser.error("Supply both original JSON trial outputs together")
    source = args.zoro.read_bytes()
    reference_digest = hashlib.sha256(source).hexdigest()
    reference = json.loads(source)
    controls = {name: json.loads(getattr(args, name).read_text()) for name in ("qwen", "internvl", "kc")}
    if any(run["reference_sha256"] != reference_digest for run in controls.values()):
        raise ValueError("The controls did not use this section 21 reference file")
    metrics, agreements = compare(reference, controls)
    report = {
        "reference": {
            "file": args.zoro.name,
            "sha256": reference_digest,
            "zoro_dev": reference["metrics"]["dev"],
            "zoro_confirmation_agreement": reference["dev_reason_agreement_with_clean"],
        },
        "metrics": metrics,
        "confirmation_agreement_with_clean": agreements,
        "runs": controls,
    }
    if args.json_qwen:
        trial = {name: json.loads(getattr(args, f"json_{name}").read_text()) for name in ("qwen", "internvl")}
        if any(run["reference_sha256"] != reference_digest for run in trial.values()):
            raise ValueError("The JSON trial did not use this section 21 reference file")
        trial_metrics, trial_agreements = compare(reference, trial)
        report["unconstrained_json_trial"] = {
            "metrics": trial_metrics,
            "confirmation_agreement_with_clean": trial_agreements,
            "runner_sha256": {name: run["runner_sha256"] for name, run in trial.items()},
        }
    args.out.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"metrics": metrics, "confirmation_agreement": agreements}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
