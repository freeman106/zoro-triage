"""Check pairing and summarize the photo noise runs across revisions."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median


PROFILES = ("clean", "gaussian", "blur")


def rate(numerator, denominator):
    return {
        "n": numerator,
        "d": denominator,
        "percent": round(100 * numerator / denominator, 2) if denominator else None,
    }


def is_severe(record, threshold):
    probability = record["p_severe"]
    if probability is not None:
        return probability >= threshold
    return None


def score(records, clean, threshold):
    size = len(records)
    severe_true = sum(row["gold_severe"] for row in records)
    mismatch_true = sum(row["gold_mismatch"] for row in records)
    severity = {row["id"]: is_severe(row, threshold) for row in records}
    clean_severity = {key: is_severe(row, threshold) for key, row in clean.items()}
    severity_hits = sum(severity[row["id"]] == row["gold_severe"] for row in records)
    severe_tp = sum(row["gold_severe"] and severity[row["id"]] is True for row in records)
    mismatch_tp = sum(row["gold_mismatch"] and row["mismatch"] is True for row in records)
    mismatch_predicted = sum(row["mismatch"] is True for row in records)
    reason = [row["reason_review"] for row in records if row["reason_review"] is not None]
    complete = [result for result in reason if result["status"] == "done"]
    initial_top = sum(row["urgency"] == "top" for row in clean.values())
    comparable_damage = [
        key for key in severity if severity[key] is not None and clean_severity[key] is not None
    ]
    return {
        "reports": size,
        "severity_accuracy": rate(severity_hits, size),
        "severe_recall": rate(severe_tp, severe_true),
        "severity_flips_from_clean": rate(
            sum(severity[key] != clean_severity[key] for key in comparable_damage),
            len(comparable_damage),
        ),
        "mismatch_precision": rate(mismatch_tp, mismatch_predicted),
        "mismatch_recall": rate(mismatch_tp, mismatch_true),
        "urgency_changes_from_clean": rate(sum(row["urgency"] != clean[row["id"]]["urgency"] for row in records), size),
        "clean_top_lost": rate(
            sum(clean[row["id"]]["urgency"] == "top" and row["urgency"] != "top" for row in records),
            initial_top,
        ),
        "first_pass_fallbacks": sum(row["mode"] != "zoro" for row in records),
        "first_pass_median_ms": median(row["ms"] for row in records),
        "first_pass_review_flags": sum(row["review"] for row in records),
        "final_review_flags": sum(row["effective_review"] for row in records),
        "hard_reviews_kept": rate(sum(row["hard_review"] and row["effective_review"] for row in records),
                                  sum(row["hard_review"] for row in records)),
        "reason_completed": rate(len(complete), len(reason)),
        "reason_selected_topics": dict(Counter(result["topic"] for result in complete)),
        "reason_complete_median_ms": median(result["ms"] for result in complete) if complete else None,
    }


def index(run, profile):
    records = run["profiles"][profile]
    indexed = {row["id"]: row for row in records}
    if not indexed or len(indexed) != len(records):
        raise ValueError(f"Empty or duplicated IDs for {run['revision']} / {profile}")
    return indexed


def pairs(records):
    return {
        key: (row["image_sha256"], row["gold_severe"], row["gold_mismatch"], row["gold_urgency"])
        for key, row in records.items()
    }


def compare(versions):
    if len({run["dataset_sha256"] for run in versions.values()}) != 1:
        raise ValueError("The report text and labels must come from the same dataset across revisions")
    benchmark = {}
    agreement = {}
    reason_pairs = {}
    for name, run in versions.items():
        clean = index(run, "clean")
        benchmark[name] = {}
        for profile in PROFILES:
            records = index(run, profile)
            if {key: labels[1:] for key, labels in pairs(records).items()} != {
                key: labels[1:] for key, labels in pairs(clean).items()
            }:
                raise ValueError(f"Report IDs or labels differ from clean for {name} / {profile}")
            if pairs(records) != pairs(index(versions["main"], profile)):
                raise ValueError(f"Photo bytes or labels differ across revisions for {name} / {profile}")
            if any(row["hard_review"] and not row["effective_review"] for row in records.values()):
                raise ValueError(f"A mandatory review flag was removed for {name} / {profile}")
            for row in records.values():
                expected = name == "dev" and row["mode"] == "zoro" and row["review"]
                if (row["reason_review"] is not None) != expected:
                    raise ValueError(f"Unexpected reasoning coverage for {name} / {profile} / {row['id']}")
            benchmark[name][profile] = score(records.values(), clean, run["severity_threshold"])

    dev_clean = index(versions["dev"], "clean")
    for profile in PROFILES:
        main = index(versions["main"], profile)
        dev = index(versions["dev"], profile)
        agreement[profile] = {
            field: rate(sum(main[key][field] == dev[key][field] for key in main), len(main))
            for field in ("urgency", "review")
        }
        if profile != "clean":
            attempted = [
                (dev_clean[key]["reason_review"], dev[key]["reason_review"])
                for key in dev
                if dev_clean[key]["reason_review"] is not None and dev[key]["reason_review"] is not None
            ]
            completed = [(a, b) for a, b in attempted if a["status"] == b["status"] == "done"]
            reason_pairs[profile] = {
                "attempted_in_both": len(attempted),
                "selected_topic_agreement": rate(sum(a["topic"] == b["topic"] for a, b in completed), len(completed)),
            }
    return {
        "metrics": benchmark,
        "main_dev_first_pass_agreement": agreement,
        "dev_reason_agreement_with_clean": reason_pairs,
        "runs": versions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy", "main", "dev", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    runs = {name: json.loads(getattr(args, name).read_text()) for name in ("legacy", "main", "dev")}
    report = compare(runs)
    args.out.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n")
    for version, scores in report["metrics"].items():
        for profile, result in scores.items():
            print(version, profile, json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
