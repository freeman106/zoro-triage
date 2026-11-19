"""Compare paired report119 demo decisions and burst timings."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median


REPORTS = 400
MATCHED = 303
CLAIM_KIND = {"fire": "fire", "storm": "storm_flood", "quake": "quake_collapse"}
PHOTO_PROXY = {"fire": "fire_smoke_burn", "storm": "water_storm", "quake": "collapse"}
URGENCY = ("non_urgent", "normal", "urgent", "top")
SIGNALS = ("life_threat", "ongoing_hazard")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_rows(rows, path):
    indexed = {}
    for row in rows:
        require(isinstance(row, dict), f"{path}: expected an object per row")
        report_id = row.get("id")
        require(isinstance(report_id, str) and report_id, f"{path}: invalid row ID")
        require(report_id not in indexed, f"{path}: duplicate ID {report_id}")
        indexed[report_id] = row
    return indexed


def read_gold(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(rows, list), f"{path}: expected a JSON array")
    gold = index_rows(rows, path)
    require(len(gold) == REPORTS, f"{path}: expected {REPORTS} reports")
    for report_id, row in gold.items():
        require(
            row["people"] in ("none", "at_risk", "injured")
            and row["time"] in ("now", "past")
            and type(row["image_severe"]) is bool
            and row["claim_group"] in CLAIM_KIND
            and row["image_group"] in PHOTO_PROXY
            and row["gold_urgency"] in URGENCY
            and type(row["gold_mismatch"]) is bool,
            f"{path}: invalid gold labels for {report_id}",
        )
    require(sum(not row["gold_mismatch"] for row in gold.values()) == MATCHED,
            f"{path}: expected {MATCHED} gold-matched reports")
    return gold


def number(value, low=0, high=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def read_decisions(path, gold, new_signals=False):
    with path.open(encoding="utf-8") as stream:
        rows = index_rows((json.loads(line) for line in stream if line.strip()), path)
    missing = gold.keys() - rows.keys()
    extra = rows.keys() - gold.keys()
    require(not missing and not extra,
            f"{path}: {len(missing)} missing IDs {sorted(missing)[:5]}, "
            f"{len(extra)} unexpected IDs {sorted(extra)[:5]}")
    prob_keys = ("people", "now", "severe", "field_photo")
    if new_signals:
        prob_keys += SIGNALS
    for report_id, row in rows.items():
        label = f"{path}:{report_id}"
        require(row["mode"] in ("zoro", "fallback"), f"{label}: invalid mode")
        require(number(row["wall_ms"]), f"{label}: invalid wall_ms")
        require(row["urgency"] in URGENCY, f"{label}: invalid urgency")
        require(isinstance(row["probs"], dict), f"{label}: invalid probs")
        for key in prob_keys:
            value = row["probs"][key]
            require(value is None or number(value, high=1), f"{label}: invalid probs.{key}")
        for key in ("mismatch", "non_field_photo", "review"):
            value = row[key]
            require(value is None or type(value) is bool, f"{label}: invalid {key}")
        for key, choices in (
            ("claim_kind", (*CLAIM_KIND.values(), "other")),
            ("photo_kind", (*PHOTO_PROXY.values(), "unclear")),
        ):
            require(row[key] is None or row[key] in choices, f"{label}: invalid {key}")
    return rows


def flag(probability):
    return None if probability is None else probability >= 0.5


def gold_values(row):
    return {
        "people": row["people"] != "none",
        "now": row["time"] == "now",
        "severe": row["image_severe"],
        "claim_kind": CLAIM_KIND[row["claim_group"]],
        "photo_kind": PHOTO_PROXY[row["image_group"]],
        "urgency": row["gold_urgency"],
        "mismatch": row["gold_mismatch"],
    }


def predicted_values(row):
    return {
        **{key: flag(row["probs"][key]) for key in ("people", "now", "severe", "field_photo")},
        **{key: row[key] for key in ("claim_kind", "photo_kind", "urgency", "mismatch",
                                    "non_field_photo", "review")},
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def paired(before, after):
    counts = Counter(zip(before, after, strict=True))
    comparable = sum(n for (a, b), n in counts.items() if a is not None and b is not None)
    agreed = sum(n for (a, b), n in counts.items() if a is not None and a == b)
    return {
        "n": len(before),
        "changed": sum(n for (a, b), n in counts.items() if a != b),
        "flipped": comparable - agreed,
        "became_available": sum(n for (a, b), n in counts.items() if a is None and b is not None),
        "became_unavailable": sum(n for (a, b), n in counts.items() if a is not None and b is None),
        "both_unavailable": counts[None, None],
        "comparable": comparable,
        "agreed": agreed,
        "agreement": ratio(agreed, comparable),
    }


def macro_f1(truth, predictions):
    f1 = []
    for label in URGENCY:
        common = sum(y == p == label for y, p in zip(truth, predictions, strict=True))
        denominator = truth.count(label) + predictions.count(label)
        f1.append(2 * common / denominator if denominator else 0)
    return sum(f1) / len(f1)


def accuracy(truth, before, after, with_f1=False):
    scores = paired(before, after)
    before_correct = [y == p for y, p in zip(truth, before, strict=True)]
    after_correct = [y == p for y, p in zip(truth, after, strict=True)]
    scores["became_correct"] = sum(not a and b for a, b in zip(before_correct, after_correct))
    scores["became_incorrect"] = sum(a and not b for a, b in zip(before_correct, after_correct))
    for name, predictions, correct in (("before", before, before_correct), ("after", after, after_correct)):
        count = sum(correct)
        scores[name] = {
            "correct": count,
            "accuracy": ratio(count, len(truth)),
            "unavailable": predictions.count(None),
        }
        if with_f1:
            scores[name]["macro_f1"] = macro_f1(truth, predictions)
    return scores


def mismatch_scores(truth, predictions):
    counts = Counter(zip(truth, predictions, strict=True))
    tp, fp = counts[True, True], counts[False, True]
    unflagged, unavailable = counts[True, False], counts[True, None]
    fn = unflagged + unavailable
    return {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": counts[False, False],
        "fn_explicit_negative": unflagged,
        "unavailable_gold_positive": unavailable,
        "unavailable_gold_negative": counts[False, None],
    }


def positive_rate(values):
    positive = values.count(True)
    unavailable = values.count(None)
    return {
        "n": len(values),
        "positive": positive,
        "negative": values.count(False),
        "unavailable": unavailable,
        "positive_rate_all": ratio(positive, len(values)),
        "positive_rate_available": ratio(positive, len(values) - unavailable),
    }


def read_burst(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(result, dict), f"{path}: expected a JSON object")
    require(number(result["elapsed_ms"]), f"{path}: invalid elapsed_ms")
    done, total, fallback = (result[key] for key in ("done", "total", "fallback"))
    require(all(type(x) is int for x in (done, total, fallback)) and 0 <= done <= total and total > 0,
            f"{path}: invalid done, total or fallback")
    runs = result.get("runs", [])
    require(isinstance(runs, list), f"{path}: expected an array of burst runs")
    if runs:
        require(all(isinstance(run, dict) and number(run["elapsed_ms"])
                    and type(run["fallback"]) is int and type(run["zoro"]) is int
                    and run["fallback"] >= 0 and run["zoro"] >= 0
                    and run["fallback"] + run["zoro"] == done for run in runs),
                f"{path}: invalid burst run")
        require(fallback == sum(run["fallback"] for run in runs),
                f"{path}: fallback must sum all burst runs")
    else:
        require(0 <= fallback <= done, f"{path}: invalid fallback count")
    return result


def compare(gold, before, after, burst_before, burst_after):
    ids = list(gold)
    matched = [i for i in ids if not gold[i]["gold_mismatch"]]
    mismatched = [i for i in ids if gold[i]["gold_mismatch"]]
    truth = {i: gold_values(row) for i, row in gold.items()}
    old = {i: predicted_values(row) for i, row in before.items()}
    new = {i: predicted_values(row) for i, row in after.items()}

    def values(rows, key, cohort=ids):
        return [rows[i][key] for i in cohort]

    def score(key, cohort):
        return accuracy(values(truth, key, cohort), values(old, key, cohort), values(new, key, cohort),
                        with_f1=key == "urgency")

    heads = {
        key: {cohort: score(key, selected) for cohort, selected in
              (("all", ids), ("gold_matched", matched), ("gold_mismatched", mismatched))}
        for key in ("people", "now", "severe", "claim_kind")
    }
    heads["urgency"] = {
        "gold_matched": score("urgency", matched),
        "gold_mismatched_descriptive": score("urgency", mismatched),
    }
    gold_mismatch = values(truth, "mismatch")
    old_mismatch, new_mismatch = values(old, "mismatch"), values(new, "mismatch")
    flags = {}
    for key in ("field_photo", "non_field_photo", "review"):
        old_values, new_values = values(old, key), values(new, key)
        flags[key] = {
            **paired(old_values, new_values),
            "before": positive_rate(old_values),
            "after": positive_rate(new_values),
        }
    wall_before = median(values(before, "wall_ms"))
    wall_after = median(values(after, "wall_ms"))
    require(burst_before["total"] == burst_after["total"], "burst workloads must have the same total")
    return {
        "n": len(ids),
        "gold_matched": len(matched),
        "gold_mismatched": len(mismatched),
        "probability_threshold": 0.5,
        "heads": heads,
        "photo_kind_source_event_proxy": {
            "mapping": PHOTO_PROXY,
            "all": score("photo_kind", ids),
        },
        "mismatch": {
            **paired(old_mismatch, new_mismatch),
            "before": mismatch_scores(gold_mismatch, old_mismatch),
            "after": mismatch_scores(gold_mismatch, new_mismatch),
        },
        "flags_without_gold": flags,
        "new_signals_without_gold": {
            key: positive_rate([flag(after[i]["probs"][key]) for i in ids]) for key in SIGNALS
        },
        "wall_ms_including_file_read": {
            "before_median": wall_before,
            "after_median": wall_after,
            "delta_after_minus_before": wall_after - wall_before,
            "before_modes": dict(sorted(Counter(values(before, "mode")).items())),
            "after_modes": dict(sorted(Counter(values(after, "mode")).items())),
        },
        "burst": {
            "before": burst_before,
            "after": burst_after,
            "delta_after_minus_before_ms": burst_after["elapsed_ms"] - burst_before["elapsed_ms"],
            "fallback_delta_after_minus_before": burst_after["fallback"] - burst_before["fallback"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("before", type=Path, help="Baseline decision JSONL")
    parser.add_argument("after", type=Path, help="Bundle decision JSONL")
    parser.add_argument("before_burst", type=Path, help="Baseline burst JSON")
    parser.add_argument("after_burst", type=Path, help="Bundle burst JSON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        gold = read_gold(args.dataset)
        before = read_decisions(args.before, gold)
        after = read_decisions(args.after, gold, new_signals=True)
        result = compare(gold, before, after, read_burst(args.before_burst), read_burst(args.after_burst))
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
