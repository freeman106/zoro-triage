"""Validate and align the synthetic paired text-shift splits without filtering rewrites."""
import argparse
from collections import Counter
import json
from pathlib import Path


def load_pairs(scenarios_path, noisy_path):
    """Return clean training reports, clean test reports, and ID-aligned noisy reports."""
    rows = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    rewrites = json.loads(Path(noisy_path).read_text(encoding="utf-8"))
    train = [r for r in rows if r["split"] == "train"]
    clean = [r for r in rows if r["split"] == "test"]
    if (len(rows), len(train), len(clean), len(rewrites)) != (1900, 1500, 400, 400):
        raise ValueError("Expected 1500 clean train reports and 400 paired test reports")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate scenario IDs")
    if {r["image_id"] for r in train} & {r["image_id"] for r in clean}:
        raise ValueError("The train and test image sets overlap")
    if sum(not r["gold_mismatch"] for r in clean) != 303:
        raise ValueError("Expected 303 matched test reports")
    for r in rows:
        urgency = "non_urgent"
        if r["time"] == "now":
            risk = int(r["people"] != "none") + int(r["image_severe"])
            urgency = ("normal", "urgent", "top")[risk]
        if r["gold_urgency"] != urgency or r["gold_mismatch"] != (r["claim_group"] != r["image_group"]):
            raise ValueError(f"{r['id']}: gold labels disagree with the original rules")
    by_id = {r["id"]: r for r in rewrites}
    if set(by_id) != {r["id"] for r in clean}:
        raise ValueError("Noisy test IDs differ from clean test IDs")
    noisy = [by_id[r["id"]] for r in clean]
    for original, rewrite in zip(clean, noisy):
        if any(original[k] != rewrite.get(k) for k in original if k != "text"):
            raise ValueError(f"{original['id']}: non-text scenario field changed")
        if not rewrite["text"].strip() or rewrite["clean_text"] != original["text"] or rewrite["text"] == original["text"]:
            raise ValueError(f"{original['id']}: missing rewrite or inconsistent clean reference")
    return train, clean, noisy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("scenarios", nargs="?", type=Path, default=here / "scenarios.json")
    parser.add_argument("noisy", nargs="?", type=Path, default=here / "test119_noisy.json")
    args = parser.parse_args()
    train, clean, noisy = load_pairs(args.scenarios, args.noisy)
    print(json.dumps({
        "train": len(train),
        "test": len(clean),
        "urgency_denominator": sum(not r["gold_mismatch"] for r in clean),
        "noise_styles": dict(Counter(r["noise_style"] for r in noisy)),
        "slot_checker_flagged": sum(bool(r["quality_issues"]) for r in noisy),
        "flagged_in_urgency_denominator": sum(bool(r["quality_issues"]) and not r["gold_mismatch"] for r in noisy),
    }, ensure_ascii=False))
