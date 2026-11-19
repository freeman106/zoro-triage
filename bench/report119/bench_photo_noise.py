"""Evaluate one source revision on paired original and degraded report photos."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import io
import json
from pathlib import Path
from random import Random
from statistics import NormalDist, median
import sys
import time

from PIL import Image, ImageChops, ImageFilter, __version__ as pillow_version


SEED = 119
SIGMA = 24
BLUR_RADIUS = 3.5
FIRST_TIMEOUT = 15.0
PROFILES = ("clean", "gaussian", "blur")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def png(image):
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()


def photos_for(source, images, reports):
    paths = sorted({report["image_path"] for report in reports})
    normal = NormalDist(mu=128, sigma=SIGMA)
    lut = [round(normal.inv_cdf((index + 0.5) / 256)) for index in range(256)]
    photos = {}
    for path in paths:
        prepared = source.prepare_photo((images / path).read_bytes())
        with Image.open(io.BytesIO(prepared)) as decoded:
            base = decoded.convert("RGB")
        # Quantize a standard-normal inverse CDF to a 256-entry table. A random
        # byte picks a bin per pixel, and the same offset applies to RGB.
        seed = int.from_bytes(hashlib.sha256(f"{SEED}:{path}".encode()).digest(), "big")
        samples = Random(seed).randbytes(base.width * base.height)
        offsets = Image.frombytes("L", base.size, samples).point(lut)
        degraded = {
            "clean": base,
            "gaussian": ImageChops.add(base, Image.merge("RGB", (offsets,) * 3), offset=-128),
            "blur": base.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS)),
        }
        photos[path] = {name: png(image) for name, image in degraded.items()}
    return photos


def first_pass(source, report, photo):
    decision = source.decide(report["text"], photo, [])
    probs = decision["probs"]
    past_hazard = (
        probs["now"] is not None and probs["ongoing_hazard"] is not None
        and probs["now"] < 0.5 <= probs["ongoing_hazard"]
    )
    record = {
        "id": report["id"],
        "image_sha256": hashlib.sha256(photo).hexdigest(),
        "gold_severe": report["image_severe"],
        "gold_mismatch": report["gold_mismatch"],
        "gold_urgency": report["gold_urgency"],
        "mode": decision["mode"],
        "urgency": decision["urgency"],
        "p_severe": probs["severe"],
        "mismatch": decision["mismatch"],
        "review": decision["review"],
        "hard_review": bool(decision["mismatch"] or decision["non_field_photo"] or past_hazard),
        "effective_review": decision["review"],
        "reason_review": None,
        "ms": decision["ms"],
        "model": decision.get("model"),
        "policy_version": decision["policy_version"],
    }
    return record, decision


def second_pass(source, report, photo, decision):
    started = time.monotonic()
    result = source.reason_review(report["text"], photo, decision)
    return {
        **(result or {}),
        "status": "done" if result is not None else "unavailable",
        "wall_ms": round((time.monotonic() - started) * 1000, 1),
    }


def summarize(records, threshold):
    scored = [row for row in records if row["p_severe"] is not None]
    attempts = [row["reason_review"] for row in records if row["reason_review"] is not None]
    completed = [attempt for attempt in attempts if attempt["status"] == "done"]
    return {
        "reports": len(records),
        "severe_scored": len(scored),
        "severe_correct": sum((row["p_severe"] >= threshold) == row["gold_severe"] for row in scored),
        "gold_severe": sum(row["gold_severe"] for row in records),
        "severe_true_positive": sum(row["gold_severe"] and row["p_severe"] >= threshold for row in scored),
        "urgency_correct": sum(row["urgency"] == row["gold_urgency"] for row in records),
        "fallbacks": sum(row["mode"] == "fallback" for row in records),
        "first_pass_median_ms": median(row["ms"] for row in records),
        "first_pass_reviews": sum(row["review"] for row in records),
        "effective_reviews": sum(row["effective_review"] for row in records),
        "reason_attempted": len(attempts),
        "reason_completed": len(completed),
        "reason_median_attempt_ms": median(attempt["wall_ms"] for attempt in attempts) if attempts else None,
        "reason_median_completed_ms": median(attempt["wall_ms"] for attempt in completed) if completed else None,
    }


def evaluate(args):
    sys.path.insert(0, str(args.source.resolve()))
    decision = importlib.import_module("decision")
    if Path(decision.__file__).resolve() != (args.source / "decision.py").resolve():
        raise RuntimeError("decision.py must come from --source")
    decision.DECISION_TIMEOUT = FIRST_TIMEOUT
    threshold = getattr(decision, "DAMAGE_THRESHOLD", 0.50)

    manifest = args.data.read_bytes()
    candidates = sorted(json.loads(manifest), key=lambda report: report["id"])
    if args.limit > len(candidates) or len(set(row["id"] for row in candidates)) != len(candidates):
        raise ValueError("Expected a dataset with enough unique report IDs")
    reports = sorted(Random(SEED).sample(candidates, args.limit), key=lambda report: report["id"])
    print(f"Preparing images for {len(reports)} reports", flush=True)
    input_photos = photos_for(decision, args.images, reports)
    profiles = {}
    first_results = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for profile in PROFILES:
            print(f"{profile}: first-pass inference", flush=True)
            futures = [
                executor.submit(first_pass, decision, report, input_photos[report["image_path"]][profile])
                for report in reports
            ]
            results = [future.result() for future in futures]
            profiles[profile] = [record for record, _ in results]
            first_results[profile] = [result for _, result in results]

    has_reason = callable(getattr(decision, "reason_review", None))
    if has_reason:
        with ThreadPoolExecutor(max_workers=args.reason_workers) as executor:
            for profile in PROFILES:
                jobs = [
                    (i, executor.submit(
                        second_pass, decision, report, input_photos[report["image_path"]][profile], result,
                    ))
                    for i, (report, result) in enumerate(zip(reports, first_results[profile]))
                    if result["mode"] == "zoro" and result["review"]
                ]
                print(f"{profile}: {len(jobs)} offline review attempts", flush=True)
                for i, future in jobs:
                    review = future.result()
                    record = profiles[profile][i]
                    record["reason_review"] = review
                    if review["status"] == "done":
                        record["effective_review"] = review["needs_review"]

    return {
        "revision": args.revision,
        "decision_sha256": hashlib.sha256((args.source / "decision.py").read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(manifest).hexdigest(),
        "severity_threshold": threshold,
        "config": {
            "sample_seed": SEED,
            "sample_size": len(reports),
            "unique_photos": len(input_photos),
            "pillow_version": pillow_version,
            "first_timeout_seconds": FIRST_TIMEOUT,
            "first_workers": args.workers,
            "reason_workers": args.reason_workers if has_reason else None,
            "reason_timeout_seconds": getattr(decision, "REVIEW_TIMEOUT", None),
            "clean": "prepare_photo then RGB PNG",
            "gaussian": {
                "sigma_pixel_intensity": SIGMA,
                "noise_seed": "SHA-256 of '119:' plus the dataset image path",
                "quantile_bins": 256,
                "channels": "one shared luminance offset per pixel, clamped to 0–255",
            },
            "blur": {"pillow_gaussian_radius_pixels": BLUR_RADIUS},
        },
        "summary": {name: summarize(rows, threshold) for name, rows in profiles.items()},
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Archived repo containing decision.py")
    parser.add_argument("--revision", help="Source commit ID for the output manifest")
    parser.add_argument("--data", type=Path, required=True, help="Path to test119.json")
    parser.add_argument("--images", type=Path, required=True, help="Root of CrisisMMD photo files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=positive_int, default=100)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--reason-workers", type=positive_int, default=2)
    args = parser.parse_args()
    result = evaluate(args)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Saved results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
