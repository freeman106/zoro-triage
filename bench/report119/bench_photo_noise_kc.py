"""Re-fit main's KcELECTRA + CLIP baseline and score exact section 21 inputs."""

import argparse
import hashlib
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import time

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
import torch
import transformers

from bench_photo_noise_llm import shared_inputs
from bench_photo_noise import positive_int
from evaluate import urgency
from shift_models import predict_kc
from supervised import B32, EVENTS, GROUPS, KC, Photos, batched, fit_kc


HERE = Path(__file__).resolve().parent
SEED = 42
TRAIN_REPORT_COUNT = 1500
TRAIN_PHOTO_COUNT = 2468


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def training_inputs(args):
    reference = json.loads(args.training_reference.read_bytes())
    if reference["method"] != "kc" or reference["n_train"] != TRAIN_REPORT_COUNT or reference["seed"] != SEED:
        raise ValueError("Expected main's 1,500-report KcELECTRA reference")
    scenarios = json.loads(args.scenarios.read_bytes())
    train = [row for row in scenarios if row["split"] == "train"]
    test = [row for row in scenarios if row["split"] == "test"]
    train = Random(SEED).sample(train, len(train))
    if len(train) != TRAIN_REPORT_COUNT or [row["id"] for row in train] != reference["train_ids"]:
        raise ValueError("Training report IDs or training order differ from main")
    if len(test) != 400 or {row["image_id"] for row in train} & {row["image_id"] for row in test}:
        raise ValueError("Bad test size or an image shared by the report train/test splits")
    if test != json.loads(args.data.read_bytes()):
        raise ValueError("The full test split differs from the section 21 report dataset")
    photo_manifest = args.crisis_root / "damage/train.json"
    raw = photo_manifest.read_text()
    try:
        photos = json.loads(raw)
    except json.JSONDecodeError:
        photos = [json.loads(line) for line in raw.splitlines() if line.strip()]
    paths = {row["image_path"] for row in photos}
    if len(photos) != TRAIN_PHOTO_COUNT or len(paths) != TRAIN_PHOTO_COUNT:
        raise ValueError(f"Expected all {TRAIN_PHOTO_COUNT} unique official damage-train photos")
    if {row["image_id"] for row in photos} & {row["image_id"] for row in test}:
        raise ValueError("Auxiliary damage training photos overlap with report test photos")
    missing = [path for path in paths if not (args.crisis_root / path).is_file()]
    if missing:
        raise ValueError(f"{len(missing)} official damage training photos are missing")
    return train, photos, photo_manifest, reference["pretrained_revisions"]


def training_spec(args, photo_manifest, revisions):
    sources = {KC: args.kc_source, B32: args.clip_source}
    for name, path in sources.items():
        if not path.is_dir():
            raise ValueError(f"Missing pinned {name} snapshot: {path}")
        if path.parent.name == "snapshots" and path.name != revisions[name]:
            raise ValueError(f"The {name} snapshot revision differs from main")
    return {
        "method": "retrained_main_section_17",
        "seed": SEED,
        "training_reports": TRAIN_REPORT_COUNT,
        "training_photos": TRAIN_PHOTO_COUNT,
        "train_batch_size": 16,
        "feature_batch_size": args.batch_size,
        "training_device": args.device,
        "feature_device": args.feature_device,
        "feature_dtype": "float32",
        "pretrained_revisions": revisions,
        "source_config_sha256": {name: sha256(path / "config.json") for name, path in sources.items()},
        "report_train_manifest_sha256": sha256(args.scenarios),
        "training_reference_sha256": sha256(args.training_reference),
        "damage_train_manifest_sha256": sha256(photo_manifest),
        "versions": {
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }


def prepare_weights(directory, spec):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "training-spec.json"
    if manifest.is_file():
        if json.loads(manifest.read_text()) != spec:
            raise ValueError(f"Saved weights belong to another training configuration: {directory}")
    elif any(directory.iterdir()):
        raise ValueError(f"Existing weights have no training specification: {directory}")
    else:
        manifest.write_text(json.dumps(spec, indent=2) + "\n")


def score_photos(args, training_rows, payloads):
    encoder = Photos(device="cpu", root=args.crisis_root, source=str(args.clip_source))
    encoder.model.to(args.feature_device)
    encoder.device = args.feature_device
    checkpoint = args.weights / "clip-photo-heads.joblib"
    if not checkpoint.is_file():
        print(f"Extract CLIP FP32 features for {len(training_rows)} train photos on {args.feature_device}", flush=True)
        features = batched(encoder, training_rows, args.batch_size)
        heads = {
            "severe": LogisticRegression(C=4, max_iter=1000).fit(
                features, [row["label"] == "severe_damage" for row in training_rows],
            ),
            "group": LogisticRegression(C=4, max_iter=1000).fit(
                features, [EVENTS[row["event_name"]] for row in training_rows],
            ),
        }
        with TemporaryDirectory(dir=args.weights) as temporary:
            staged = Path(temporary) / checkpoint.name
            joblib.dump(heads, staged)
            staged.replace(checkpoint)
    heads = joblib.load(checkpoint)

    predictions = {}
    with TemporaryDirectory(dir=args.weights) as temporary:
        encoder.root = Path(temporary)
        for profile, photos in payloads.items():
            print(f"Predict CLIP on {profile} photos", flush=True)
            paths = list(photos)
            rows = []
            for path in paths:
                digest, png = photos[path]
                filename = f"{digest}.png"
                (encoder.root / filename).write_bytes(png)
                rows.append({"image_path": filename})
            features = batched(encoder, rows, args.batch_size)
            predictions[profile] = {
                path: (bool(severe), str(group))
                for path, severe, group in zip(
                    paths, heads["severe"].predict(features), heads["group"].predict(features), strict=True
                )
            }
    return predictions


def score_texts(args, training_rows, reports):
    checkpoint = args.weights / f"kc-{TRAIN_REPORT_COUNT}"
    if not checkpoint.is_dir():
        print(f"Fit KcELECTRA on {len(training_rows)} reports on {args.device}", flush=True)
        torch.manual_seed(SEED)
        with TemporaryDirectory(dir=args.weights) as temporary:
            staged = Path(temporary) / checkpoint.name
            fit_kc(training_rows, device=args.device, source=str(args.kc_source), save_dir=staged)
            staged.rename(checkpoint)
        if args.device == "cuda":
            torch.cuda.empty_cache()
    predictor = predict_kc(checkpoint, args.device)
    return {
        row["id"]: (bool(risk), bool(now), GROUPS[claim])
        for row, (risk, now, claim) in zip(reports, batched(predictor, reports, args.batch_size), strict=True)
    }


def weights_manifest(directory):
    paths = [directory / "clip-photo-heads.joblib", *(directory / f"kc-{TRAIN_REPORT_COUNT}").iterdir()]
    return {str(path.relative_to(directory)): sha256(path) for path in sorted(paths) if path.is_file()}


def evaluate(args):
    zoro, reports, payloads = shared_inputs(args)
    train, photo_train, photo_manifest, revisions = training_inputs(args)
    spec = training_spec(args, photo_manifest, revisions)
    prepare_weights(args.weights, spec)

    start = time.monotonic()
    photos = score_photos(args, photo_train, payloads)
    if args.feature_device == "cuda":
        torch.cuda.empty_cache()
    photo_time = round(time.monotonic() - start, 3)
    start = time.monotonic()
    text = score_texts(args, train, reports)
    text_time = round(time.monotonic() - start, 3)

    profiles = {}
    for profile, profile_predictions in photos.items():
        records = []
        for row in reports:
            path = row["image_path"]
            damage, group = profile_predictions[path]
            risk, now, claim = text[row["id"]]
            records.append({
                "id": row["id"],
                "image_sha256": payloads[profile][path][0],
                "gold_severe": row["image_severe"],
                "gold_mismatch": row["gold_mismatch"],
                "gold_urgency": row["gold_urgency"],
                "severe_prediction": damage,
                "urgency": urgency(risk, damage, now),
                "mismatch": claim != group,
                "claim_group": claim,
                "photo_group": group,
                "people": risk,
                "now": now,
                "invalid_fields": [],
                "transport_error": False,
                "hard_review": False,
                "reason_review": None,
            })
        profiles[profile] = records
    legacy_accuracy = {}
    for profile, rows in profiles.items():
        consistent = [row for row in rows if not row["gold_mismatch"]]
        legacy_accuracy[profile] = {
            "correct": sum(row["urgency"] == row["gold_urgency"] for row in consistent),
            "total": len(consistent),
        }
    return {
        "model": "KcELECTRA+CLIP+rules",
        "backend": "transformers+scikit-learn",
        "mode": "supervised_categorical",
        "weights_format": "retrained",
        "review_mode": None,
        "runtime": spec,
        "urgency_rule": "main_section_17_legacy",
        "mismatch_rule": "main_section_17_all_group",
        "legacy_urgency_accuracy": legacy_accuracy,
        "weights_sha256": weights_manifest(args.weights),
        "dataset_sha256": zoro["dataset_sha256"],
        "reference_sha256": sha256(args.zoro_result),
        "runner_sha256": sha256(Path(__file__)),
        "timings": {"photo_wall_seconds": photo_time, "text_wall_seconds": text_time},
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "test119.json")
    parser.add_argument("--scenarios", type=Path, default=HERE / "scenarios.json")
    parser.add_argument("--training-reference", type=Path, default=HERE / "results/noisy-shift/kc-1500.json")
    parser.add_argument("--zoro-result", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", type=Path)
    inputs.add_argument("--prepared-photos", type=Path)
    for name in ("crisis-root", "clip-source", "kc-source", "weights", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--feature-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=positive_int, default=24)
    args = parser.parse_args()
    result = evaluate(args)
    args.out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    print("Finished supervised benchmark", flush=True)


if __name__ == "__main__":
    main()
