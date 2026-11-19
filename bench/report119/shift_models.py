"""Fit clean-text baselines, save the fitted models, then score paired clean/noisy reports."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
from tempfile import NamedTemporaryFile, TemporaryDirectory

from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from torch import nn
import transformers
from transformers import AutoModel, AutoTokenizer

from evaluate import urgency
from shift_data import load_pairs
from supervised import (
    B32, EVENTS, GROUPS, KC, MULTI,
    Photos, TextClip, batched, fit_kc, kc_predictor,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def embeddings(rows, field, source, make_encoder, cache, device, size):
    # Preserve duplicates and order when evaluating the prior mini-batched feature pipeline.
    signature = digest([str(source), field, device, size, [r[field] for r in rows]])
    file = cache / f"{field}-{signature}.npy"
    if not file.is_file():
        print("Extract", field, len(rows), device, flush=True)
        encoder = make_encoder()
        features = batched(encoder, rows, size)
        del encoder
        with NamedTemporaryFile(dir=cache, suffix=".npy.tmp", delete=False) as stream:
            np.save(stream, features, allow_pickle=False)
        Path(stream.name).replace(file)
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.load(file, allow_pickle=False)


def mlp():
    return MLPClassifier(hidden_layer_sizes=(128,), solver="lbfgs", alpha=0.1, max_iter=300, random_state=42)


def evaluate(model, n, train, test_sets, predictions, args, refs):
    records = []
    stats = {}
    for style, test in test_sets.items():
        pred = predictions[style]
        if len(pred) != len(test):
            raise ValueError("Prediction count differs from the test count")
        correct = dict.fromkeys(("urgency", "now", "people"), 0)
        for i, (row, item) in enumerate(zip(test, pred)):
            gold = {"urgency": row["gold_urgency"], "now": row["time"] == "now", "people": row["people"] != "none"}
            for target in correct:
                if target != "urgency" or not row["gold_mismatch"]:
                    correct[target] += int(item[target] == gold[target])
            if style == "clean":
                records.append({"id": row["id"], "gold": gold, "matched": not row["gold_mismatch"]})
            if records[i]["id"] != row["id"]:
                raise ValueError("Clean and noisy predictions are not aligned")
            records[i][style] = item
        stats[style] = {key: {"correct": num, "total": 303 if key == "urgency" else 400}
                        for key, num in correct.items()}
    info = {"method": model, "n_train": n, "seed": 42, "train_ids": [r["id"] for r in train],
            "metrics": stats, "records": records, "pretrained_revisions": refs,
            "feature_device": args.feature_device, "training_device": args.device if model == "kc" else "cpu",
            "versions": {"torch": str(torch.__version__), "transformers": transformers.__version__,
                         "sklearn": sklearn.__version__, "numpy": np.__version__}}
    write_json(args.out / f"{model}-{n}.json", info)
    print(model, n, json.dumps(stats), flush=True)


def predict_kc(checkpoint, device):
    model = AutoModel.from_pretrained(checkpoint).to(device)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    head = nn.Linear(model.config.hidden_size, 7).to(device)
    head.load_state_dict(torch.load(checkpoint / "multitask-head.pt", map_location=device, weights_only=True))
    return kc_predictor(model, tok, head, device)


def run_kc(train, sets, test_images, photo_features, weights, args, sources):
    image_root = args.data / "crisismmd"
    raw = (image_root / "damage/train.json").read_text()
    try:
        real = json.loads(raw)
    except json.JSONDecodeError:
        real = [json.loads(line) for line in raw.splitlines() if line.strip()]
    real = [r for r in real if (image_root / r["image_path"]).is_file()]
    if len(real) != 2468 or {r["image_id"] for r in real} & {r["image_id"] for r in sets["clean"]}:
        raise ValueError("The auxiliary training photos differ from the original 2,468 photos or overlap test")
    feats = photo_features(real)
    heads = {
        "severe": LogisticRegression(C=4, max_iter=1000).fit(feats, [r["label"] == "severe_damage" for r in real]),
        "group": LogisticRegression(C=4, max_iter=1000).fit(feats, [EVENTS[r["event_name"]] for r in real]),
    }
    img_heads = weights / "clip-photo-heads.joblib"
    joblib.dump(heads, img_heads)
    heads = joblib.load(img_heads)
    severe = heads["severe"].predict(test_images)
    group = heads["group"].predict(test_images)

    for n in args.sizes:
        subset = train[:n]
        checkpoint = weights / f"kc-{n}"
        signature = digest([subset, str(sources[KC]), str(torch.__version__), args.device, "seed42-b16-v1"])
        if not checkpoint.is_dir():
            print("Fit KcELECTRA", n, args.device, flush=True)
            torch.manual_seed(42)
            with TemporaryDirectory(dir=weights) as directory:
                target = Path(directory) / "checkpoint"
                fit_kc(subset, device=args.device, source=str(sources[KC]), save_dir=target)
                (target / "fit-signature.txt").write_text(signature)
                target.rename(checkpoint)
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        if (checkpoint / "fit-signature.txt").read_text() != signature:
            raise ValueError(f"Checkpoint belongs to a different fit; choose a fresh output directory: {checkpoint}")
        predictor = predict_kc(checkpoint, args.device)
        preds = {}
        for style, rows in sets.items():
            people_now_claim = batched(predictor, rows, args.batch)
            preds[style] = [
                {"urgency": urgency(risk, damage, now), "people": bool(risk), "now": bool(now),
                 "mismatch": bool(GROUPS[claim] != photo_group)}
                for (risk, now, claim), damage, photo_group in zip(people_now_claim, severe, group)
            ]
        evaluate("kc", n, subset, sets, preds, args, {r: p.name for r, p in sources.items()})
        del predictor
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()


def run_mlp(train, sets, test_images, photo_features, weights, args, sources):
    def text_features(rows):
        return embeddings(rows, "text", sources[MULTI], lambda: TextClip(args.feature_device, str(sources[MULTI])),
                          args.out / "cache", args.feature_device, args.batch)

    clean_text = text_features(train)
    features = np.hstack((photo_features(train), clean_text))
    matched = [i for i, r in enumerate(train) if not r["gold_mismatch"]]
    heads = {
        "urgency": mlp().fit(features[matched], [train[i]["gold_urgency"] for i in matched]),
        "mismatch": mlp().fit(features, [r["gold_mismatch"] for r in train]),
        # Auxiliary text-only probes do not feed into the multimodal urgency head.
        "people": mlp().fit(clean_text, [r["people"] != "none" for r in train]),
        "now": mlp().fit(clean_text, [r["time"] == "now" for r in train]),
    }
    checkpoint = weights / "multilingual-clip-mlp-1500.joblib"
    joblib.dump(heads, checkpoint)
    heads = joblib.load(checkpoint)
    preds = {}
    for style, rows in sets.items():
        text = text_features(rows)
        joint = np.hstack((test_images, text))
        outputs = {k: h.predict(text if k in ("people", "now") else joint) for k, h in heads.items()}
        preds[style] = [{key: str(v[i]) if key == "urgency" else bool(v[i]) for key, v in outputs.items()}
                        for i in range(len(rows))]
    evaluate("mlp", len(train), train, sets, preds, args, {r: p.name for r, p in sources.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["kc", "mlp"])
    parser.add_argument("--data", type=Path, default=Path("/ephemeral/crisis_bench"))
    parser.add_argument("--noisy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--feature-device", choices=["cpu", "cuda"])
    parser.add_argument("--batch", type=int, default=24, help="Inference batch size; training uses the original batch of 16")
    parser.add_argument("--sizes", type=int, nargs="+", choices=[50, 200, 1500], default=[50, 200, 1500],
                        help="KcELECTRA training sizes")
    args = parser.parse_args()
    args.feature_device = args.feature_device or args.device
    train, clean, noisy = load_pairs(args.data / "scenarios.json", args.noisy)
    order = random.Random(42).sample(train, len(train))
    sets = {"clean": clean, "noisy": noisy}
    weights = args.out / "weights"
    cache = args.out / "cache"
    weights.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)
    repos = (B32, KC if args.mode == "kc" else MULTI)
    sources = {r: Path(hf_hub_download(r, "config.json", local_files_only=True)).parent for r in repos}
    def photo_features(rows):
        return embeddings(rows, "image_path", sources[B32], lambda: Photos(args.feature_device, args.data / "crisismmd",
                          str(sources[B32])), cache, args.feature_device, args.batch)
    test_images = photo_features(clean)
    if args.mode == "kc":
        run_kc(order, sets, test_images, photo_features, weights, args, sources)
    else:
        run_mlp(order, sets, test_images, photo_features, weights, args, sources)


if __name__ == "__main__":
    main()
