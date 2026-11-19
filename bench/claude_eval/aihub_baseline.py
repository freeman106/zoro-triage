"""Supervised baseline on the same 900 validation calls Zoro was scored on.

Train: AI Hub Training labels (TL_*). Test: sample900.json (drawn from Validation, disjoint from Training).
Model: char 2-4gram TF-IDF + logistic regression on the full call transcript, one model per target.
Learning curve over training sizes to show how many labelled calls a trained model needs.
"""
import glob, json, random, sys, time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = sys.argv[1]
train = []
for f in glob.glob(ROOT + "/train/**/*.json", recursive=True):
    x = json.load(open(f))
    x = x if isinstance(x, dict) else x[0]
    u = x.get("utterances") or x.get("utterences") or []
    text = "\n".join(("접수요원" if str(y.get("speaker")) == "0" else "신고자") + ": " + y.get("text", "").strip() for y in u if y.get("text", "").strip())
    if text:
        train.append({"dialogue": text, "urgency": x.get("urgencyLevel"), "large": x.get("disasterLarge")})
test = json.load(open(ROOT + "/sample900.json"))
zoro = json.load(open(ROOT + "/aihub_out.json"))
print("train calls:", len(train), " test calls:", len(test), flush=True)

targets = {"구급 vs 비구급": lambda r: r["large"] == "구급", "대분류 4종": lambda r: r["large"], "긴급도 상중하": lambda r: r["urgency"]}
zoro_pred = {"구급 vs 비구급": lambda i: zoro[i]["large"] == "구급", "대분류 4종": lambda i: zoro[i]["large"], "긴급도 상중하": lambda i: zoro[i]["urgency"]}
rng = random.Random(0)
rng.shuffle(train)
for name, y in targets.items():
    yt = [y(r) for r in test]
    zacc = accuracy_score(yt, [zoro_pred[name](r["id"]) for r in test])
    line = [f"{name}: Zoro(0건) {zacc:.3f}"]
    for n in (100, 1000, 10000, len(train)):
        sub = train[:n]
        t = time.time()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, max_features=300000)
        X = vec.fit_transform([r["dialogue"] for r in sub])
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight=None).fit(X, [y(r) for r in sub])
        acc = accuracy_score(yt, clf.predict(vec.transform([r["dialogue"] for r in test])))
        line.append(f"TF-IDF+LR({n}건) {acc:.3f}")
    print(" | ".join(line), flush=True)
