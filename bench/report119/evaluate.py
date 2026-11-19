"""Batch-one zero-shot evaluation. Run from /ephemeral/crisis_bench."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path("crisismmd")
CRITERIA = {
    "top": "현재 사건이고, 글에 고립·부상 등 위험 인원이 있으며 사진에서도 심각한 물적 피해가 보인다.",
    "urgent": "현재 사건이고, 글의 위험 인원과 사진의 심각한 물적 피해 중 정확히 하나가 있다.",
    "normal": "현재 사건이고, 글에 위험 인원이 없으며 사진에 심각한 물적 피해도 없다.",
    "non_urgent": "이미 지난 사건에 대한 공유·문의·기록이다. 당시 인명 위험과 사진 피해는 무관하다.",
}
DIRECT = {
    "urgency": {"type": "choice", "instructions": "이 합성 신고의 긴급도를 아래 실험 규칙으로 선택하라.", "criteria": CRITERIA},
    "mismatch": {"type": "noul", "instructions": "글이 주장하는 재난 종류(화재, 폭풍·홍수, 지진)와 사진의 재난 종류가 서로 다른가?"},
}
PARTS = {
    "people": {"type": "noul", "instructions": "신고 글에서 사건 당시 고립되거나 부상당한 등 위험한 사람이 언급되는가?"},
    "severe": {"type": "noul", "instructions": "사진에 구조물 붕괴, 지붕 소실, 도로 유실 등 심각한 물적 피해가 보이는가?"},
    "now": {"type": "noul", "instructions": "글은 사건이 현재 진행 중이라고 신고하는가? 과거 사진 공유나 문의는 아니오."},
}


def urgency(people, severe, now):
    if not now:
        return "non_urgent"
    return ("normal", "urgent", "top")[int(people) + int(severe)]


def zoro_predictor(mode):
    import requests
    session = requests.Session()

    def call(state, questions, img=None):
        body = {"model": "zoro-1", "state": state, "questions": questions}
        if img:
            body["images"] = [img]
        res = session.post("http://127.0.0.1:30001/v1/systemone", json=body, timeout=180)
        res.raise_for_status()
        return res.json()["answers"]

    def predict(row):
        img = base64.b64encode((ROOT / row["image_path"]).read_bytes()).decode()
        state = {"신고": row["text"]}
        if mode == "direct":
            a = call(state, DIRECT, img)
            return {"urgency": a["urgency"]["choice"], "mismatch": a["mismatch"]["noul"] >= 0.5}
        a = call(state, {k: PARTS[k] for k in ("people", "now")})
        b = call("첨부 사진", {"severe": PARTS["severe"]}, img)
        c = call(state, {"mismatch": DIRECT["mismatch"]}, img)
        flags = {k: v["noul"] >= 0.5 for k, v in (a | b).items()}
        return dict(flags, urgency=urgency(**flags), mismatch=c["mismatch"]["noul"] >= 0.5)
    return predict


def qwen_predictor(mode="qwen"):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(name, max_pixels=448*448)
    def ask(questions, state, img=None):
        output_format = ("urgency는 criteria의 키, mismatch는 JSON boolean."
                         if questions is DIRECT else "각 질문 키의 값은 JSON boolean.")
        prompt = json.dumps(questions, ensure_ascii=False) + "\nJSON 객체 하나로 답하라. " + output_format + state
        content = [] if img is None else [{"type": "image"}]
        content.append({"type": "text", "text": prompt})
        msg = [{"role": "user", "content": content}]
        rendered = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[rendered], images=None if img is None else [img], return_tensors="pt").to("cuda")
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        answer = processor.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        try:
            obj, _ = json.JSONDecoder().raw_decode(answer[answer.index("{"):])
        except ValueError:
            obj = {}
        return obj if isinstance(obj, dict) else {}, answer

    @torch.inference_mode()
    def predict(row):
        with Image.open(ROOT / row["image_path"]) as f:
            img = f.convert("RGB")
        state = " 신고 글:\n" + row["text"]
        if mode == "qwen":
            obj, raw = ask(DIRECT, state, img)
            level = obj.get("urgency")
            mismatch = obj.get("mismatch")
            result = {"urgency": level if isinstance(level, str) and level in CRITERIA else "invalid",
                      "mismatch": mismatch if isinstance(mismatch, bool) else None}
            if not obj:
                result["raw"] = raw
            return result

        calls = [({k: PARTS[k] for k in ("people", "now")}, state, None),
                 ({"severe": PARTS["severe"]}, " 첨부 사진", img),
                 ({"mismatch": DIRECT["mismatch"]}, state, img)]
        result = {}
        for questions, context, image in calls:
            obj, raw = ask(questions, context, image)
            for key in questions:
                result[key] = obj.get(key) if isinstance(obj.get(key), bool) else None
                if result[key] is None:
                    result.setdefault("invalid_raw", {})[key] = raw
        flags = {key: result[key] for key in PARTS}
        result["urgency"] = urgency(**flags) if None not in flags.values() else "invalid"
        return result
    return predict


def summary(test, preds):
    from sklearn.metrics import accuracy_score, f1_score
    ok = [r for r in test if not r["gold_mismatch"]]
    y = [r["gold_urgency"] for r in ok]
    p = [preds[r["id"]]["urgency"] for r in ok]
    tp = sum(r["gold_mismatch"] and preds[r["id"]]["mismatch"] is True for r in test)
    pos = sum(preds[r["id"]]["mismatch"] is True for r in test)
    return dict(n=len(test), n_urgency=len(ok), accuracy=accuracy_score(y, p),
                macro_f1=f1_score(y, p, labels=list(CRITERIA), average="macro", zero_division=0),
                precision=tp / pos if pos else 0, recall=tp / sum(r["gold_mismatch"] for r in test),
                invalid=sum(preds[r["id"]]["mismatch"] is None or preds[r["id"]]["urgency"] == "invalid" for r in test))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["direct", "parts", "qwen", "qwen_parts"])
    parser.add_argument("--data", default="scenarios.json")
    args = parser.parse_args()
    test = [r for r in json.loads(Path(args.data).read_text()) if r["split"] == "test"]
    assert len(test) == 400 and all(r["text"] for r in test)
    signature = hashlib.sha256(json.dumps(test, sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()[:12]
    path = Path(f"pred119-{args.mode}-{signature}.jsonl")
    predict = qwen_predictor(args.mode) if args.mode.startswith("qwen") else zoro_predictor(args.mode)
    preds = {a["id"]: a for a in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
    for _ in range(2):
        predict(test[0])
    with path.open("a") as f:
        for i, row in enumerate(test):
            if row["id"] in preds:
                continue
            t = time.perf_counter()
            a = predict(row)
            a.update(id=row["id"], ms=(time.perf_counter()-t)*1000)
            f.write(json.dumps(a, ensure_ascii=False)+"\n")
            f.flush()
            preds[row["id"]] = a
            if (i+1)%25 == 0:
                print(args.mode, i+1, flush=True)
    print(args.mode, summary(test, preds), "median_ms", statistics.median(a["ms"] for a in preds.values()), flush=True)


if __name__ == "__main__":
    main()
