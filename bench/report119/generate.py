"""Generate independent batches on the Mac, then merge only validated text fields."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
FIELDS = "id claim people_detail time past_detail reporter style region".split()
PROMPT = """합성 연구용 119 문자·앱 신고를 슬롯별로 작성한다. 한국어 1~4문장, 지역과 재난 주장, 신고자 관점, 말투를 반영한다.
사람의 상태와 시점을 명확히 한다. '언급 없음'은 사람에 관한 문장을 생략한다. 과거 사건의 인명 상태도 과거형으로 쓰고 현재 위험을 새로 만들지 않는다.
사진의 물적 피해 크기나 건물 상태는 쓰지 않는다. '지진으로 건물 붕괴'는 지진 신고로만 표현한다. 인명 슬롯의 잔해 언급은 허용한다.
가상의 문안만 작성한다. 각 원본 id와 text만 가진 JSON 배열을 출력한다. 도구를 쓰지 않는다. 입력:\n"""


def validate(items, rows):
    ids = [r["id"] for r in rows]
    if not isinstance(items, list) or len(items) != len(ids):
        raise ValueError("Unexpected batch length")
    texts = {}
    for item in items:
        if set(item) != {"id", "text"} or not isinstance(item["text"], str):
            raise ValueError("Unexpected output schema")
        value = item["text"].strip()
        if not re.search("[가-힣]", value) or len(value) > 650:
            raise ValueError("Invalid Korean message")
        texts[item["id"]] = value
    if set(texts) != set(ids):
        raise ValueError("Unexpected IDs")
    return texts


def generate(engine, prompt):
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "answer"
        if engine == "claude":
            cmd = ["claude", "-p", "--model", "haiku", "--tools", "", "--no-session-persistence"]
        else:
            cmd = ["codex", "exec", "-m", "gpt-6-luna", "--skip-git-repo-check", "--ephemeral",
                   "-s", "read-only", "-o", str(output), "-"]
        result = subprocess.run(cmd, cwd=tmp, input=prompt, capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout)[-1200:])
        raw = result.stdout if engine == "claude" else output.read_text()
        for start in re.finditer(r"\[", raw):
            try:
                items, _ = json.JSONDecoder().raw_decode(raw[start.start():])
            except ValueError:
                continue
            if isinstance(items, list) and items and all(isinstance(x, dict) for x in items):
                return items
        raise ValueError("Generator returned no JSON array: " + raw[:180])


def render_batch(engine, rows, instruction):
    # Render locations locally so external generators can use a neutral placeholder.
    fields = [dict(r, region="[지역]") for r in rows]
    prompt = instruction + " 지역명 위치에 [지역]을 정확히 한 번 쓰고 지명을 만들지 않는다.\n" + json.dumps(fields, ensure_ascii=False)
    items = generate(engine, prompt)
    texts = validate(items, rows)
    output = []
    for r in rows:
        text = texts[r["id"]]
        if "[지역]" in text:
            text = text.replace("[지역]", r["region"])
        else:
            text = r["region"] + ", " + text
        output.append({"id": r["id"], "text": text})
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["codex", "claude"])
    mode = p.parse_args().mode
    source = ROOT / "scenarios.json"
    data = json.loads(source.read_text())
    cache = ROOT / "generated"
    cache.mkdir(exist_ok=True)
    lock = (cache / (mode + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for engine in ("claude", "codex"):
        if engine != mode:
            continue
        rows = sorted((r for r in data if r["generator"] == engine), key=lambda r: r["split"] != "test")
        def process(i):
            batch = rows[i:i+15]
            prompt = PROMPT + json.dumps([{k: r[k] for k in FIELDS} for r in batch], ensure_ascii=False)
            key = hashlib.sha256(prompt.encode()).hexdigest()[:12]
            path = cache / f"{engine}-{i:04}-{key}.json"
            if not path.exists():
                for attempt in range(3):
                    try:
                        items = render_batch(engine, [{k: r[k] for k in FIELDS} for r in batch], PROMPT)
                        validate(items, batch)
                        temp = path.with_suffix(".tmp")
                        temp.write_text(json.dumps(items, ensure_ascii=False))
                        temp.replace(path)
                        break
                    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as e:
                        print(f"{engine} batch {i} failed: {e}", flush=True)
                        if attempt == 2:
                            raise
                        time.sleep(3)
            texts = validate(json.loads(path.read_text()), batch)
            for row in batch:
                row["text"] = texts[row["id"]]
            print(engine, "batch", i // 15 + 1, "validated", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(process, range(0, len(rows), 15)))


if __name__ == "__main__":
    main()
