"""Measure the photo-only field-photo head on the same 400 reports, without gold scene labels."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
import base64
import json
import sys
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"demo119"))
from scene_question import FIELD_PHOTO_QUESTION
from evaluate import PARTS


def one(row):
    path=Path("crisismmd")/row["image_path"]
    image=base64.b64encode(path.read_bytes()).decode()
    body={"model":"zoro-1","state":"첨부 사진","images":[image],"questions":{
        "severe":PARTS["severe"], "field_photo":{"type":"noul","instructions":FIELD_PHOTO_QUESTION}}}
    request=Request("http://127.0.0.1:30001/v1/systemone",json.dumps(body,ensure_ascii=False).encode(),
                    {"Content-Type":"application/json"})
    with urlopen(request,timeout=30) as f:
        response=json.load(f)
    probs={k:response["answers"][k]["noul"] for k in body["questions"]}
    assert all(type(p) in (float,int) and 0<=p<=1 for p in probs.values())
    return dict(id=row["id"],image_id=row["image_id"],model=response["model"],**probs)


if __name__=="__main__":
    rows=json.loads(Path("test119.json").read_text())
    assert len(rows)==len({r["id"] for r in rows})==400
    directory=Path("scene119-v1")
    directory.mkdir(exist_ok=True)
    thumbs=directory/"thumbs"
    thumbs.mkdir(exist_ok=True)
    for r in rows:
        dest=thumbs/(r["id"]+".jpg")
        if not dest.exists():
            with Image.open(Path("crisismmd")/r["image_path"]) as img:
                img.thumbnail((240,155))
                img.convert("RGB").save(dest,quality=82)
    out=directory/"predictions.jsonl"
    seen={v["id"] for v in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
    with ThreadPoolExecutor(4) as executor,out.open("a") as f:
        for result in executor.map(one,[r for r in rows if r["id"] not in seen]):
            f.write(json.dumps(result)+"\n")
            f.flush()
            seen.add(result["id"])
            if len(seen)%100==0:
                print("field",len(seen),flush=True)
    preds=list(map(json.loads,out.read_text().splitlines()))
    assert len(preds)==400 and seen=={r["id"] for r in rows}
    print("flagged",sum(p["field_photo"]<0.5 for p in preds),"/",len(preds),"reports",flush=True)
