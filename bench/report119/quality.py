"""Audit slots and repair flagged generated messages using their assigned generator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
import json
from pathlib import Path
import re
import time

from generate import FIELDS, PROMPT, ROOT, render_batch, validate

CLAIM = {"fire": r"화재|산불|불[길이났나]|불\s*(났|나|붙|이|번|땜)|불입|불이에|불.?났|불\s*신고",
         "storm": r"태풍|침수|폭우|홍수|물에\s*잠|물[이 ]*차|물이\s*들|물난리|큰비",
         "quake": r"지진|땅이\s*흔들"}
PEOPLE = {"노부모":r"부모|어르신|부친|모친|어머니|아버지", "아이들":r"아이|애들|애기|아기|애랑|애하고|애가|애와",
          "옆집":r"할머|할매|할무|할매|노인|어르신", "차 안":r"차|차량", "가족 한":r"다리|출혈|피[가를]|피흘",
          "사람이 쓰러":r"쓰러|의식|실신|기절|정신을 잃", "건물 잔해":r"깔|끼|잔해|매몰",
          "연기를":r"연기|호흡|숨", "모두 대피":r"다친.{0,12}(없|업)|무사|대피|대핀|안 다|안다|부상.{0,5}없|안전"}
DAMAGE = r"무너|무너졋|붕괴|반파|완파|처참|지붕.{0,12}(날|무너)|쑥대밭|초토화|파손|파괴|산산조각|피해.{0,4}(심각|심하|심해|경미|가벼)"
PAST = r"어제|지난|예전|전날|뉴스|기록|정리.{0,8}(끝|됐|되)|보험|저번|며칠\s*전"

def issues(r,t):
    checks = {"지역": any(w[:2] in t for w in r["region"].split()), "재난 종류": re.search(CLAIM[r["claim_group"]], t), "물적 피해 단서 삭제": not re.search(DAMAGE,t),
              "과거 시점": r["time"] != "past" or re.search(PAST,t)}
    if r["time"] == "past" and re.search(r"(빨리|당장|어서|지금)\s*.{0,6}(와주|와 주|오세요|구해|구조해)|갇혀\s*있(어요|습니다)", t):
        checks["과거 사건에 현재 구조 요청을 추가하지 말 것"] = False
    for prefix, pattern in PEOPLE.items():
        if r["people_detail"].startswith(prefix):
            checks["위험 인원 상태"] = re.search(pattern,t)
    return [k for k,v in checks.items() if not v]


def fix(engine, rows, texts, cache):
    for i in range(0,len(rows),15):
        batch = rows[i:i+15]
        for attempt in range(5):
            bad = [r for r in batch if issues(r,texts[r["id"]])]
            if not bad:
                break
            items = []
            for r in bad:
                item = {k:r[k] for k in FIELDS}
                item["claim"] = "지진" if r["claim_group"] == "quake" else r["claim"]
                item["반드시_포함할_단어"] = {"quake":"지진", "storm": "태풍" if "태풍" in r["claim"] else "침수", "fire": "산불" if "산불" in r["claim"] else "화재"}[r["claim_group"]]
                items.append(item)
            prompt = PROMPT.split("입력:")[0] + " 반드시 지정한 단어를 글에 넣고 인명 슬롯을 충실히 반영하라. 건물 피해 묘사는 생략하라.\n"
            try:
                replacements = validate(render_batch(engine,items,prompt),bad)
            except (ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:
                print(engine,"repair retry",str(exc)[:100],flush=True)
                time.sleep(2)
                continue
            for r in bad:
                key = r["id"]
                t = replacements[key]
                if not issues(r,t):
                    texts[key] = t
                    path = cache / (key+".json")
                    tmp = path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(t,ensure_ascii=False))
                    tmp.replace(path)
        if any(issues(r,texts[r["id"]]) for r in batch):
            raise ValueError("Unresolved quality flags: "+str([r["id"] for r in batch if issues(r,texts[r["id"]])]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("split", choices=["test","all"])
    args=p.parse_args()
    source = ROOT/"scenarios.json"
    data=json.loads(source.read_text())
    rows=[r for r in data if args.split=="all" or r["split"]=="test"]
    cache=ROOT/"generated"
    texts={}
    for path in sorted(cache.glob("*-*-*.json")):
        for a in json.loads(path.read_text()):
            if a["id"] in texts:
                raise ValueError("Duplicate cached ID: "+a["id"])
            texts[a["id"]] = a["text"]
    repairs=cache/"repairs"
    repairs.mkdir(exist_ok=True)
    for r in rows:
        key=r["id"]
        if (repairs/(key+".json")).exists():
            texts[key]=json.loads((repairs/(key+".json")).read_text())
    bad=[r for r in rows if issues(r,texts[r["id"]])]
    print("rows",len(rows),"flagged",len(bad),flush=True)
    with ThreadPoolExecutor(4) as pool:
        jobs=[]
        for engine in ("claude","codex"):
            group=[r for r in bad if r["generator"]==engine]
            for i in range(0,len(group),15):
                jobs.append(pool.submit(fix,engine,group[i:i+15],texts,repairs))
        for job in jobs:
            job.result()
    for r in rows:
        r["text"]=texts[r["id"]]
    path = source if args.split=="all" else ROOT/"test119.json"
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=0))
    tmp.replace(path)
    print("verified",len(rows),path,flush=True)


if __name__=="__main__":
    main()
