"""Build photo+text 119 report scenarios on real CrisisMMD damage-labelled photos.

Gold labels come from fixed facts + a fixed triage rule, never from an LLM:
  image facts (human labels): disaster group from the event, severe damage or not
  text facts (chosen here):   claimed disaster, people at risk, ongoing or past
  urgency  = past -> non_urgent; people & severe -> top; people xor severe -> urgent; neither -> normal
  mismatch = claimed disaster group differs from the photo's (fire vs water/wind vs earthquake not mixed)
Usage: python build_scenarios.py <crisismmd_root> <out.json>
"""
import json, random, sys

ROOT, OUT = sys.argv[1], sys.argv[2]
GROUP = {"california_wildfires": "fire", "hurricane_harvey": "storm", "hurricane_irma": "storm", "hurricane_maria": "storm",
         "srilanka_floods": "storm", "mexico_earthquake": "quake", "iraq_iran_earthquake": "quake"}
CLAIM = {"fire": ["산불", "주택 화재", "건물 화재"], "storm": ["태풍 피해", "침수", "폭우로 인한 침수"], "quake": ["지진으로 건물 붕괴", "지진 피해"]}
PEOPLE = {"none": ["언급 없음", "모두 대피해서 다친 사람 없음"],
          "at_risk": ["노부모가 거동을 못 해 집 안에 고립", "아이들과 2층에 고립되어 못 나감", "옆집 할머니가 안에 있는데 연락이 안 됨", "차 안에 사람이 갇혀 있음"],
          "injured": ["가족 한 명이 다리를 다쳐 피를 흘림", "사람이 쓰러져서 의식이 없음", "건물 잔해에 깔린 사람이 있음", "연기를 마신 사람이 숨을 잘 못 쉼"]}
PAST = ["어제 찍은 사진을 공유", "지난주 일인데 보험 처리 문의", "뉴스에서 본 사진을 보내며 우리 동네도 괜찮은지 질문", "이미 정리가 끝났고 기록용으로 신고"]
REPORTER = ["당사자", "이웃", "멀리 사는 가족", "지나가던 사람"]
STYLE = ["다급하고 짧은 문자", "차분하고 자세한 설명", "맞춤법이 틀린 급한 문자", "어르신 말투", "횡설수설하는 말투"]
REGION = ["서울 은평구 불광동", "경기 남양주시 와부읍", "강원 강릉시 옥계면", "부산 사하구 하단동", "충북 청주시 오송읍", "경북 포항시 남구 오천읍",
          "전남 구례군 구례읍", "경남 창원시 마산합포구", "인천 미추홀구 용현동", "대구 북구 침산동", "울산 울주군 언양읍", "제주 서귀포시 표선면",
          "충남 아산시 배방읍", "전북 익산시 망성면", "광주 광산구 송정동", "강원 삼척시 원덕읍"]


def rows(split):
    txt = open(f"{ROOT}/damage/{split}.json").read()
    try:
        return json.loads(txt)
    except Exception:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def urgency(s):
    if s["time"] == "past":
        return "non_urgent"
    p, sev = s["people"] != "none", s["image_severe"]
    return "top" if p and sev else "urgent" if p or sev else "normal"


def build(split, n, seed):
    rng = random.Random(seed)
    imgs = rows(split)
    rng.shuffle(imgs)
    by_sev = {True: [x for x in imgs if x["label"] == "severe_damage"], False: [x for x in imgs if x["label"] != "severe_damage"]}
    out = []
    for i in range(n):
        r = rng.random()
        kind = "mismatch" if r < 0.2 else "past" if r < 0.35 else "consistent"
        sev = rng.random() < 0.5                       # balance severe / not severe photos
        pool = by_sev[sev]
        x = pool[i % len(pool)] if split == "test" else rng.choice(pool)
        g = GROUP[x["event_name"]]
        if kind == "mismatch":
            claim_group = "fire" if g != "fire" else rng.choice(["storm", "quake"])
        else:
            claim_group = g
        people = rng.choice(["none", "at_risk", "injured"])
        s = {"id": f"{split}-{i:04d}", "split": split, "image_id": x["image_id"], "image_path": x["image_path"], "event": x["event_name"],
             "image_group": g, "image_severe": sev, "image_label": x["label"], "claim_group": claim_group,
             "claim": rng.choice(CLAIM[claim_group]), "people": people, "people_detail": rng.choice(PEOPLE[people]),
             "time": "past" if kind == "past" else "now", "past_detail": rng.choice(PAST) if kind == "past" else "",
             "reporter": rng.choice(REPORTER), "style": rng.choice(STYLE), "region": rng.choice(REGION),
             "generator": "claude" if i % 2 == 0 else "codex"}
        s["gold_mismatch"] = claim_group != g
        s["gold_urgency"] = urgency(s)
        out.append(s)
    return out


data = build("train", 1500, 1) + build("test", 400, 2)
json.dump(data, open(OUT, "w"), ensure_ascii=False, indent=0)
import collections
for split in ("train", "test"):
    d = [s for s in data if s["split"] == split]
    print(split, len(d), "mismatch", sum(s["gold_mismatch"] for s in d),
          dict(collections.Counter(s["gold_urgency"] for s in d if not s["gold_mismatch"])),
          "unique images", len({s["image_id"] for s in d}))
