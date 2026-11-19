import sqlite3, json
from PIL import Image, ImageDraw, ImageFont
con = sqlite3.connect("file:/home/ubuntu/demo119/runtime-field-live/inbox.sqlite?mode=ro", uri=True)
rows = [json.loads(d) for (d,) in con.execute("SELECT data FROM cards")]
pick = [c for c in rows if any("사진 피해" in r and "불확실" in r for r in (c.get("decision") or {}).get("reasons", []))]
gold = {s["id"]: s for s in json.load(open("/ephemeral/crisis_bench/scenarios.json"))}
W, H, cols = 300, 225, 5
nrows = (len(pick) + cols - 1) // cols
sheet = Image.new("RGB", (W * cols, (H + 44) * nrows), "white")
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
draw = ImageDraw.Draw(sheet)
for n, c in enumerate(pick):
    im = Image.open(c["path"]).convert("RGB")
    im.thumbnail((W, H))
    x, y = (n % cols) * W, (n // cols) * (H + 44)
    sheet.paste(im, (x + (W - im.width) // 2, y))
    d = c["decision"]
    g = gold.get(c.get("scenario_id") or "", {})
    lab = g.get("image_label", "?")
    event = c["path"].split("/")[-3]
    draw.text((x + 4, y + H + 2), "#%d p(severe)=%.2f %s" % (n + 1, d["probs"]["severe"], d["urgency"]), fill="black", font=font)
    draw.text((x + 4, y + H + 22), "label=%s %s" % (lab, event), fill="black", font=font)
    print(n + 1, round(d["probs"]["severe"], 2), d["urgency"], lab, event)
sheet.save("/home/ubuntu/uncertain_photos.jpg", quality=85)
