"""CLIP ViT-B/32 binary severe-damage head (trained on 2,468 labelled photos) on the same 193 held-out photos Zoro was scored on. Result: HOLDOUT_CLIP 157 193 81.35."""
import sys,json,joblib
from pathlib import Path
base=Path('/ephemeral/crisis_bench/noisy-supervised')
sys.path.insert(0,str(base))
from supervised import Photos,B32,batched
from huggingface_hub import hf_hub_download
data=base.parent
root=data/'crisismmd'
raw=(root/'damage/test.json').read_text().strip()
rows=json.loads(raw) if raw.startswith('[') else list(map(json.loads,raw.splitlines()))
used={r['image_path'] for r in json.loads((data/'scenarios.json').read_text())}
hold=[r for r in rows if r['image_path'] not in used]
gold=[r['label']=='severe_damage' for r in hold]
archive=json.loads((Path.home()/'severity_holdout.json').read_text())
assert gold==[r['gold'] for r in archive] and len(hold)==193
source=Path(hf_hub_download(B32,'config.json',local_files_only=True)).parent
x=batched(Photos('cpu',root,str(source)),hold)
head=joblib.load(base/'run-v1/weights/clip-photo-heads.joblib')['severe']
pred=head.predict(x)
num=int(sum(pred==gold)); print('HOLDOUT_CLIP',num,len(gold),100*num/len(gold),flush=True)
