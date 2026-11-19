"""Same six heads vs the requested eight heads on a small train-only timing control."""
import json, random, statistics, time
from pathlib import Path
from unittest.mock import patch
import before, after
from bench_bundle import single

rng=random.Random(119)
train=rng.sample([r for r in json.loads(Path('/ephemeral/crisis_bench/scenarios.json').read_text()) if r['split']=='train'],48)
root=Path('/ephemeral/crisis_bench/crisismmd')
labels=['original6','grouped6','grouped8']
out={name:[] for name in labels}

def grouped_six(row):
    original=after._systemone
    def send(url,state,questions,image,deadline):
        keep={k:v for k,v in questions.items() if k not in ('life_threat','ongoing_hazard')}
        return original(url,state,keep,image,deadline)
    start=time.perf_counter()
    data=(root/row['image_path']).read_bytes()
    with patch.object(after,'_systemone',send):
        evidence,model=after._model_evidence(row['text'],data,{},start+after.DECISION_TIMEOUT)
    before._model_result(evidence,model,{})
    return (time.perf_counter()-start)*1000

def measure(key,row):
    if key=='grouped6':
        return grouped_six(row)
    v=single(row,root,before if key=='original6' else after)
    assert v['mode']=='zoro'
    return v['wall_ms']
for name in labels: measure(name,train[0])
for i,row in enumerate(train):
    # Rotate all six orders equally.
    from itertools import permutations
    for name in list(permutations(labels))[i%6]:
        out[name].append(measure(name,row))
result={'train_n':len(train),'median_ms':{k:statistics.median(v) for k,v in out.items()},'timings_ms':out}
Path('measured/train-control.json').write_text(json.dumps(result,indent=2))
print(result['median_ms'],flush=True)
