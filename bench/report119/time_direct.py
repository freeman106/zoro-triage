"""Remeasure direct Zoro with no other process loading or running a GPU model."""
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from evaluate import zoro_predictor

test = json.loads(Path("test119.json").read_text())
predict = zoro_predictor("direct")
predict(test[-1])
latencies = []
for row in test:
    t = perf_counter()
    predict(row)
    latencies.append((perf_counter()-t)*1000)
    if len(latencies) % 50 == 0:
        print("direct latency sample",len(latencies),flush=True)
Path("direct119-ms.json").write_text(json.dumps(latencies))
print("direct isolated",len(latencies),"median_ms",median(latencies),flush=True)
