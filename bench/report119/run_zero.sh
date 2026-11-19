#!/usr/bin/env bash
set -euo pipefail
cd /ephemeral/crisis_bench
export HF_HOME=/ephemeral/cache/huggingface HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=
for mode in direct parts qwen qwen_parts; do
    ~/venvs/bench/bin/python -u evaluate.py "$mode" --data test119.json
done
