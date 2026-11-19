#!/usr/bin/env bash
set -euo pipefail
cd /ephemeral/crisis_bench
export HF_HOME=/ephemeral/cache/huggingface HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
for mode in mlp kc; do
    ~/venvs/bench/bin/python -u supervised.py "$mode"
done
