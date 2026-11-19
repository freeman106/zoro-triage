#!/usr/bin/env bash
# Rebuild the Zoro-1 serving environment on a fresh Brev L40S instance (same AWS g6e image as the old one).
# Stage 1 (no HF token needed): venvs, packages, trida-stack + patch, serve script, systemd unit.
# Model download and service start happen after the user runs `hf auth login`.
set -euo pipefail
cd ~
UV=~/.local/bin/uv
[ -x "$UV" ] || { curl -LsSf https://astral.sh/uv/install.sh | sh; UV=~/.local/bin/uv; }

# Same storage layout as the old instance: HF cache and data live under /ephemeral on the root disk.
sudo mkdir -p /ephemeral && sudo chown ubuntu:ubuntu /ephemeral
mkdir -p /ephemeral/cache/huggingface ~/.cache
[ -e ~/.cache/huggingface ] || ln -s /ephemeral/cache/huggingface ~/.cache/huggingface

# Zoro serving venv (pinned to the versions verified on the old instance).
[ -d ~/venvs/zoro ] || $UV venv -q ~/venvs/zoro --python 3.11 --seed
# Exact freeze of the working environment (210 packages), from PyPI as originally installed
# (the old torch reports 2.14.1+cu130 although its distribution version is plain 2.14.1).
$UV pip install -q --python ~/venvs/zoro/bin/python -r ~/zoro-requirements.txt

# Demo backend venv (only needs Pillow + requests; the old one also had benchmark extras).
[ -d ~/venvs/bench ] || $UV venv -q ~/venvs/bench --python 3.11 --seed
$UV pip install -q --python ~/venvs/bench/bin/python pillow requests huggingface_hub

# Serving code at the same commit, plus tonight's verified patch (CPU work off the event loop).
[ -d ~/trida-stack ] || git clone -q https://github.com/trillion-labs/trida-stack ~/trida-stack
cd ~/trida-stack && git checkout -q 0ddcf516e458a82892d7fe0d0dabeb113d3cf424
git apply --check ~/zoro-patch.diff 2>/dev/null && git apply ~/zoro-patch.diff || echo "patch already applied or failed: check git diff"
cd ~

# systemd unit identical to the old instance.
sudo tee /etc/systemd/system/zoro.service >/dev/null <<'UNIT'
[Unit]
Description=Zoro-1 decision server (trida-stack on SGLang), 127.0.0.1:30001
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment=LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib:/usr/local/cuda-13.2/lib64:/usr/local/cuda-13.2:/usr/local/cuda-13.2/targets/x86_64-linux/lib/:/usr/local/cuda-13.2/extras/CUPTI/lib64:/usr/local/lib:/usr/lib
Environment=PATH=/home/ubuntu/venvs/zoro/bin:/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/home/ubuntu/zoro-serve.sh
Restart=always
RestartSec=10
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
UNIT
chmod +x ~/zoro-serve.sh ~/zoro 2>/dev/null || true
sudo systemctl daemon-reload
echo "STAGE1_DONE: $(~/venvs/zoro/bin/python -c 'import sglang, torch; print(sglang.__version__, torch.__version__, torch.cuda.is_available())')"
