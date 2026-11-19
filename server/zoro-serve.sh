#!/usr/bin/env bash
# Zoro-1 server: typed decisions (/v1/systemone), reasoned decisions and OpenAI-compatible chat from one engine.
# Run by the systemd unit "zoro" (do not start by hand while the service is up: the port would clash).
#   sudo systemctl restart zoro | systemctl status zoro | tail -F ~/zoro-serve.log
# Binds to localhost only; laptops reach it through an SSH tunnel.
set -euo pipefail
source ~/venvs/zoro/bin/activate
cd ~/trida-stack/inference/zoro-1
exec python serve.py --mode decide --host 127.0.0.1 --port "${ZORO_PORT:-30001}" --log "${ZORO_LOG:-$HOME/zoro-serve.log}" --mem-fraction "${ZORO_MEM_FRACTION:-0.8}" "$@"
