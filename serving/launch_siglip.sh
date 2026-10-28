#!/bin/bash
# Start the SigLIP embedding server (serving/siglip_server.py) in the vLLM container, in the
# background, and wait until it is healthy:  serving/launch_siglip.sh <model name> [port]
# The model name selects configs/models/<name>.yaml (checkpoint, revision); the endpoint is
# http://127.0.0.1:<port>.
set -euo pipefail
NAME=${1:?model name}; PORT=${2:-8200}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF=${VLLM_SIF:-/lustre/work/pdl16831/shared/vllm-sifs/vllm_cuda13.sif}
HF_CACHE=${HF_HUB_CACHE:?set HF_HUB_CACHE to the Hugging Face cache}
LOGS=${OVANNOT_LOGS:-$ROOT/logs}; mkdir -p "$LOGS"
cfg() { grep -E "^$1:" "$ROOT/configs/models/$NAME.yaml" | head -1 | sed -E "s/^$1: *//; s/ *#.*//"; }
CKPT=$(cfg checkpoint); REV=$(cfg revision)
SNAP="$HF_CACHE/models--${CKPT/\//--}/snapshots/$REV"
[[ -d "$SNAP" ]] || { echo "missing weights: $SNAP (prefetch on the login node)" >&2; exit 1; }

apptainer exec --nv --bind /lustre --env HF_HUB_OFFLINE=1 "$SIF" \
  python3 "$ROOT/serving/siglip_server.py" --checkpoint "$SNAP" --port "$PORT" \
  > "$LOGS/$NAME.log" 2>&1 &
PID=$!
for _ in $(seq 1 60); do
  if curl -sf --noproxy '*' "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "$NAME ready on :$PORT (pid $PID)"; exit 0
  fi
  kill -0 $PID 2>/dev/null || { echo "$NAME died, see $LOGS/$NAME.log" >&2; exit 1; }
  sleep 10
done
echo "$NAME not healthy after 10 min" >&2; exit 1
