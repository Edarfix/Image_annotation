#!/bin/bash
# Start the SAM 3 server (serving/sam3_server.py) in the vLLM container, in the background,
# and wait until it is healthy:  serving/launch_sam3.sh [port]
# The endpoint is http://127.0.0.1:<port>, as configs/models/sam3.yaml expects.
set -euo pipefail
PORT=${1:-8100}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF=${VLLM_SIF:-/lustre/work/pdl16831/shared/vllm-sifs/vllm_cuda13.sif}
HF_CACHE=${HF_HUB_CACHE:?set HF_HUB_CACHE to the Hugging Face cache}
LOGS=${OVANNOT_LOGS:-$ROOT/logs}; mkdir -p "$LOGS"
REV=$(grep -E '^revision:' "$ROOT/configs/models/sam3.yaml" | sed -E 's/^revision: *//')
SNAP="$HF_CACHE/models--facebook--sam3/snapshots/$REV"
[[ -d "$SNAP" ]] || { echo "missing weights: $SNAP (gated: needs an HF token)" >&2; exit 1; }

apptainer exec --nv --bind /lustre --env HF_HUB_OFFLINE=1 "$SIF" \
  python3 "$ROOT/serving/sam3_server.py" --checkpoint "$SNAP" --port "$PORT" \
  > "$LOGS/sam3.log" 2>&1 &
PID=$!
for _ in $(seq 1 60); do
  if curl -sf --noproxy '*' "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "sam3 ready on :$PORT (pid $PID)"; exit 0
  fi
  kill -0 $PID 2>/dev/null || { echo "sam3 died, see $LOGS/sam3.log" >&2; exit 1; }
  sleep 10
done
echo "sam3 not healthy after 10 min" >&2; exit 1
