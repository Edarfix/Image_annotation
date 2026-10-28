#!/bin/bash
# Start a vLLM OpenAI-compatible server for one model, in the background, and wait until it is
# healthy. Meant to be sourced or called from a SLURM job on a GPU node:
#
#   serving/launch_vllm.sh qwen3_vl_30b [port] [extra vllm args...]
#
# Exports nothing (it runs in a subshell); the endpoint is http://127.0.0.1:<port>/v1, which
# is what configs/models/<name>.yaml expects (or set its endpoint_env). The weights are read
# from the local Hugging Face cache, offline. Logs: $OVANNOT_LOGS (default ./logs)/vllm_<name>.log
set -euo pipefail
NAME=${1:?model name}; PORT=${2:-8000}; shift $(( $# > 1 ? 2 : 1 ))
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF=${VLLM_SIF:-/lustre/work/pdl16831/shared/vllm-sifs/vllm_cuda13.sif}
HF_CACHE=${HF_HUB_CACHE:?set HF_HUB_CACHE to the Hugging Face cache}
LOGS=${OVANNOT_LOGS:-$ROOT/logs}; mkdir -p "$LOGS"
# FlashInfer downloads kernels (cubins) on first use into its package folder, read-only in the
# image, and JIT-compiles into $HOME (3 GB quota): give it a writable cache instead.
FI_CACHE=${FLASHINFER_CACHE:-$ROOT/.cache/flashinfer}; mkdir -p "$FI_CACHE/cubins"
FI_PKG=/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins

cfg() { grep -E "^$1:" "$ROOT/configs/models/$NAME.yaml" | head -1 | sed -E "s/^$1: *//; s/ *#.*//"; }
CKPT=$(cfg checkpoint); REV=$(cfg revision); SERVED=$(cfg served_model_name)
SNAP="$HF_CACHE/models--${CKPT/\//--}/snapshots/$REV"
[[ -d "$SNAP" ]] || { echo "missing weights: $SNAP (prefetch on the login node)" >&2; exit 1; }

apptainer exec --nv --bind /lustre --bind "$FI_CACHE/cubins:$FI_PKG" \
  --env HF_HUB_OFFLINE=1 --env FLASHINFER_WORKSPACE_BASE="$FI_CACHE" "$SIF" \
  vllm serve "$SNAP" --served-model-name "$SERVED" --host 127.0.0.1 --port "$PORT" \
  --max-model-len 32768 --limit-mm-per-prompt '{"image": 4}' --gpu-memory-utilization 0.85 \
  "$@" > "$LOGS/vllm_$NAME.log" 2>&1 &
PID=$!
for _ in $(seq 1 180); do  # up to 30 min (first start compiles kernels)
  if curl -sf --noproxy '*' "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "vllm $NAME ready on :$PORT (pid $PID)"; exit 0
  fi
  kill -0 $PID 2>/dev/null || { echo "vllm $NAME died, see $LOGS/vllm_$NAME.log" >&2; exit 1; }
  sleep 10
done
echo "vllm $NAME not healthy after 30 min" >&2; exit 1
