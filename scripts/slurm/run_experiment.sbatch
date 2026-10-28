#!/bin/bash
# Serve the models an experiment needs on this node, then run it (or the GPU smoke tests).
#
#   SERVE="vllm:qwen3_vl_30b" sbatch scripts/slurm/run_experiment.sbatch \
#       --exp configs/experiments/p01_mllm_grounding_raw_rf100vl_dev_zero_shot.yaml
#   SERVE="vllm:qwen3_vl_30b sam3" sbatch scripts/slurm/run_experiment.sbatch --smoke
#
# SERVE lists the servers, one GPU each: "vllm:<model name>" or "sam3"; "siglip:<model name>"
# shares the GPU of the server before it. A leading --smoke
# runs `pytest -m gpu` first; the remaining arguments, if any, go to scripts/run.py. Paths come from the environment:
# OVANNOT_DATA, HF_HUB_CACHE, and the uv settings of the cluster (see AGENTS.md).
#SBATCH --job-name=ovannot-run
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.log
set -euo pipefail
module use /lustre/work/pdl16831/shared/modulefiles && module load ohmyjob
cd "${SLURM_SUBMIT_DIR:-.}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}" OVANNOT_LOGS="$PWD/logs/$SLURM_JOB_ID"
mkdir -p "$OVANNOT_LOGS"

# One port range per server kind, so that each matches the default endpoint of its model
# config (vLLM 8000+, SAM 3 8100, SigLIP 8200+) and two servers never share a port.
GPU=0; VLLM_PORT=8000; SIGLIP_PORT=8200
for server in ${SERVE:-}; do
  case "$server" in
    vllm:*) name=${server#vllm:}
            var="OVANNOT_ENDPOINT_$(echo "$name" | tr '[:lower:]' '[:upper:]')"
            CUDA_VISIBLE_DEVICES=$GPU serving/launch_vllm.sh "$name" $VLLM_PORT
            export "$var=http://127.0.0.1:$VLLM_PORT/v1"
            GPU=$((GPU + 1)); VLLM_PORT=$((VLLM_PORT + 1)) ;;
    sam3)   CUDA_VISIBLE_DEVICES=$GPU serving/launch_sam3.sh 8100
            export OVANNOT_ENDPOINT_SAM3="http://127.0.0.1:8100"
            GPU=$((GPU + 1)) ;;
    siglip:*) name=${server#siglip:}  # small: shares the GPU of the previous server
            var="OVANNOT_ENDPOINT_$(echo "$name" | tr '[:lower:]' '[:upper:]')"
            CUDA_VISIBLE_DEVICES=$(( GPU > 0 ? GPU - 1 : 0 )) serving/launch_siglip.sh "$name" $SIGLIP_PORT
            export "$var=http://127.0.0.1:$SIGLIP_PORT"
            SIGLIP_PORT=$((SIGLIP_PORT + 1)) ;;
    *)      echo "unknown server $server" >&2; exit 1 ;;
  esac
done

uv sync --locked -q
if [[ "${1:-}" == "--smoke" ]]; then
  shift
  srun ohmyjob uv run pytest -m gpu -v -s -p no:cacheprovider
fi
if [[ $# -gt 0 ]]; then
  srun ohmyjob uv run python scripts/run.py --workers 32 "$@"
fi
