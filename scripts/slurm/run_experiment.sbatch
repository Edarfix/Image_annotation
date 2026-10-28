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

GPU=0; PORT=8000
for server in ${SERVE:-}; do
  case "$server" in
    vllm:*) name=${server#vllm:}
            var="OVANNOT_ENDPOINT_$(echo "$name" | tr '[:lower:]' '[:upper:]')"
            CUDA_VISIBLE_DEVICES=$GPU serving/launch_vllm.sh "$name" $PORT
            export "$var=http://127.0.0.1:$PORT/v1" ;;
    sam3)   CUDA_VISIBLE_DEVICES=$GPU serving/launch_sam3.sh $PORT
            export OVANNOT_ENDPOINT_SAM3="http://127.0.0.1:$PORT" ;;
    siglip:*) name=${server#siglip:}  # small: shares the GPU of the previous server
            (( GPU > 0 )) && GPU=$((GPU - 1))
            var="OVANNOT_ENDPOINT_$(echo "$name" | tr '[:lower:]' '[:upper:]')"
            CUDA_VISIBLE_DEVICES=$GPU serving/launch_siglip.sh "$name" $PORT
            export "$var=http://127.0.0.1:$PORT" ;;
    *)      echo "unknown server $server" >&2; exit 1 ;;
  esac
  GPU=$((GPU + 1)); PORT=$((PORT + 100))
done

uv sync --locked -q
if [[ "${1:-}" == "--smoke" ]]; then
  shift
  srun ohmyjob uv run pytest -m gpu -v -s -p no:cacheprovider
fi
if [[ $# -gt 0 ]]; then
  srun ohmyjob uv run python scripts/run.py --workers 32 "$@"
fi
