# Serving

Models run on the B200 cluster (aarch64 compute nodes). MLLMs are served with vLLM behind
OpenAI-compatible endpoints; the pipelines reach them through `configs/models/*.yaml` and never
know where they run. One `launch_<model>.sh` per model, with a health check.

**State (2026-10-28, D13).** Servers run inside the cluster's vLLM Apptainer image
(`/lustre/work/pdl16831/shared/vllm-sifs/vllm_cuda13.sif`: vLLM 0.23.1, transformers 5.13,
torch 2.11 cu130, aarch64), one GPU each, on the node of the job that uses them:

| Script | Serves | Endpoint (config) |
|---|---|---|
| `launch_vllm.sh <model> [port]` | any `openai_mllm` model of `configs/models/` | `http://127.0.0.1:8000/v1` |
| `launch_sam3.sh [port]` | SAM 3 through `sam3_server.py` (`/health`, `/detect`) | `http://127.0.0.1:8100` |

Both wait for the health check and fail loudly. Weights are read offline from `$HF_HUB_CACHE`
(prefetch them on the login node). `scripts/slurm/run_experiment.sbatch` starts the servers
listed in `SERVE`, exports their endpoints (`endpoint_env` of each model config), then runs the
smoke tests and / or the experiment:

```bash
SERVE="vllm:qwen3_vl_30b" sbatch scripts/slurm/run_experiment.sbatch --smoke \
    --exp configs/experiments/p01_mllm_grounding_raw_rf100vl_dev_zero_shot.yaml
```

Model traffic never goes through the cluster proxy (`trust_env=False`).

Candidate models, all with Hugging Face metadata checked on 2026-10-23 (licences still to be
read in full before integration):

| Model | Hugging Face id | Licence (HF tag) | Role |
|---|---|---|---|
| SAM 3 | `facebook/sam3` (gated, manual approval) | other; TODO(verify): read the licence (the brief names it the SAM License) | Grounder, masks (`configs/models/sam3.yaml`) |
| SAM 3.1 | `facebook/sam3.1` (gated) | other; TODO(verify) | Optional; the checkpoint is a video model (`Sam3VideoModel`) |
| Rex-Omni | `IDEA-Research/Rex-Omni` | other; TODO(verify) | Second proposer (M4, if needed) |
| Qwen3.5-397B-A17B | `Qwen/Qwen3.5-397B-A17B` | apache-2.0 | Concept-card compiler |
| Qwen3.5-122B-A10B | `Qwen/Qwen3.5-122B-A10B` | apache-2.0 | Verifier |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | apache-2.0 | Verifier |
| Qwen3-VL-235B-A22B-Instruct | `Qwen/Qwen3-VL-235B-A22B-Instruct` | apache-2.0 | Direct grounding baseline |
| Qwen3-VL-30B-A3B-Instruct | `Qwen/Qwen3-VL-30B-A3B-Instruct` | apache-2.0 | p01 for now (D14, `configs/models/qwen3_vl_30b.yaml`) |

Kimi K3 is out of scope (D5).
