# Serving

Models run on the B200 cluster (aarch64 compute nodes). MLLMs are served with vLLM behind
OpenAI-compatible endpoints; the pipelines reach them through `configs/models/*.yaml` and never
know where they run. One `launch_<model>.sh` per model, with a health check.

**State: M2.** Scheduler, storage paths and node layout are filled in on the cluster.

Candidate models, all with Hugging Face metadata checked on 2026-10-23 (licences still to be
read in full before integration):

| Model | Hugging Face id | Licence (HF tag) | Role |
|---|---|---|---|
| SAM 3 | `facebook/sam3` (gated) | other; TODO(verify): read the licence (the brief names it the SAM License) | Grounder, masks |
| SAM 3.1 | `facebook/sam3.1` (gated) | other; TODO(verify) | Optional; the checkpoint is a video model (`Sam3VideoModel`) |
| Rex-Omni | `IDEA-Research/Rex-Omni` | other; TODO(verify) | Second proposer (M4, if needed) |
| Qwen3.5-397B-A17B | `Qwen/Qwen3.5-397B-A17B` | apache-2.0 | Concept-card compiler |
| Qwen3.5-122B-A10B | `Qwen/Qwen3.5-122B-A10B` | apache-2.0 | Verifier |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | apache-2.0 | Verifier |
| Qwen3-VL-235B-A22B-Instruct | `Qwen/Qwen3-VL-235B-A22B-Instruct` | apache-2.0 | Direct grounding baseline |

Kimi K3 is out of scope (D5).
