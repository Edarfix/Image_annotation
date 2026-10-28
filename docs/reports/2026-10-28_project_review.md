# Project review: code, pipelines and the literature (2026-10-28)

**Audience.** The owner, and the agents who will review the project in depth. It is
self-contained: read it with [BRIEF.md](../BRIEF.md) (the specification) and
[decisions.md](../decisions.md) (D1–D17) at hand.

**Scope.** It describes the state of `main` at the end of 2026-10-28. M0 was reviewed and fixed;
M1–M3 were implemented in a first pass; task 1 was added. It then compares the pipelines with
the literature, and ends with a ranked list of what to do next. Statements about papers come
from the sources listed at the end. What could not be checked is marked TODO(verify).

---

## 1. Summary

| | State |
|---|---|
| Tasks | Task 2 (detection) and task 1 (classification with "none") both run end to end |
| Benchmarks | RF100-VL: 100 datasets converted, dev subset of 5 (D12). ELEVATER: 5 datasets, open-set split (D16) |
| Evaluation | RF100-VL protocol (COCO AP at maxDets 500) and task 1 metrics (accuracy, "none", Miller et al.'s open-set AuPR / P@95R / R@95P). Sanity-tested, **not frozen** (Q4) |
| Models served | Qwen3-VL-30B-A3B (vLLM), SigLIP 2 So400m (own server), SAM 3 (own server, **never run**: weights gated, token pending) |
| Pipelines | `p00_sam3_raw` (code only), `p01_mllm_grounding_raw` (runs), `c00_siglip2_zeroshot`, `c01_rar_siglip2_qwen3vl` (run) |
| Tests | 91 CPU tests in a few seconds, plus GPU smoke tests (`pytest -m gpu`). ruff clean |
| GPU used | About 30 min of one B200 in total, of the 4 h allowed |

**Headline numbers.** All are dev numbers, with small samples; see §5 for their limits.

- `p01` (Qwen3-VL-30B-A3B, direct grounding, zero-shot): **14.7 mAP** on the 5 RF100-VL dev
  datasets. The range is 0.03 (dental X-rays, 32 classes) to 43.3 (into-the-vale).
- Task 1, ELEVATER open-set dev (5 × 200 images): `c01` (retrieve and rank) **42.6%
  accuracy** with "none" (60.8% closed-set), against 33.8% (49.1%) for the SigLIP 2 baseline
  `c00`. c01 over-uses "none" (§5.2).

**Main risks.** In order:
1. The evaluation is not yet checked against published baseline predictions.
2. p00 (the SAM 3 floor) has never run.
3. MLLM boxes all score 1.0, which cripples AP.
4. The dev samples are small.

---

## 2. Repository map and data flow

```
configs/experiments/<x>.yaml ─┐            ┌─ configs/models/*.yaml (checkpoint, revision, licence, endpoint)
configs/datasets/<group>.yaml ─┤            │
                               ▼            ▼
scripts/run.py ──► runner/run.py ──► ModelHub (lazy wrappers: openai_mllm, sam3_http, siglip_http)
                     │   │                      │ HTTP, no proxy, disk cache (models/cache.py)
                     │   │                      ▼
                     │   │      serving/: vllm serve | sam3_server.py | siglip_server.py
                     │   │      (inside shared/vllm-sifs/vllm_cuda13.sif, started by scripts/slurm/run_experiment.sbatch)
                     │   ▼
                     │  DatasetLoader (datasets/<name>/loader.py)
                     │   spec() ─► restrict_to_setting (D4, D17) ─► Pipeline.prepare / annotate (thread pool)
                     │   images()                                    stages/ (grounding.py, classification.py)
                     │   ground_truth() ───────────────► eval/ (detection.py, classification.py)
                     ▼
results/runs/<date>_<pipeline>_<group>_<setting>_<subset>_<hash>/
   config.resolved.yaml, metrics.json (versioned) · <dataset>/{annotations.jsonl, predictions.json,
   metrics.json, timings.json, failures.jsonl} · run.log          ─► scripts/make_leaderboard.py
```

Size: about 2,500 lines in `src/`, 240 in `pipelines/`, 470 in `datasets/`, 290 in
`serving/`, 400 in `scripts/`, and 1,600 lines of tests.

**Key contracts** (see [AGENTS.md](../../AGENTS.md)):
- Boxes are absolute-pixel xyxy everywhere; only model wrappers convert formats.
- Pipelines never see ground truth.
- Each setting strips what it does not allow (D4, D17).
- `annotate` never raises: stages record typed failures.
- Prompts live in files.
- Plugins are imported by path (D8).
- Pipelines run on threads, never processes.
- A run is identified by the hash of its resolved config **and the git commit**. A new commit
  starts a new run; cached model calls make re-runs cheap (p01 replays in 2 s on a CPU).

---

## 3. Code review

### 3.1 Fixed today (each covered by a regression test)

| Issue | Where | Effect before the fix |
|---|---|---|
| Atomic writes created 0600 files | `utils/io.py` | Results unreadable by the group on shared storage |
| numpy scalars not JSON-serialisable | `core/types.py` | Any model returning `np.float32` would crash the run |
| `few_shot` passed `domain` | `core/dataset.py` | Violated D4; specs shared lists with the loader |
| `1e-3` read as a string | `core/config.py` | Silent type changes and hash changes in configs |
| `data/`, `cache/` symlinks not ignored | `.gitignore` | Cluster paths would be committed |
| mermaid browser not installed (npm 11, proxy) | `scripts/setup_mermaid.sh` | Diagrams could not be rendered |
| pycocotools `summarize()` gives mAP -1 with maxDets 500 | `eval/detection.py` | Headline mAP would read 0. AP is now read from the precision array, as detectron2's `COCOevalMaxDets` does (D15) |
| **Class in image file names** (ELEVATER `banded/banded_0002.jpg`; Roboflow names) | both `download.py` | A pipeline reading `ImageRecord.path` could see the label. Files now have opaque names |
| FlashInfer writing into the read-only image | `serving/launch_vllm.sh` | vLLM could not start on B200 |
| Two servers on one port (SigLIP on SAM 3's 8100) | `scripts/slurm/run_experiment.sbatch` | p00's smoke test talked to the wrong server; now one port range per kind, and the smoke test checks the served checkpoint |
| Grounding answers in plain text or malformed JSON counted as failures | `models/mllm_client.py` | p01 failure rate 13% on paper-parts, now 5% |

### 3.2 Open issues and risks (by severity)

1. **The evaluation is not validated against an external reference.** The sanity tests pass
   (ground truth scores 1, empty scores 0, shifted boxes score lower), but no published number
   has been reproduced. The planned check scores the RF20-VL baseline predictions
   (`predictions_rf20vl.zip`, rf100-vl.org) and compares them with the paper. Until then, do
   not freeze the evaluation (Q4).
2. **SAM 3 has never run.** `serving/sam3_server.py` follows the transformers 5.13 API checked
   in the container (`get_vision_features`, then `vision_embeds=` per phrase,
   `post_process_instance_segmentation`), but has not been executed on a GPU. Expect a fix or
   two at first launch.
3. **MLLM boxes score 1.0** (p01). This is the RF100-VL and Qwen3-VL protocol, but AP then ranks
   boxes by output order, so p01's mAP understates what the boxes are worth. P/R/F1 at IoU 0.5
   are less affected.
4. **The cache key ignores the serving stack.** It includes model, revision, prompt, image hash
   and generation parameters, but not the vLLM / transformers versions or the image
   preprocessing. Upgrading the container serves stale answers: clear `$OVANNOT_CACHE/<model>`
   when the image changes.
5. **Small dev samples.** RF100-VL dev has 472 images over 5 datasets. ELEVATER dev has 200
   images per dataset, so a binomial standard error of up to about 3.5 points on accuracy.
   Differences under about 5 points are noise.
6. **The runner aborts on a `prepare()` exception.** That is irrelevant today (no pipeline
   prepares anything expensive), but p02 concept cards will need per-class failure handling.
7. **Not yet produced by the runner:** `errors.json` (TIDE-style decomposition) and `viz/`.
   Both are required by the brief.
8. **SAM 3 server throughput.** It handles one request at a time (a lock) and does not batch
   across images. Fine for dev, slow for the full test set (14,446 images × classes).
9. **ELEVATER data provenance.** The mirror is not checked against the original host, which is
   dead. Some GPT-3 descriptions in the official knowledge files are poor (e.g. GTSRB).

### 3.3 Strengths worth keeping

- The setting is enforced in one place (`restrict_to_setting`), and the tests show that the
  dataset name never reaches a prompt.
- Every model call is cached on disk, keyed by content hashes. A parser fix replays a 472-image
  run in 2 s without a GPU.
- Each pipeline folder can be read on its own: 11-section README, rendered diagram, prompts
  in files, config with defaults, contract tests.
- Model traffic never uses the cluster proxy (`trust_env=False`): proprietary images stay
  local.

---

## 4. Benchmarks and protocols

| | RF100-VL (task 2) | ELEVATER open-set (task 1) |
|---|---|---|
| Source | HF mirror `probicheaux/rf100-vl-fsod` @ `4bcf86c4` + official domain map | HF mirror `ytaek-oh/elevater` @ `48bf249c` + official toolkit knowledge (MIT) |
| Dev | `valid` split of 5 datasets, the largest per domain (D12, approved) | 200 seeded `val` images × 5 datasets |
| Test | `test` of all 100, and of the 95 held-out (D3) | 500 seeded `test` images × 5 datasets |
| Setting | `zero_shot` (names); `few_shot` adds the 10-shot `train` boxes | `described` (names + descriptions, D17) |
| Metric | COCO mAP@[.5:.95], maxDets 500; P/R/F1 at IoU 0.5 | accuracy, closed accuracy, macro-F1, "none" P/R/F1, AuPR / P@95R / R@95P |
| Deviations | Annotator instructions not found yet (`few_shot` has no text instructions) | The open-set protocol is ours (D9 correction): a fixed seeded class split, not Miller et al.'s per-image query sets |

---

## 5. Pipelines and results

### 5.1 Detection

| Pipeline | Idea | State | Dev mAP (5 datasets) |
|---|---|---|---:|
| `p00_sam3_raw` | SAM 3, one text prompt per raw class name; floor of the SAM 3 branch | code + fake-model tests | not run |
| `p01_mllm_grounding_raw` | Qwen3-VL-30B-A3B, all classes in one ODinW-style prompt, JSON boxes, score 1.0 | runs | **14.7** |

`p01` per dataset (run `..._dev_047845dd`, parser of 2026-10-28 evening):

| Dataset | Domain | mAP | AP50 | F1@0.5 | Failures |
|---|---|---:|---:|---:|---:|
| ufba-425 | Lab Imaging | 0.0 | 0.1 | 0.4 | 0% |
| apoce-aerial-... | Aerial | 24.3 | 31.3 | 40.8 | 0% |
| 2024-frc | Industrial | 1.5 | 4.1 | 10.7 | 0% |
| into-the-vale | Flora/Fauna | 43.3 | 60.7 | 68.8 | 0% |
| paper-parts | Document | 3.9 | 7.1 | 3.7 | 5.1% |

**Reading.**
- The spread matches the RF100-VL paper's finding: grounding MLLMs do well on natural scenes and
  close to 0 on medical and document data, where class names are jargon ("tooth 11",
  "black_robot").
- paper-parts shows high recall (42%) and very low precision (2%): the model over-detects
  layout elements. A verifier or a calibrated score would help most there.
- Dev and test numbers are not comparable: the published numbers are test numbers over 100
  datasets.

### 5.2 Classification (task 1)

| Pipeline | Idea | Models |
|---|---|---|
| `c00_siglip2_zeroshot` | Contrastive zero-shot with description ensembling (CLIP / CuPL / ELEVATER style); "none" when the top sigmoid probability is below a threshold | SigLIP 2 So400m/14@384 |
| `c01_rar_siglip2_qwen3vl` | **Retrieve and rank** (RAR, Liu et al. 2024): SigLIP 2 shortlists 5 classes; Qwen3-VL sees them with descriptions plus "None of the above" and answers one letter. Option probabilities come from first-token logprobs; if the ranker fails, the retriever decides | SigLIP 2 + Qwen3-VL-30B-A3B |

Dev results (ELEVATER open-set, `described`, 200 images per dataset, runs `..._c041d26e` and
`..._b06137a9`, in %):

| Dataset | c00 acc. | c01 acc. | c00 closed acc. | c01 closed acc. | c00 "none" F1 | c01 "none" F1 | c00 AuPR | c01 AuPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eurosat | 35.0 | 35.5 | 54.4 | 61.3 | 38.8 | 37.6 | 90.7 | 86.4 |
| dtd | 49.5 | 58.0 | 66.7 | 74.7 | 43.1 | 23.5 | 85.6 | 79.1 |
| gtsrb | 51.0 | 41.5 | 68.1 | 74.3 | 12.9 | 46.1 | 87.4 | 84.3 |
| fer2013 | 16.0 | 50.0 | 36.8 | 66.1 | 8.2 | 38.0 | 67.4 | 96.4 |
| fgvc_aircraft | 17.5 | 28.0 | 19.6 | 27.8 | 18.2 | 38.3 | 67.5 | 81.5 |
| **mean** | 33.8 | **42.6** | 49.1 | **60.8** | 24.2 | **36.7** | 79.7 | **85.5** |

**Reading.**
- c01 beats c00 on the mean of every metric. Its closed-set accuracy is higher on all five
  datasets (+11.7 points on average): the MLLM ranks the shortlist better than SigLIP 2.
- **c01 says "none" far too often.** The share of "none" answers is 81% on eurosat, 80% on
  gtsrb, 73% on fgvc_aircraft and 48% on fer2013, against about 20% of open-set images. So on
  eurosat and gtsrb its open accuracy (with "none") barely beats, or loses to, c00, despite a
  better closed accuracy. The "None of the above" option attracts the MLLM when the
  descriptions do not match the image well. This is the first thing to fix (§7, classification
  step 1). The probabilities are stored (`class_scores`, `extras.p_none`), so a "none" rule
  tuned on dev needs no GPU.
- c00's `none_threshold` (0.01) is a placeholder: its "none" F1 is low too.
- Costs: c01 takes 0.3–1.1 s per image under 32 concurrent requests (one SigLIP embedding and
  one 1-token MLLM call); c00 takes 0.8–3.3 s, dominated by the first text-embedding calls.

**Is c01 state of the art?**
- It follows the strongest *training-free* recipe found in the literature for many-class
  recognition with MLLMs: a contrastive shortlist, then MLLM ranking (RAR). It adds an explicit
  reject option and logprob scores, which the open-set metrics need.
- No published number exists for our open-set protocol, so SOTA cannot be claimed
  numerically.
- Known gaps with RAR:
  - no fine-tuned ranker;
  - no in-context ranking examples;
  - no position-bias mitigation.

---

## 6. Against the literature

### 6.1 Where detection stands (RF100-VL)

| System | Setting | RF100-VL mAP | Source |
|---|---|---:|---|
| Grounding DINO | zero-shot | 15.7 | RF100-VL paper, Tab. 2 |
| SAM 3 | zero-shot | 15.2 (14.3 in an earlier arXiv version) | SAM 3, ICLR 2026, Tab. 2 |
| Gemini 2.5 Pro | zero-shot | 11.6 in the paper; 13.3 in a later Roboflow post | RF100-VL Tab. 2; Roboflow blog |
| Qwen2.5-VL-72B | zero-shot | 5.6 | RF100-VL Tab. 2 |
| SAM 3, fine-tuned | 10-shot | 36.5 | SAM 3, ICLR 2026 |
| CVPR 2026 RF20-VL challenge winner (Superb AI) | few-shot, fine-tuned, 20 datasets | 53.9 (GDino baseline 33.3) | Superb AI technical write-up |

**Implications for this project.**

1. **The zero-shot ceiling is low (about 15–16 mAP) and SAM 3 sits at it.** `p00` should
   land near 15 on the full test set; our dev subset is easier (p01 already gets 14.7 on it).
   The gap to few-shot fine-tuned systems (about 37–54) is the room the training-free
   pipelines (p02–p04) are meant to close partly.
2. **The 2026 challenge winner uses this repository's p03 idea.** Its pseudo-labelling stage
   gets proposals from its detector and **SAM 3**, then **Qwen3-VL-32B classifies or rejects
   each box**. It also runs a training-free per-domain search for class-name aliases (like
   p02's concept-card phrases), test-time augmentation (multi-scale, flip, tiling) and
   multi-source fusion with WBF. The write-up gives no per-stage ablation and no training-free
   number, so the gain of each stage is unknown. TODO(verify): the full write-up or paper, for
   ablations.
3. **Newer open-vocabulary detectors are not clearly better out of distribution.**
   - LLMDet reports 52.1 ODinW13 against MM-Grounding-DINO's 52.5 (Swin-T).
   - VL-SAM-v3 (May 2026) improves SAM 3 on LVIS with memory-retrieved visual priors.
   - No RF100-VL number was found for Rex-Omni, LLMDet, DINO-X or VL-SAM-v3. Their value on our
     OOD data is untested.
4. **MLLM grounding improves.** Qwen3.5-family models report about 50 on ODinW13 in the
   Qwen3.5-Omni report, against 48.6 for Qwen3-VL-235B. TODO(verify): which number belongs to
   which model, in that report's table. ODinW numbers do not transfer to RF100-VL.

### 6.2 Where classification stands

- **Contrastive encoders.** SigLIP 2 (Feb 2025) is a strong open baseline. Meta's Perception
  Encoder (PE-Core, NeurIPS 2025) reports better zero-shot classification than SigLIP 2 at
  every size, PE-Core-G being the strongest. TODO(verify): exact numbers in its Tab. 5. PE-Core
  is the natural next retriever for c00/c01.
- **MLLMs as classifiers.** "On Large Multimodal Models as Open-World Image Classifiers"
  (arXiv 2503.21851) evaluates 13 models on 10 benchmarks. It finds granularity and
  fine-grained classes hard, and finds that tailored prompting and reasoning help. RAR (arXiv
  2403.13805) shows that retrieval plus MLLM ranking beats either alone on fine-grained,
  few-shot and zero-shot recognition. TLAC and AutoSEP are other two-stage or unlabeled-data
  variants.
- **Open-set.** Miller et al. (ECCV 2024) show that contrastive VLM classifiers make confident
  open-set errors (ImageNet AuPR about 71–81, R@95P below 11%). This is the motivation for
  c01's explicit "none" option.

---

## 7. Recommended next steps (ranked by expected value per GPU-hour)

**Detection**

1. **Run p00 as soon as the HF token is in place.** It gives the floor, checks the SAM 3
   server, and checks the evaluation against the SAM 3 paper's 15.2 on the full test set.
   That check runs once, at M5, not for tuning.
2. **Reproduce a published number from `predictions_rf20vl.zip`**, then freeze the evaluation
   (Q4).
3. **p03 verifier.** SAM 3 proposals, then a Qwen3-VL yes/no on a context crop plus the full
   image with the box drawn. The score combines SAM 3's score and P(yes) from logprobs. This
   is the strongest literature signal (the challenge winner) and it is already in the brief.
   It also fixes p01's constant scores if applied to p01's boxes.
4. **Logprob confidence for p01 boxes.** The probability of the label tokens, so that AP ranks
   the boxes.
5. **p02 concept cards with alias search on dev only.** Global settings only, per D3. The
   challenge winner's greedy alias search per domain is the training-free version.
6. **Tiling or zoom for small objects** (aerial, PCB, cells): the winner's TTA, and
   DeepEyes-style zoom.
7. **Extra proposers and fusion.** MM-Grounding-DINO base and large and Grounding DINO base are
   already in the group's model cache (`shared/models/hf_local`, `shared/models/huggingface/hub`);
   fuse them with WBF. Rex-Omni only if recall plateaus (brief).
8. **Larger or newer MLLMs.** Qwen3-VL-235B-A22B (the brief's model), then Qwen3.5.

**Classification**

1. **Fix the "none" over-rejection, on dev.** For example, choose "none" only when `p_none`
   beats the best class by a margin, or combine `p_none` with the retriever's top
   probability. Tune `top_k` after measuring the shortlist recall (`extras.candidates`).
   Everything needed is stored in the runs: no GPU.
2. **Average over option permutations** to cancel the ranker's letter-position bias.
3. **PE-Core as retriever** instead of SigLIP 2.
4. **Wire `few_shot`.** ELEVATER train images as in-context examples, as in RAR's ICL variant.
5. **Larger ranker** (Qwen3-VL-235B or Qwen3.5) for fine-grained sets (FGVC-Aircraft).

---

## 8. Decisions pending with the owner

| Id | Question | Default taken |
|---|---|---|
| Q4 | Freeze the evaluation | Not frozen; freeze after step 7.D2 |
| Q5 / D16 | Task 1 benchmark: 5 ELEVATER datasets, 20% of classes withheld, seed 0 | Implemented as default |
| D17 | `described` setting for task 1 | Implemented |
| Q6 | SAM 3 access | Waiting for the token |

---

## Sources

- RF100-VL paper (NeurIPS 2025 D&B): https://arxiv.org/abs/2505.20612
- SAM 3 (ICLR 2026): https://arxiv.org/abs/2511.16719 and the
  [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/e0982cbc81401df3430ee1ff780dc7a2-Paper-Conference.pdf)
- SAM 3.1 release notes: https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md
- Ultralytics SAM 3 page (earlier numbers): https://docs.ultralytics.com/models/sam-3
- Roboflow, GPT-5 vision tests (Gemini 2.5 Pro 13.3 zero-shot): https://roboflow.com/blog/gpt-5-vision-multimodal-evaluation
- CVPR 2026 challenge: https://blog.roboflow.com/roboflow20-vl-challenge-at-cvpr-2026/ ;
  winner's write-up: https://superb-ai.com/en/resources/blog/cvpr-challenge-win-technical-walkthrough-en
- Qwen3-VL report: https://arxiv.org/abs/2511.21631 ; cookbook (box format):
  https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb
- Qwen3.5-Omni report (ODinW13): https://arxiv.org/html/2604.15804v1
- LLMDet: https://arxiv.org/abs/2501.18954 · VL-SAM-v3: https://arxiv.org/abs/2605.03456 ·
  Rex-Omni: https://arxiv.org/abs/2510.12798
- RAR: https://arxiv.org/abs/2403.13805 (code: https://github.com/Liuziyu77/RAR)
- Miller et al., open-set recognition with VLMs (ECCV 2024): https://arxiv.org/abs/2403.16528
- LMMs as open-world classifiers: https://huggingface.co/papers/2503.21851
- AutoSEP: https://arxiv.org/abs/2506.03195
- Perception Encoder: https://arxiv.org/abs/2504.13181 · SigLIP 2: https://arxiv.org/abs/2502.14786
- ELEVATER toolkit: https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC
- detectron2 `COCOevalMaxDets`: https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py
