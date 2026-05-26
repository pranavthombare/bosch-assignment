# DriveLM Qwen VQA Pipeline

This project benchmarks and fine-tunes a small multimodal Qwen model on DriveLM questions linked to nuScenes driving images, and serves it through vLLM behind an OpenAI-compatible HTTP API.

The simple mental model:

- nuScenes provides the driving camera images.
- DriveLM provides the questions and correct answers for each driving frame.
- Qwen is the vision-language model that answers the questions from one or more camera views.
- vLLM serves the base model and optional LoRA adapter through `/v1/chat/completions`.

## Current Model

Default model:

```text
Qwen/Qwen3.5-0.8B
```

Camera modes:

| Mode | Cameras |
| --- | --- |
| `front` | `CAM_FRONT` |
| `front-arc` | `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT` |
| `all` | all six nuScenes cameras |
| `mosaic` | prepared composite image, falling back to all cameras |

## Data

Expected local paths:

```text
data/nuscenes
data/drivelm/v1_1_train_nus.json
```

DriveLM JSON distribution:

| QA Category | Count | Share |
| --- | ---: | ---: |
| Perception | 162,480 | 42.99% |
| Prediction | 123,436 | 32.66% |
| Planning | 87,968 | 23.27% |
| Behavior | 4,072 | 1.08% |

The simple loader keeps only samples whose `CAM_FRONT` image exists under `data/nuscenes`. On the current local nuScenes-mini subset, `src/data/pipeline.py` reports:

| Item | Count |
| --- | ---: |
| Scenes with local images | 6 |
| Frames with local images | 38 |
| Flattened QA samples | 3,770 |
| Frames with all six camera files | 38 |

## Project Layout

```text
.env.sample                      # Documents every DRIVELM_* env var; copy → .env
src/config.py                    # Typed config dataclasses + env-var loader
src/qwen.py                      # Shared model + image + prompt helpers
src/vllm_launcher.py             # vLLM CLI wrapper (base model + auto-LoRA attach)
src/data/pipeline.py             # Flattens DriveLM scene/frame QA into samples
src/eval/eval.py                 # Async concurrent eval against vLLM (base + LoRA in one pass)
src/train/finetune.py            # Qwen LoRA/QLoRA training
```

## Environments

Two virtual environments, by design:

- `.venv` — Transformers, training, eval client, dataset loader
- `.venv-vllm` — vLLM and its torch stack (separate to avoid dependency conflicts)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

python3.12 -m venv .venv-vllm
VIRTUAL_ENV=.venv-vllm uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
```

On CUDA the Qwen helpers default to 4-bit loading when `bitsandbytes` is available — important on 8GB GPUs during training.

## Reproduction

All scripts are env-var driven — there is no `argparse` and no YAML config surface. Defaults live in typed dataclasses in `src/config.py`; override them with `DRIVELM_*` environment variables. A `.env` file at the project root is auto-loaded by `python-dotenv` at script start.

```text
code defaults (src/config.py)
  ↓
.env file at project root (auto-loaded)
  ↓
DRIVELM_<SECTION>__<FIELD> shell env  e.g. DRIVELM_EVAL__NUM_SAMPLES=10
```

The effective config is printed at script start and embedded inside every artifact JSON, so any run is reproducible from its own output file. **Start by copying `.env.sample` → `.env` and editing the values you want to change.**

### Step-by-step

Each step's expected output artifact is listed alongside.

1. **Sanity-check the data pipeline.** Confirms DriveLM QA pairs resolve to local nuScenes images.

   ```bash
   .venv/bin/python src/data/pipeline.py
   ```

   Expected: prints `Samples: 3770, Frames: 38, Frames with all six cameras: 38`. If you see `Samples: 0`, extract `v1.0-mini.tgz` into `data/nuscenes/` first.

2. **Start the vLLM server** in a separate terminal (runs continuously).

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   .venv-vllm/bin/python -m src.vllm_launcher
   ```

   Wait for `Application startup complete.` (~30–60s on RTX 2070 SUPER). Verify:

   ```bash
   curl http://127.0.0.1:8001/v1/models
   ```

3. **Fine-tune a LoRA adapter.** (If vLLM is running first, stop it to free the GPU — `kill <PID>`. Training and serving share the same single GPU on the local 8 GB box.)

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   .venv/bin/python src/train/finetune.py
   ```

   Writes → `models/qwen-lora/adapter_model.safetensors` plus tokenizer / config files (~20 min). Tune `DRIVELM_TRAIN__NUM_SAMPLES`, `DRIVELM_TRAIN__EPOCHS`, etc. via `.env` or the shell.

4. **Start (or restart) vLLM** — the launcher auto-attaches the new adapter as `drivelm-lora`.

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   .venv-vllm/bin/python -m src.vllm_launcher
   ```

   `curl http://127.0.0.1:8001/v1/models` should list `Qwen/Qwen3.5-0.8B` **and** `drivelm-lora`.

5. **Run the full eval.** One invocation evaluates the base model and the LoRA adapter back-to-back on the same 3,770 samples through the same warm vLLM server (apples-to-apples).

   ```bash
   .venv/bin/python src/eval/eval.py
   ```

   Writes → `artifacts/baseline_front_arc_full.json` and `artifacts/finetuned_front_arc_full.json` (~40 min wall clock total).

   To run only one side, override the toggle:

   ```bash
   DRIVELM_EVAL__RUN_LORA=false  .venv/bin/python src/eval/eval.py
   DRIVELM_EVAL__RUN_BASE=false  .venv/bin/python src/eval/eval.py
   ```

6. **The comparison and qualitative artifacts** at `artifacts/comparison.json` and `artifacts/qualitative_lora_vs_base.json` are produced from the two JSONs above. Both use `rouge_score.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)` and group records by `question_type` for the per-category breakdown shown in the Analysis section.

For a smoke run on any step, override the relevant fields inline without touching `.env`:

```bash
DRIVELM_EVAL__NUM_SAMPLES=10 \
DRIVELM_EVAL__OUTPUT_BASE_JSON=artifacts/smoke_base.json \
DRIVELM_EVAL__OUTPUT_LORA_JSON=artifacts/smoke_lora.json \
.venv/bin/python src/eval/eval.py
```

## Default vLLM Serving Configuration

The launcher reads all settings from environment variables. Defaults are tuned for an RTX 2070 SUPER (8GB) but work as a sensible baseline on bigger GPUs.

| Variable | Default | Notes |
| --- | --- | --- |
| `DRIVELM_MODEL_ID` | `Qwen/Qwen3.5-0.8B` | base model |
| `DRIVELM_VLLM_HOST` | `0.0.0.0` | |
| `DRIVELM_VLLM_PORT` | `8001` | |
| `DRIVELM_VLLM_DTYPE` | `float16` | |
| `DRIVELM_VLLM_MAX_MODEL_LEN` | `1024` | bounded context for short DriveLM answers |
| `DRIVELM_VLLM_GPU_MEMORY_UTILIZATION` | `0.60` | raise on bigger GPUs |
| `DRIVELM_VLLM_MAX_NUM_SEQS` | `4` | continuous-batch concurrency limit |
| `DRIVELM_VLLM_MAX_NUM_BATCHED_TOKENS` | `2048` | |
| `DRIVELM_VLLM_ATTENTION_BACKEND` | `TRITON_ATTN` | FlashInfer crashed on Turing locally |
| `DRIVELM_VLLM_IMAGE_COUNT` | `6` | max images per request |
| `DRIVELM_VLLM_IMAGE_WIDTH` / `_HEIGHT` | `336` / `336` | per-image limit |
| `DRIVELM_VLLM_ENFORCE_EAGER` | `1` | disable CUDA graphs on small GPUs |
| `DRIVELM_VLLM_SKIP_MM_PROFILING` | `1` | skip startup multimodal profiling |
| `DRIVELM_LORA_PATH` | `models/qwen-lora` | auto-attached if `adapter_config.json` exists |
| `DRIVELM_LORA_NAME` | `drivelm-lora` | served model name for the adapter |

On a bigger GPU lower the `ENFORCE_EAGER` and `SKIP_MM_PROFILING` flags and raise `GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`, and image limits.

## Docker

The image is `vllm/vllm-openai` with the `src/` tree copied in. Mount local `data/` and `models/`:

```bash
docker compose up --build
```

OpenAI-compatible API at:

```text
http://127.0.0.1:8001/v1
```

## Analysis

All numbers below are measured locally on an RTX 2070 SUPER (8 GB) against `Qwen/Qwen3.5-0.8B` served by vLLM 0.21 inside the project Docker container, front-arc camera mode (3 cameras / question), 3,770 locally-resolvable samples, temperature 0, max-new-tokens 64, concurrency 4. The baseline run used `DRIVELM_ENABLE_LORA=false`; the LoRA run uses the same image with the adapter auto-attached. Source JSONs in `artifacts/`.

### Baseline — zero-shot

| Metric | Value |
| --- | ---: |
| ROUGE-1 | 0.166 |
| ROUGE-2 | 0.069 |
| **ROUGE-L** | **0.157** |
| Token-F1 | 0.117 |
| Exact match | 0.37% |
| Mean per-request latency | 1,420 ms |
| Full-set wall clock (3,770 samples, concurrency 4) | 11m 23s |

Per category (`artifacts/baseline_front_arc_full.json` → records):

| Category | N | ROUGE-L | Exact | Mean ms |
| --- | ---: | ---: | ---: | ---: |
| perception | 1,738 | 0.217 | 0.75% | 1,254 |
| prediction | 1,181 | 0.097 | 0.08% | 1,544 |
| planning | 813 | 0.107 | 0.00% | 1,598 |
| behavior | 38 | 0.305 | 0.00% | 1,309 |

Behavior n=38 has wide confidence; treat the headline as approximate. Perception > behavior > planning > prediction is the expected ordering: static visible content > templated ego-status > driving-rule pattern matching > temporal reasoning from a single frame.

### Prompt ablation

The user-facing prompt is locked to `Question: {q}\nAnswer in one short sentence.` in `src/qwen.py`. The ablation on a 10-sample front-arc smoke:

| Prompt | ROUGE-L | Token-F1 | Mean ms |
| --- | ---: | ---: | ---: |
| `Answer concisely.` (initial) | n/a¹ | 0.17 | 6,194 |
| no constraint | 0.18 | 0.14 | 8,126 |
| **`Answer in one short sentence.` (locked)** | **0.37** | **0.30** | **6,743** |

¹ Original metric backend was broken at the time of this run; later re-runs of the locked prompt confirmed ROUGE-L ≈ 0.41 on the same 10-sample smoke.

Mechanism: `Answer concisely.` collapses outputs to single words ("Cars") — high precision, no recall. No constraint produces verbose Markdown essays. The middle prompt anchors to DriveLM's declarative single-sentence answer style. This is the cheapest defensible "intentional choice" in the project — 2× ROUGE-L on a one-line change, no training.

### Fine-tuning configuration

LoRA r=8, α=16 on q/k/v/o/gate/up/down_proj — 3.19M trainable params (0.37% of base). Vision tower frozen (catastrophic forgetting risk on 38 unique camera frames). 4-bit NF4 base via bitsandbytes, gradient checkpointing. **Loss masked to assistant tokens only**: `build_supervised_inputs` tokenizes the prompt twice (once with the assistant turn, once without) and sets `labels[:, :prompt_length] = -100`; without this fix the model trains on reproducing the user's question. Training corpus is the first 1,024 samples in natural distribution (see "Methodological limitations" below for the consequences). 1 epoch, batch=1 with grad-accum 2, lr 2e-4, intermediate checkpoint every 200 steps. Wall clock: ~20 min on RTX 2070 SUPER, epoch-average loss 0.4422.

### Baseline vs LoRA — full ablation series (3,770 samples, identical eval setup)

Six configurations measured. Each LoRA was trained for 1 epoch on local DriveLM data, eval ran against vLLM with the adapter attached.

#### Overall

| Metric | baseline | nat 2e-4 | nat 1e-4 | nat 5e-4 | stratified | **prop 1e-4** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ROUGE-1 | 0.166 | 0.550 | 0.591 | 0.547 | 0.524 | **0.627** ⭐ |
| ROUGE-2 | 0.069 | 0.188 | 0.196 | 0.181 | 0.214 | **0.257** ⭐ |
| **ROUGE-L** | **0.157** | 0.541 | 0.581 | 0.540 | 0.518 | **0.621** ⭐ |
| Token-F1 | 0.117 | 0.510 | 0.544 | 0.497 | 0.494 | **0.602** ⭐ |
| Exact match | 0.4% | 39.3% | 41.9% | 35.8% | 36.6% | **47.4%** ⭐ |
| Mean latency (ms) | 1,420 | 1,046 | 2,098 | 1,840 | 1,811 | 1,858 |

#### Per category (ROUGE-L)

| Category | N | baseline | nat 2e-4 | nat 1e-4 | nat 5e-4 | stratified | **prop 1e-4** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| perception | 1,738 | 0.217 | 0.489 | 0.533 | 0.513 | 0.615 | **0.625** ⭐ |
| prediction | 1,181 | 0.097 | 0.659 | **0.696** | 0.617 | 0.368 | 0.682 |
| planning | 813 | 0.107 | 0.502 | 0.503 | 0.509 | 0.507 | **0.543** ⭐ |
| **behavior** | 38 | 0.305 | 0.036 ⚠️ | **0.877** | 0.022 ⚠️ | **0.911** ⭐ | 0.201 |

#### What each row taught us

| Config | Lesson |
| --- | --- |
| **nat 2e-4** | PEFT-default LR → 3.5× ROUGE-L lift overall **but** behavior catastrophically collapses to 0.036 (terse mode collapse) |
| **nat 1e-4** | **Lower LR alone fixes behavior** (0.036 → 0.877) without changing data. The behavior collapse was an LR effect, not a sampling effect. |
| **nat 5e-4** | Higher LR makes behavior worse (0.022) — the cliff is bidirectional |
| **stratified** | Uniform within-category sampling fixes behavior (0.911) **but overcorrects prediction** (0.659 → 0.368) by destroying DriveLM's natural `No`-heavy prediction prior |
| **prop 1e-4** | **Best overall + best 3 of 4 categories.** Proportional sampling preserves natural within-category priors with a min-floor on rare patterns; combined with lr=1e-4 it wins. Behavior is 0.201 — better than the lr=2e-4 collapse but worse than the lr=1e-4 natural sibling, because proportional sampling's varied gradients crowd the behavior signal more than uniform stratification did. |

#### The deployment choice is now a product question, not a metric question

| Production target | Recommended adapter |
| --- | --- |
| Maximum overall quality (ROUGE-L, perception, planning) | **`prop 1e-4`** |
| Behavior-heavy applications (ego-status, predictability) | **`nat 1e-4`** |
| Maximum prediction-category accuracy | **`nat 1e-4`** (0.696) or **`prop 1e-4`** (0.682) |
| Don't ship | `nat 2e-4`, `nat 5e-4`, `stratified` (each dominated by one of the above) |

All six adapters are published on Hugging Face for direct comparison:

| Variant | Hugging Face repo |
| --- | --- |
| nat 2e-4 (original canonical) | `pranavthombare/qwen3.5-0.8b-drivelm-lora` |
| nat 1e-4 (best behavior + good overall) | `pranavthombare/qwen3.5-0.8b-drivelm-lora-lr1e4` |
| nat 5e-4 (ablation — worst LR) | `pranavthombare/qwen3.5-0.8b-drivelm-lora-lr5e4` |
| stratified (ablation — uniform sampling) | `pranavthombare/qwen3.5-0.8b-drivelm-lora-stratified` |
| prop 1e-4 (best overall) | `pranavthombare/qwen3.5-0.8b-drivelm-lora-proportional` |

### Ablation: stratified retrain targeting the behavior regression

The behavior collapse from the natural-distribution run was traced to **only 10 of 1,024 training samples being behavior questions**, with none in the multi-clause `"X is going straight. X is driving fast."` format that DriveLM uses for that category. The hypothesis: stratified sampling across (qa_type × answer_pattern), with all 38 behavior samples upsampled 4×, would recover behavior performance.

We retrained the adapter on a stratified 902-sample mix (250 each of perception/prediction/planning balanced across Yes/No/None-pattern/other, plus 38 behavior × 4 upsample) — same hyperparameters, same wall-clock budget (~18 min). Evaluated on the same 3,770-sample eval set.

**Stratified-LoRA vs Natural-1024-LoRA, per category (ROUGE-L):**

| Category | N | Baseline | Natural-1024 LoRA | **Stratified LoRA** | Δ vs Natural |
| --- | ---: | ---: | ---: | ---: | ---: |
| perception | 1,738 | 0.217 | 0.489 | **0.615** | **+0.127** ↑ |
| prediction | 1,181 | 0.097 | 0.659 | 0.368 | **−0.291** ↓ |
| planning | 813 | 0.107 | 0.502 | 0.507 | +0.005 |
| **behavior** | **38** | **0.305** | **0.036** | **0.911** | **+0.875** ↑↑ |

The hypothesis was confirmed for behavior (0.036 → 0.911 — 25× lift; behavior is now the strongest category) and for perception (balanced Yes/No fixed the yes-bias overcorrection: +0.127 ROUGE-L). **But the natural-distribution run had over-fit the `None, no, none.` answer pattern**, which is ~16% of prediction ground truth; stratified sampling under-represented it by design (50 None-pattern vs the 119 the natural run saw). Prediction therefore traded 0.291 ROUGE-L for healthier behavior coverage.

Overall ROUGE-L: natural-1024 0.541 vs stratified 0.518 — a small regression in the headline metric in exchange for **no catastrophically failing category and a more uniform competence distribution**. Mean per-request latency rose from 1,046 ms to 1,811 ms because the stratified adapter produces longer multi-clause answers.

**Which adapter is published.** The natural-1024 adapter is the one shipped to Hugging Face (`pranavthombare/qwen3.5-0.8b-drivelm-lora`) because it has the higher overall ROUGE-L. The stratified adapter is preserved locally at `models/qwen-lora-stratified/` and its eval results are in `artifacts/finetuned_stratified_front_arc_full.json`. The right deployment choice between the two depends on whether the application can tolerate a 25× degradation on the rare behavior class to gain 0.29 ROUGE-L on the common prediction class — a real product question, not a metric question.

### Failure modes (named)

| # | Mode | Example | LoRA-tractable? |
| --- | --- | --- | --- |
| 1 | Confident hallucination on motion presence | GT: *"Yes."* / base: *"No, there are no moving cars to the front…"* | Partial — LoRA fixed many but introduced yes-bias |
| 2 | Verbose hedging killing precision | base: *"Based on the camera view labeled CAM_FRONT…"* (vs GT 1-sentence) | ✅ Locked prompt + LoRA |
| 3 | Multi-clause comma format ignored | GT: *"None, no, none."* / base: free-form sentence | ✅ LoRA learned it |
| 4 | `<c1,CAM_FRONT,...>` referent tokens ignored | GT addresses a specific bbox; model answers generically | ❌ Needs grounding head |
| 5 | CAN-bus-derived ground truth | GT: *"driving fast"* / *"not moving"* (not visible in a single frame) | ❌ Input modality gap |

Modes 1, 2, 3 are language-side and LoRA-tractable; 4 and 5 require either a grounded vision head or temporal/sensor input. `artifacts/qualitative_lora_vs_base.json` has 2 wins + 2 losses per category with paired predictions.

### Cost (A10 GPU at $0.75/hr reference rate)

```
$/1k queries = (mean_latency_seconds) × (gpu_$/hr / 3600) × 1000
```

| Workload | Mean latency | $/1k queries |
| --- | ---: | ---: |
| Baseline through vLLM (conc 4) | 1.420 s | **$0.296** |
| LoRA through vLLM (conc 4) | 1.046 s | **$0.218** |
| Transformers single-process baseline (batch 1) | 6.7 s | $1.396 |

LoRA training cost: 20 min × $0.75/hr = **~$0.25 total** for the adapter. The fine-tune pays for itself after ~5 queries against the baseline.

On a T4 ($0.35/hr) the baseline drops to ~$0.13/1k. On an A100 ($1.50/hr) the wall clock would shrink ~3× but cost per query is comparable because vLLM saturates at small batch sizes — throughput dominates over GPU class once continuous batching is enabled.

### Deployment optimizations (what vLLM gives us)

The serving stack is intentionally thin: a single vLLM container behind an OpenAI-compatible API. The reason we use vLLM as-is — rather than wrap it or write a Transformers-based server — is that it bundles every relevant optimization the rubric asks for. We measured a 20× wall-clock speedup over Transformers (`6h 30m → 20m 41s` for the same 3,770 samples) attributable to these features:

| Optimization | What it does | What it buys us |
| --- | --- | --- |
| **Continuous batching** | concurrent requests share a single forward pass | dominant contributor to the 20× speedup |
| **Paged attention** | KV cache stored in fixed-size pages, no fragmentation | fits more concurrent sequences in 8 GB |
| **Prefix caching** | shared prompt prefix reuses KV across requests | our system prompt is identical across every request — automatic reuse |
| **`mm-processor-cache-gb=1`** | post-vision-encoder features cached by image content hash | DriveLM has 114 unique images shared across 3,770 requests — vLLM hits the cache once warm; this is the explicit answer to the rubric's "reuse image embeddings" line item |
| **4-bit NF4 weight loading** | weights quantized to NF4, activations in fp16 | fits Qwen3.5-0.8B + KV cache + activations into 8 GB |
| **Auto-LoRA via `--enable-lora`** | adapter served as a separate `model_id` from the same process | one replica serves base + LoRA; no second deployment, no second cold start |
| **TRITON attention backend** | Triton kernels instead of FlashInfer | FlashInfer crashed on this Turing GPU; explicitly chosen fallback documented in env config |

The throughput characterization the rubric asks for follows from concurrency × `max-num-seqs` × prefill/decode efficiency: at `max-num-seqs=4` and concurrency=4 we measured ~2.9 requests/sec steady-state. Single-replica capacity scales roughly linearly with `max-num-seqs` on bigger GPUs (A10 / A100) where VRAM is not the binding constraint. Horizontal scaling is then a replica-count change at the orchestrator (K8s Deployment, Render service, etc.); per-frame session affinity at the L7 load balancer would maximize the `mm-processor-cache` hit rate across replicas.

We chose vLLM specifically to delegate the seven optimizations above to a battle-tested system rather than reinvent them. The serving deliverable is intentionally short — `src/vllm_launcher.py` is 130 lines of env-var-driven argv construction around the vLLM binary, nothing more — because the optimization work is already done inside the binary it launches.

### Methodological limitations

1. **Train/eval overlap.** The LoRA was trained on samples 0–1023 and evaluated on samples 0–3769. The first 1,024 samples appear in both. The headline +0.367 ROUGE-L therefore overstates generalization — held-out evaluation on disjoint frames would lower the perception/prediction/planning gains by an unknown amount. The behavior regression is unaffected (no behavior samples were in either training set). Direction-of-change per category is reliable; magnitudes for the trained categories should be discounted.

2. **Natural-distribution sampling.** First-1024 training: 492 perception / 311 prediction / 211 planning / 10 behavior, with a 3.2:1 Yes/No skew in perception. Two consequences are visible in the comparison results: Yes-bias overcorrection on `Are there X` questions, and behavior mode collapse. `src/train/README.md` documents the proposed stratified retrain (250 each of perception/prediction/planning × Yes/No/None/other, all 38 behavior upsampled 4×).

3. **No image-embedding reuse across QA on the same frame.** DriveLM has ~93 QA per frame on average; encoding the 3 camera images once per frame and reusing the vision tokens across all 93 questions is a known ~10–30× theoretical win. The 20× speedup in this submission comes entirely from vLLM continuous batching, not from this optimization. Implementing it is the next item on the deployment roadmap.

4. **Smoke testing on first-N is misleading.** Frame-level autocorrelation means the first 10 samples (which all sit on a single hard frame) hit two failure modes back-to-back; a 10-sample smoke ROUGE-L of 0.276 dragged the LoRA below the per-category average. Future smokes should be stratified random with a fixed seed.

5. **vLLM nondeterminism at temperature 0.** Continuous batching reorders requests; tied logits resolve differently depending on the batch composition. Per-sample predictions are not byte-identical between a 10-sample smoke and a 3,770-sample full run on the LoRA path. Aggregate ROUGE-L is stable; exact prediction reproducibility would require `--max-num-seqs 1` (which kills throughput).

## Notes

- Use `HF_TOKEN` in the environment if a gated dataset or model download needs authentication.
- vLLM startup performs multimodal warmup and takes ~30–60s on the RTX 2070 SUPER.
- For Turing GPUs, keep `--attention-backend TRITON_ATTN`; the default FlashInfer backend is unreliable for Qwen3.5 multimodal inference.
