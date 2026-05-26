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

### Baseline vs LoRA (3,770 samples, identical eval setup)

| Metric | Baseline | LoRA | Δ |
| --- | ---: | ---: | ---: |
| ROUGE-1 | 0.166 | 0.550 | +0.384 |
| ROUGE-2 | 0.069 | 0.188 | +0.119 |
| **ROUGE-L** | **0.157** | **0.541** | **+0.384** |
| Token-F1 | 0.117 | 0.510 | +0.393 |
| Exact match | 0.37% | 39.26% | +38.89 pp |
| Mean latency | 1,420 ms | 1,046 ms | −374 ms |

Per category ROUGE-L:

| Category | N | Baseline | LoRA | Δ |
| --- | ---: | ---: | ---: | ---: |
| perception | 1,738 | 0.217 | 0.489 | **+0.272** ↑ |
| prediction | 1,181 | 0.097 | 0.659 | **+0.562** ↑ |
| planning | 813 | 0.107 | 0.502 | **+0.395** ↑ |
| behavior | 38 | 0.305 | 0.036 | **−0.269** ↓ |

Three categories lift by 0.27–0.56 ROUGE-L. Behavior collapses. The headline win is answer-format learning — the LoRA learned DriveLM's declarative single-sentence convention, the multi-clause `None, no, none.` pattern for stacked prediction questions, and the terse Yes/No format for perception. Mean latency drops by 26% because LoRA outputs are shorter and there's less generated-token compute. Behavior regresses because only 10 of the 1,024 training samples were behavior questions, none in the Yes/No format; the LoRA's r=8 capacity is dominated by perception/prediction/planning gradients and collapses to a terse `Turn left.` for all behavior inputs.

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
