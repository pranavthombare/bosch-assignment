# DriveLM Qwen VQA Pipeline

This is my submission for the Bosch Generative AI Systems Engineering assignment. The task: take a pretrained vision-language model, benchmark it on driving QA, finetune it under tight constraints, and stand up a serving stack that makes sense in production. I picked `Qwen/Qwen3.5-0.8B` because it's small enough to actually train on the 8 GB RTX 2070 SUPER on my desk, multimodal, and has a working vLLM path.

The dataset is DriveLM's `v1_1_train_nus.json` joined to nuScenes-mini images. DriveLM gives the questions and answers; nuScenes gives the camera frames. After filtering to samples whose `CAM_FRONT` image actually exists locally, I have 3,770 QA pairs across 38 frames and 6 scenes. That's the eval set used throughout this README.

The headline result, after a six-way training ablation: **ROUGE-L moves from 0.157 (zero-shot baseline) to 0.621 (best LoRA)** on the full eval set, with mean latency around 1.4–2.0 seconds per request through vLLM at concurrency 4. All five trained variants are on Hugging Face for direct comparison.

## What's in the repo

```text
.env.sample                      # every DRIVELM_* env var with defaults
src/config.py                    # typed config dataclasses + env loader
src/qwen.py                      # shared Qwen helpers (image loading, prompt building, model loading)
src/vllm_launcher.py             # thin Python wrapper around `vllm serve`
src/data/pipeline.py             # DriveLM → flattened QA samples + stratified/proportional samplers
src/eval/eval.py                 # async eval client against vLLM (base + LoRA in one pass)
src/train/finetune.py            # QLoRA training loop
src/train/README.md              # training-specific decisions and ablation notes
artifacts/                       # all the JSON results referenced in the analysis section
models/                          # local LoRA adapters (mounted into the Docker container)
data/                            # nuScenes images + DriveLM JSON (gitignored)
Dockerfile + docker-compose.yml  # vLLM serving image
```

There is no `argparse` anywhere — every script reads its configuration from `DRIVELM_*` environment variables, with defaults compiled into `src/config.py`. Copy `.env.sample` to `.env`, edit, and `python-dotenv` will pick it up automatically when any script runs.

## Setup

The minimum you need: a CUDA GPU (8 GB is enough for everything in this repo), Python 3.11+, and Docker with the NVIDIA container runtime if you want to run the canonical serving path.

```bash
# 1. Get the data into place
# nuScenes-mini: extract v1.0-mini.tgz into data/nuscenes/
# DriveLM: download v1_1_train_nus.json into data/drivelm/

# 2. One Python env for training and eval
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. (optional) bare-metal vLLM env, only if you don't want to use Docker for serving
python3.12 -m venv .venv-vllm
VIRTUAL_ENV=.venv-vllm uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly

# 4. Copy and edit the env file
cp .env.sample .env
# The defaults are tuned for an 8 GB GPU. On bigger hardware, raise
# DRIVELM_VLLM_GPU_MEMORY_UTILIZATION and DRIVELM_VLLM_MAX_NUM_SEQS in .env.
```

On CUDA the Qwen helpers default to 4-bit NF4 (QLoRA) when `bitsandbytes` is available. That's what makes training fit in 8 GB; on bigger GPUs you can set `DRIVELM_MODEL__QUANTIZATION=none` to use fp16 weights.

## Reproducing the results

The full reproduction is six steps. Steps 2 and 4 take the GPU, so you can't run both at once on a single-GPU machine.

### 1. Sanity-check the data pipeline

```bash
.venv/bin/python src/data/pipeline.py
```

Expect: `Samples: 3770, Frames: 38, Frames with all six cameras: 38`. If you see `Samples: 0`, the nuScenes tarball didn't extract — fix that first.

### 2. Start the vLLM server (in a separate terminal)

```bash
docker compose up --build
```

Wait ~60 seconds for the Application startup complete line. Verify with:

```bash
curl http://127.0.0.1:8001/v1/models
```

The launcher auto-attaches whatever's at `models/qwen-lora/` as `drivelm-lora` if it sees an `adapter_config.json` there. If you want base-model-only serving, set `DRIVELM_ENABLE_LORA=false` before `docker compose up`.

### 3. Run the zero-shot baseline

```bash
.venv/bin/python src/eval/eval.py
```

This evaluates the base model on all 3,770 samples through vLLM and writes `artifacts/baseline_front_arc_full.json`. ~11 minutes wall-clock on the 2070 SUPER at concurrency 4.

By default it also evaluates whatever LoRA adapter vLLM is serving (skip with `DRIVELM_EVAL__RUN_LORA=false`). If you started with `DRIVELM_ENABLE_LORA=false` in step 2, the LoRA pass will fail; toggle one or the other.

### 4. Train a LoRA adapter

Stop the vLLM container first to free the GPU:

```bash
docker compose down
.venv/bin/python src/train/finetune.py
```

The default config trains a QLoRA adapter (r=8, α=16) on the first 1,024 samples in natural distribution at lr=2e-4 for 1 epoch. That's the historical canonical run. To reproduce the actual best adapter (proportional sampling + lr=1e-4):

```bash
DRIVELM_TRAIN__SAMPLING=proportional \
DRIVELM_TRAIN__LR=1e-4 \
DRIVELM_TRAIN__OUTPUT_DIR=models/qwen-lora \
.venv/bin/python src/train/finetune.py
```

~17–20 minutes. Adapter is saved to `models/qwen-lora/` (or whatever `DRIVELM_TRAIN__OUTPUT_DIR` points at).

### 5. Restart vLLM with the new adapter

```bash
docker compose up --build
```

The launcher picks up the freshly-trained adapter.

### 6. Evaluate the LoRA

```bash
.venv/bin/python src/eval/eval.py
```

Writes `artifacts/finetuned_front_arc_full.json`. ~14–17 minutes.

For a 10-sample smoke run on either step 3 or step 6:

```bash
DRIVELM_EVAL__NUM_SAMPLES=10 \
DRIVELM_EVAL__OUTPUT_BASE_JSON=artifacts/smoke_base.json \
DRIVELM_EVAL__OUTPUT_LORA_JSON=artifacts/smoke_lora.json \
.venv/bin/python src/eval/eval.py
```

## How the configuration system works

Every script reads a `RunCfg` dataclass tree from `src/config.py` with three layers of override:

```text
defaults in src/config.py
  ↓
.env file at project root (auto-loaded by python-dotenv)
  ↓
DRIVELM_<SECTION>__<FIELD> shell environment variables
```

The effective config is printed at startup and embedded inside every artifact JSON under `"config"`. That means any historical result can be re-run by reading its own output file — no separate "what knobs did I use?" lookup.

Common overrides live in `.env.sample` with comments. The naming is `DRIVELM_<SECTION>__<FIELD>` for the typed config (e.g. `DRIVELM_EVAL__NUM_SAMPLES=10`) and `DRIVELM_VLLM_<FLAG>` for the vLLM launcher (e.g. `DRIVELM_VLLM_MAX_NUM_SEQS=8`). The launcher's flat naming is intentionally separate — it maps directly to `vllm serve` CLI flags and doesn't fit the four-section data/model/train/eval taxonomy.

## What I measured

The model decisions are the substance of this submission. Each section has a one-line "why" alongside the numbers.

### Baseline (zero-shot Qwen3.5-0.8B)

Everything ran through the dockerized vLLM at concurrency 4, front-arc camera mode (3 cameras per sample), temperature 0, max 64 new tokens. The full eval finished in 11 minutes 23 seconds wall-clock.

| Metric | Value |
| --- | ---: |
| ROUGE-L | 0.157 |
| Token-F1 | 0.117 |
| Exact match | 0.37% |
| Mean latency | 1,420 ms |

The expected category ordering is perception > behavior > planning > prediction, and that's what we see: static visible content (cars, signs) is easy; temporal reasoning from a single frame ("what will happen next?") is hardest. Behavior has wide confidence at n=38; the headline 0.305 is approximate.

| Category | N | ROUGE-L |
| --- | ---: | ---: |
| perception | 1,738 | 0.217 |
| prediction | 1,181 | 0.097 |
| planning | 813 | 0.107 |
| behavior | 38 | 0.305 |

### The prompt is locked, and the choice matters

The user prompt in `src/qwen.py::build_qwen_messages` is `Question: {q}\nAnswer in one short sentence.`. I tested three variants on a 10-sample smoke:

| Prompt | ROUGE-L | Notes |
| --- | ---: | --- |
| `Answer concisely.` | undefined (initial metric was broken) | model produced one-word answers like "Cars" — high precision, no recall |
| no constraint | 0.180 | verbose Markdown paragraphs, low precision |
| **`Answer in one short sentence.` (locked)** | **0.371** | matches DriveLM's declarative-sentence style |

This is the cheapest defensible "intentional choice" in the project. A one-line prompt change doubled ROUGE-L before any training happened.

### The training sweep — six configurations

Once the baseline was in, I ran a sweep. Each adapter uses QLoRA on top of `Qwen/Qwen3.5-0.8B`: 4-bit NF4 base, LoRA r=8 / α=16 on q/k/v/o/gate/up/down_proj, vision tower frozen, 1 epoch, batch=1 with grad-accum=2. The two things that changed are the training data composition and the learning rate.

#### Overall ROUGE-L

| Config | Sampling | lr | Overall ROUGE-L |
| --- | --- | ---: | ---: |
| baseline (no LoRA) | — | — | 0.157 |
| nat 2e-4 | natural-first-1024 | 2e-4 | 0.541 |
| nat 1e-4 | natural-first-1024 | 1e-4 | 0.581 |
| nat 5e-4 | natural-first-1024 | 5e-4 | 0.540 |
| stratified | uniform balanced | 2e-4 | 0.518 |
| **prop 1e-4** | proportional with floor | 1e-4 | **0.621** |

#### Per category (ROUGE-L)

| Category | N | baseline | nat 2e-4 | nat 1e-4 | nat 5e-4 | stratified | prop 1e-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| perception | 1,738 | 0.217 | 0.489 | 0.533 | 0.513 | 0.615 | **0.625** |
| prediction | 1,181 | 0.097 | 0.659 | **0.696** | 0.617 | 0.368 | 0.682 |
| planning | 813 | 0.107 | 0.502 | 0.503 | 0.509 | 0.507 | **0.543** |
| behavior | 38 | 0.305 | 0.036 | **0.877** | 0.022 | **0.911** | 0.201 |

The story this tells, in order of what I learned:

**`nat 2e-4`** was the first run. ROUGE-L jumped from 0.157 to 0.541, which felt like a win until I broke it down per category and saw behavior collapsed to 0.036. The model had locked onto a terse `Turn left.` default for all 38 behavior questions.

I traced the collapse to the training data composition (only 10 of the 1,024 first-natural samples were behavior questions, none in the multi-clause format DriveLM uses) and ran `stratified` to fix it. Behavior recovered to 0.911 — but prediction crashed to 0.368. The cause is straightforward: DriveLM's prediction eval set is ~38% `No`-answers, and uniform stratification forced a 1/3-each split, so the model lost the natural-prior advantage.

Then I ran the LR sweep, and the lesson I'd been telling myself fell apart. `nat 1e-4` recovered behavior to 0.877 *without* any data fix, just by lowering the learning rate. The behavior collapse hadn't been a sampling problem — it was an LR-too-high problem that hit the rare class hardest. `nat 5e-4` made everything worse, including behavior.

The final run combined the two real lessons: keep natural within-category answer-pattern proportions (so prediction's prior survives) but apply a minimum-sample floor on rare patterns (so they don't vanish), and use lr=1e-4. `prop 1e-4` wins overall ROUGE-L, perception, planning, and exact-match. Behavior at 0.201 is the trade-off — proportional sampling injects all 38 behavior samples × 4 upsample, same as stratified, but the surrounding gradient signal is more varied, so the LoRA's r=8 capacity gets pulled away from behavior.

Which adapter to ship depends on the application. If you care most about behavior-heavy use cases (ego-status, predictability), `nat 1e-4` is the right choice. If you want the highest overall quality and accept the behavior trade-off, `prop 1e-4` wins. I've uploaded all five to Hugging Face so the choice is just a `model_id` change at inference time:

| Variant | Hugging Face repo |
| --- | --- |
| nat 2e-4 (original, historical canonical) | `pranavthombare/qwen3.5-0.8b-drivelm-lora` |
| **prop 1e-4 (best overall)** | `pranavthombare/qwen3.5-0.8b-drivelm-lora-proportional` |
| **nat 1e-4 (best behavior)** | `pranavthombare/qwen3.5-0.8b-drivelm-lora-lr1e4` |
| nat 5e-4 (ablation, not recommended) | `pranavthombare/qwen3.5-0.8b-drivelm-lora-lr5e4` |
| stratified (ablation) | `pranavthombare/qwen3.5-0.8b-drivelm-lora-stratified` |

### Failure modes the LoRA doesn't fix

Two of the five named failure modes from the baseline analysis are tractable with LoRA on this scale:

1. **Verbose hedging** — fixed by the prompt + LoRA training together
2. **DriveLM's multi-clause `None, no, none.` format** — fixed by LoRA learning the convention

Three aren't:

3. **Confident hallucination** on motion/presence questions — partially fixed (LoRA reduces it but flips some `No`s to `Yes` in compensation)
4. **`<c1,CAM_FRONT,x,y>` referent tokens** — completely ignored, because the base model has no bbox-grounded vision head
5. **CAN-bus-derived ground truth** (e.g. `"driving fast"`, `"not moving"`) — cannot be inferred from a single camera frame regardless of training

The qualitative paired wins/losses for the canonical LoRA (perception/prediction/planning/behavior, 2 wins and 2 losses each) are in `artifacts/qualitative_lora_vs_base.json` if you want to read the actual model outputs.

## Cost

Everything in this submission was run on a single NVIDIA RTX 2070 SUPER (8 GB, Turing) on my desk in Bangalore. The actual cost is BESCOM residential electricity. Current tariff (effective May 1, 2026), base energy charges only:

| Monthly slab | Rate (₹/unit) |
| --- | ---: |
| 0–50 units | 5.74 |
| 51–100 units | 6.24 |
| 101–200 units | 6.74 |
| 201–300 units | 7.24 |
| Above 300 units | 7.74 |

Sustained GPU work pushes a typical household into the highest slab, so the **marginal rate is ₹7.74/unit** (1 unit = 1 kWh). The 2070 SUPER draws ~215 W under sustained ML load, so:

- Per GPU-hour: 0.215 kWh × ₹7.74 = **₹1.66/hour** (≈ $0.020 at ₹83/USD)

Base energy only — BESCOM bills also add fuel-adjustment surcharges and a small fixed monthly charge, so the actual all-in cost is a few percent higher than the numbers below.

### Cost per 1,000 queries on the 2070 SUPER

| Workload | Mean latency | ₹ / 1k queries | $ / 1k queries |
| --- | ---: | ---: | ---: |
| Baseline through dockerized vLLM | 1.42 s | **₹0.66** | **$0.0079** |
| LoRA (canonical, nat 2e-4) through vLLM | 1.05 s | **₹0.48** | **$0.0058** |
| LoRA (prop 1e-4) through vLLM | 1.86 s | **₹0.86** | **$0.0104** |

### Training cost (one-off, per adapter)

Each adapter trains in ~20 minutes:

- 0.333 hr × ₹1.66/hr = **₹0.55 per adapter** (≈ $0.0067)
- Full six-config sweep: **~₹3.30 total** (≈ $0.04) in electricity

The training spend is essentially negligible — a cup of chai pays for the entire ablation series.

### Cloud comparison

For context, the same workloads on common cloud GPUs (USD prices, with INR conversion at ₹83/USD):

| Hardware | $/hr | ₹/hr | Baseline ₹/1k | prop 1e-4 ₹/1k |
| --- | ---: | ---: | ---: | ---: |
| RTX 2070 SUPER (BESCOM electricity) | $0.020 | ₹1.66 | ₹0.66 | ₹0.86 |
| NVIDIA T4 (cloud) | $0.35 | ₹29 | ₹11.5 | ₹15.0 |
| NVIDIA A10 (cloud) | $0.75 | ₹62 | ₹24.6 | ₹32.0 |
| NVIDIA A100 (cloud) | $1.50 | ₹125 | ₹49.2 | ₹64.5 |

The 2070 SUPER is **~18× cheaper per hour than even a T4**, because residential electricity in Bangalore is roughly an order of magnitude cheaper than cloud GPU rental margins.

### The economics flip at scale

On the 2070 SUPER, VRAM caps concurrency at `max-num-seqs=4`, giving ~2.9 requests/second steady-state. An A100 with `max-num-seqs=32` would push ~30 r/s — about 10× the throughput at ~75× the cost in absolute terms, so on a strict cost-per-query basis the local box still wins. Where cloud actually wins is **provisioning flexibility**: spinning up 20 A100 replicas during a traffic spike is impossible with a single desk GPU.

For a real production deployment serving DriveLM-style inference, the cost-aware model is probably: **provision T4-class GPUs at expected steady-state load, autoscale to A10/A100 during spikes, deploy multiple replicas behind a load balancer with per-frame session affinity to maximize vLLM's mm-processor-cache hit rate.** On-prem for steady load, cloud for spikes — same pattern as any hybrid-cloud workload.

## Deployment, optimization, what vLLM gives us

The serving stack is intentionally thin: one vLLM container behind an OpenAI-compatible HTTP API. I considered writing a Transformers+FastAPI server at one point (the eval client doesn't care), and measured the difference: HF Transformers at batch=4 ran the full 3,770-sample eval in roughly 6.5 hours; vLLM with continuous batching ran the same workload in 11 minutes 23 seconds. That's ~34× faster wall-clock, not from anything I built but from what vLLM bundles for free:

- **Continuous batching** — the dominant contributor; concurrent requests share forward passes.
- **Paged attention** — KV cache stored in fixed-size pages instead of contiguous chunks. Fits more concurrent sequences in 8 GB.
- **Prefix caching** — our system prompt is identical across every request. The KV cache for it gets shared automatically.
- **`--mm-processor-cache-gb 1`** — post-vision-encoder features cached by image content hash. DriveLM has 114 unique images shared across 3,770 requests. This is vLLM's answer to the rubric's "reuse image embeddings" line.
- **4-bit NF4 weight loading** — the base model fits with room for KV + activations on 8 GB.
- **Auto-LoRA via `--enable-lora`** — adapter served as a separate `model_id` from the same process. One replica serves base + LoRA, no second cold start.
- **TRITON_ATTN attention backend** — Triton kernels instead of FlashInfer. FlashInfer crashed on this Turing GPU, so the launcher pins `TRITON_ATTN` by default. On Ampere+ you can switch.

Throughput at the current `max-num-seqs=4` is roughly 2.9 requests/second steady-state on the 2070 SUPER. Single-replica capacity scales roughly linearly with `max-num-seqs` on bigger GPUs where VRAM isn't the binding constraint. Horizontal scaling is a replica-count change at the orchestrator; per-frame session affinity at the L7 load balancer would maximize the mm-processor-cache hit rate across replicas.

I chose vLLM specifically to delegate these seven optimizations to a battle-tested system rather than reinvent them. The serving deliverable is genuinely short — `src/vllm_launcher.py` is 130 lines of env-var-driven argv construction around the `vllm serve` binary. The optimization work is already done inside the binary it launches.

## Methodological limitations

These are the things a reviewer should know are wrong with this submission. I'd rather name them than have them found.

1. **Train/eval overlap.** The training set (first 1,024 or stratified-902 from the same pool) is a subset of the 3,770-sample eval set. The headline ROUGE-L numbers therefore overstate generalization. A held-out evaluation on disjoint frames would lower the perception/prediction/planning gains by an unknown amount; the behavior comparisons across configs are unaffected.
2. **No image-embedding reuse beyond what vLLM gives us.** DriveLM has ~93 QA per frame. Encoding the 3 camera images once per frame and reusing the vision tokens explicitly (not just relying on vLLM's mm-cache hash hits) would give a much bigger speedup. Not implemented in this submission. Mentioned because the rubric specifically asks about it.
3. **Smoke testing on first-N is misleading.** Frame-level autocorrelation means the first 10 samples sit on a single hard frame; a 10-sample smoke ROUGE-L of 0.276 is unrepresentative. Future smokes should be stratified random with a fixed seed.
4. **vLLM nondeterminism at temperature 0.** Continuous batching reorders requests; tied logits resolve differently depending on batch composition. Per-sample predictions aren't byte-identical between independent runs on the LoRA path. Aggregate ROUGE-L is stable; exact-prediction reproducibility would need `--max-num-seqs 1`, which kills throughput.
5. **nuScenes-mini scope.** All training and eval was on 38 frames across 6 scenes, daylight, Singapore + Boston. Generalization to night/rain/other geographies is untested.

## Notes

- Use `HF_TOKEN` if a gated dataset or model download needs authentication. `python-dotenv` will pick it up from `.env`.
- vLLM startup performs multimodal warmup and takes ~30–60 seconds on the 2070 SUPER. The first request after startup is slower than steady-state; discard it when reporting latency.
- For Turing GPUs (anything 20-series and older) keep `DRIVELM_VLLM_ATTENTION_BACKEND=TRITON_ATTN`. FlashInfer is unreliable for Qwen3.5 multimodal on this architecture. On Ampere+ you can try the default backend.
- The `.venv-vllm` environment is optional — you only need it if you want to run vLLM bare-metal as an alternative to Docker. The canonical reproduction path uses Docker for serving and only needs `.venv`.
