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
src/data/pipeline.py             # Flattens DriveLM scene/frame QA into samples
src/serve/qwen.py                # Shared Qwen image selection and prompt helpers
src/serve/vllm_server.py         # vLLM launcher (base model + optional LoRA)
src/eval/benchmark_endpoint.py   # Async concurrent benchmark against vLLM (default)
src/eval/benchmark.py            # Transformers benchmark (single-process, no vLLM)
src/eval/evaluate_finetuned.py   # Evaluate LoRA adapters via Transformers
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

## Assignment Workflow

1. Check that DriveLM questions link to local nuScenes images:

   ```bash
   .venv/bin/python src/data/pipeline.py
   ```

2. Start the vLLM server (separate terminal, runs continuously):

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   .venv-vllm/bin/python -m src.serve.vllm_server
   ```

   Confirm it's up:

   ```bash
   curl http://127.0.0.1:8001/v1/models
   ```

3. Run the concurrent baseline benchmark against vLLM:

   ```bash
   .venv/bin/python src/eval/benchmark_endpoint.py \
     --camera-mode front-arc \
     --concurrency 4 \
     --output-json artifacts/baseline_front_arc_full.json
   ```

4. Fine-tune a LoRA adapter (stop vLLM first to free GPU):

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   .venv/bin/python src/train/finetune.py \
     --num-samples 1024 \
     --camera-mode front \
     --quantization auto
   ```

5. Restart vLLM — the launcher auto-attaches `models/qwen-lora/` if it exists:

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   .venv-vllm/bin/python -m src.serve.vllm_server
   ```

   Then benchmark the LoRA path by setting `--model-id drivelm-lora`:

   ```bash
   .venv/bin/python src/eval/benchmark_endpoint.py \
     --model-id drivelm-lora \
     --camera-mode front-arc \
     --concurrency 4 \
     --output-json artifacts/finetuned_front_arc_full.json
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

## Transformers Benchmark (alternate, no vLLM)

`src/eval/benchmark.py` runs the same evaluation through Transformers in-process — useful for environments without vLLM, for debugging the prompt path, or for sanity-checking outputs against the vLLM client. On an RTX 2070 SUPER it is ~5× slower than the vLLM path even with `--batch-size 4` because there is no continuous batching.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python src/eval/benchmark.py \
  --camera-mode front-arc \
  --image-long-edge 448 \
  --max-new-tokens 64 \
  --batch-size 4 \
  --output-json artifacts/qwen_baseline_results.json
```

## Fine-Tuning

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python src/train/finetune.py \
  --num-samples 1024 \
  --camera-mode front \
  --image-long-edge 448 \
  --quantization auto
```

Adapters are saved under:

```text
models/qwen-lora
```

For an 8GB GPU, start with `front` or `front-arc` before trying all-camera training.

## Evaluating a LoRA Adapter

Two paths, both work:

- **Via vLLM (recommended):** restart `src.serve.vllm_server` after training; it auto-registers the adapter as `drivelm-lora`. Hit `benchmark_endpoint.py --model-id drivelm-lora`.
- **Via Transformers:** `src/eval/evaluate_finetuned.py` loads the base model + PEFT adapter and runs the same benchmark loop. Single-process, no server needed.

## Docker

The image is `vllm/vllm-openai` with the `src/` tree copied in. Mount local `data/` and `models/`:

```bash
docker compose up --build
```

OpenAI-compatible API at:

```text
http://127.0.0.1:8001/v1
```

## Notes

- Use `HF_TOKEN` in the environment if a gated dataset or model download needs authentication.
- vLLM startup performs multimodal warmup and takes ~30–60s on the RTX 2070 SUPER.
- For Turing GPUs, keep `--attention-backend TRITON_ATTN`; the default FlashInfer backend is unreliable for Qwen3.5 multimodal inference.
- The prompt is locked to `"Question: {question}\nAnswer in one short sentence."` in `src/serve/qwen.py`. A 3-point ablation showed this beats `"Answer concisely."` (token-F1 0.30 vs 0.17) and beats no constraint (ROUGE-L 0.37 vs 0.18) on a 10-sample smoke.
