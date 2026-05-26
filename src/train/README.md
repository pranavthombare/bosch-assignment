# Training

`finetune.py` runs a small Qwen LoRA/QLoRA SFT loop against the local DriveLM samples.

```bash
# defaults already match the run documented here; override via .env or shell
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python src/train/finetune.py
```

Override knobs from `.env.sample` (uncomment the ones you need):
```text
DRIVELM_TRAIN__NUM_SAMPLES=1024
DRIVELM_TRAIN__EPOCHS=1
DRIVELM_DATA__CAMERA_MODE=front-arc
DRIVELM_DATA__IMAGE_LONG_EDGE=448
DRIVELM_MODEL__QUANTIZATION=auto
DRIVELM_TRAIN__GRADIENT_ACCUMULATION_STEPS=2
DRIVELM_TRAIN__CHECKPOINT_EVERY=200
DRIVELM_TRAIN__OUTPUT_DIR=models/qwen-lora
```

## Key design choices

- **LoRA rank 8, α 16** on `q/k/v/o/gate/up/down_proj` — standard LLaMA-family targets, ~3.2M trainable params (0.37% of base).
- **Vision tower frozen** (`freeze_vision_modules`). The 0.8B model's CLIP-trained vision encoder is small and broadly capable; finetuning it on 1k samples risks catastrophic forgetting.
- **4-bit NF4 base** via bitsandbytes on CUDA. Required to fit Qwen3.5-0.8B + front-arc activations + LoRA grads in 8 GB.
- **Loss masked to assistant tokens only.** `build_supervised_inputs` tokenizes the prompt twice (once with `add_generation_prompt=True`, once with the full assistant turn) and uses the length difference to set `labels[:, :prompt_length] = -100`. Earlier versions of this file masked only the pad token, which made the model train on reproducing the question too — measurable harm before the fix.
- **Intermediate checkpoints every 200 steps.** A prior run crashed with CUDA OOM at step 900 (orphaned GPU processes from killed eval jobs) and lost ~45 min. The save_pretrained inside the loop now bounds the worst-case loss to ~10 min.

## Known sampling bias (current 1024-sample run)

`first_valid_samples(N)` walks the dataset in scene/frame/qa-type order, so the first 1024 inherit DriveLM's natural distribution. Empirically measured on the actual training slice:

| Category | Yes | No | None-ptn | other | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| perception | **123** | **38** | — | 331 | 492 |
| prediction | 7 | 119 | 49 | 136 | 311 |
| planning | 23 | 18 | 10 | 160 | 211 |
| behavior | 0 | 0 | 0 | 10 | 10 |

Three issues are visible by inspection:

1. **Yes/No imbalance in perception (3.2:1 in favor of Yes).** The trained LoRA flips some baseline-correct "No" answers to "Yes" on the eval set — a direct overcorrection.
2. **"None, no, none." pattern at 16 % of prediction training.** The LoRA over-applies the pattern to questions it doesn't fit (e.g. "What kind of traffic sign is `<obj>`?" → "None.").
3. **Behavior has only 10 / 1024 (≈1 %) training samples and none of the Yes/No format.** The LoRA collapses to a single terse answer ("Turn left.") for all behavior questions, regressing ROUGE-L from 0.371 → 0.036.

Quantitatively, the natural-distribution LoRA gave +0.37 ROUGE-L overall but −0.34 on behavior. See `artifacts/comparison.json` and `artifacts/qualitative_lora_vs_base.json` for the full breakdown.

## Recommended next training run

Stratify the training sample at two levels — category × answer-pattern. Target distribution:

| Category | Yes | No | None-ptn | other | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| perception | 50 | 50 | — | 150 | 250 |
| prediction | 50 | 50 | 50 | 100 | 250 |
| planning | 50 | 50 | 25¹ | 125 | 250 |
| behavior | — | — | — | 38² | 38 |
| **Total** | 150 | 150 | 75 | 413 | **788** |

¹ Capped to the number of None-pattern planning samples available.
² All 38 behavior samples, upsampled ~4× during training so the gradient weight is comparable to the ~250-sample categories.

This is not yet implemented in `finetune.py`. A `--stratified` flag plus a helper in `src/data/pipeline.py` is the minimum surface area. The hypothesis to validate: stratification recovers the behavior category without giving up the perception/prediction/planning wins of the natural-distribution run.

## Outputs

Adapters and processor files are written to `DRIVELM_TRAIN__OUTPUT_DIR` (default `models/qwen-lora`). `src/vllm_launcher.py` auto-attaches the adapter at startup if `adapter_config.json` exists there; restart the launcher after training and the next `src/eval/eval.py` run will exercise both base and LoRA.
