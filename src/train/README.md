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

## Stratified retrain (implemented and ablated)

The proposed stratification is implemented in `src/data/pipeline.py::stratified_samples` and toggled by `DRIVELM_TRAIN__STRATIFIED=true`. The plan:

| Category | Yes | No | None-ptn | other | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| perception | 50 | 50 | — | 150 | 250 |
| prediction | 50 | 50 | 50 | 100 | 250 |
| planning | 50 | 50 | 25¹ | 125 | 250 |
| behavior | — | — | — | 38² | 38 |
| **Total** | 150 | 150 | 75 | 413 | **788** distinct, **902** with behavior 4× upsample |

¹ Capped to the number of None-pattern planning samples available.
² All 38 behavior samples replicated 4× in the training list so the gradient weight is comparable to the ~250-sample categories.

```bash
DRIVELM_TRAIN__STRATIFIED=true \
.venv/bin/python src/train/finetune.py
```

### Postscript: the LR sweep changed the conclusion

After running the stratified ablation, a 3-point LR sweep (1e-4 / 2e-4 / 5e-4) on the original natural-distribution data showed that **the behavior collapse was an LR effect, not a sampling effect.** Just lowering LR to 1e-4 recovered behavior to ROUGE-L 0.877 — almost matching the stratified run's 0.911 — without any data change and without the prediction regression.

This means the stratified ablation, while measurably useful for behavior, **was solving the wrong problem.** The right diagnosis was hyperparameter, not data composition.

The follow-up experiment combined both signals — **proportional sampling (preserves natural within-category priors with a min-floor on rare patterns) + lr=1e-4** — and produced the best overall adapter (ROUGE-L 0.621, see top-level README's "Baseline vs LoRA — full ablation series" table).

### Result of the original stratified ablation

The hypothesis (stratification recovers behavior without losing the wins) was **partially confirmed**:

- ✅ Behavior recovered: ROUGE-L 0.036 → **0.911** (25× lift, now the strongest category)
- ✅ Perception improved: 0.489 → **0.615** (+0.127 — Yes-bias overcorrection fixed by balanced Yes/No)
- ✅ Planning unchanged: 0.502 → 0.507
- ❌ Prediction regressed: 0.659 → **0.368** (−0.291). The natural-distribution training had **119 No + 49 None-pattern prediction samples**; the eval set is rich in those exact answer patterns. Stratified sampling deliberately under-represented them (50 each), so the adapter no longer over-fits the `None, no, none.` shortcut and gives back ROUGE-L on that category.

Overall ROUGE-L 0.541 → 0.518 (small headline regression in exchange for a uniformly competent adapter with no catastrophic category failure).

**Published artifact policy.** All six adapters from the sweep are published on Hugging Face. The original `pranavthombare/qwen3.5-0.8b-drivelm-lora` remains as the historical canonical (nat-1024 lr=2e-4); the newer `qwen3.5-0.8b-drivelm-lora-proportional` is the recommended one for max overall quality, and `qwen3.5-0.8b-drivelm-lora-lr1e4` is recommended for behavior-heavy applications. Pulling whichever one you want back into `models/qwen-lora` from HF is the standard reproduction step; `vllm_launcher.py` auto-attaches whatever's there.

The "weighted stratification" idea this section originally proposed — preserve natural within-category proportions with a min-floor on rare patterns — was implemented in `proportional_samples()` and was the run that produced the best overall adapter. The genuinely-next experiment would push behavior upsampling higher (8× or 12× instead of 4×) to close the behavior gap on the proportional variant without giving up the overall-quality wins.

## Outputs

Adapters and processor files are written to `DRIVELM_TRAIN__OUTPUT_DIR` (default `models/qwen-lora`). `src/vllm_launcher.py` auto-attaches the adapter at startup if `adapter_config.json` exists there; restart the launcher after training and the next `src/eval/eval.py` run will exercise both base and LoRA.
