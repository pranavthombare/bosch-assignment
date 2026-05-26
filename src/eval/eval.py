"""Async concurrent evaluation against a vLLM OpenAI-compatible endpoint.

A single invocation evaluates the base model and (if enabled) the LoRA adapter
back-to-back against the same sample order. Both runs hit the same warm vLLM
server, so the comparison is apples-to-apples. Toggle either path off with
``eval.run_base: false`` or ``eval.run_lora: false`` in the YAML.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RunCfg, config_dict, load_config, print_config
from src.data.pipeline import DriveLMDataset
from src.qwen import load_image, select_image_paths


SYSTEM_TEXT = (
    "You are answering autonomous-driving visual questions from synchronized "
    "nuScenes camera views. Use the camera labels when spatial context matters."
)


# ---------------------------------------------------------------------------
# Sample selection and metrics (also imported by src/train/finetune.py)
# ---------------------------------------------------------------------------


def first_valid_samples(dataset: DriveLMDataset, limit: int, nuscenes_dir: Path) -> list[dict]:
    """Walk the dataset and keep samples with at least one resolvable image.

    ``limit == 0`` (the default) returns every usable sample.
    """
    samples: list[dict] = []
    for sample in dataset.samples:
        image_paths = sample.get("image_paths", {})
        if not image_paths:
            continue
        if not any((nuscenes_dir / rel_path).exists() for rel_path in image_paths.values()):
            continue
        samples.append(sample)
        if limit > 0 and len(samples) >= limit:
            break
    return samples


def simple_token_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = set(pred_tokens) & set(ref_tokens)
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_text_metrics(predictions: list[str], references: list[str]) -> dict[str, float | str]:
    if not predictions:
        return {"metric_backend": "rouge_score", "exact_match": 0.0, "token_f1": 0.0}

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for key in rouge_totals:
            rouge_totals[key] += scores[key].fmeasure

    n = len(predictions)
    exact_match = sum(
        int(prediction.strip().lower() == reference.strip().lower())
        for prediction, reference in zip(predictions, references)
    ) / n
    token_f1 = sum(
        simple_token_f1(prediction, reference)
        for prediction, reference in zip(predictions, references)
    ) / n
    return {
        "metric_backend": "rouge_score",
        "rouge1": rouge_totals["rouge1"] / n,
        "rouge2": rouge_totals["rouge2"] / n,
        "rougeL": rouge_totals["rougeL"] / n,
        "exact_match": exact_match,
        "token_f1": token_f1,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def image_to_data_url(path: Path, max_long_edge: int | None) -> str:
    image = load_image(path, use_cache=True, max_long_edge=max_long_edge)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_chat_messages(
    selected_images: list[tuple[str, Path]],
    question: str,
    image_long_edge: int | None,
) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": SYSTEM_TEXT}]
    for camera, path in selected_images:
        content.append({"type": "text", "text": f"Camera view: {camera}"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path, image_long_edge)}})
    content.append({"type": "text", "text": f"Question: {question}\nAnswer in one short sentence."})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Single-pass eval against one served model id
# ---------------------------------------------------------------------------


async def evaluate_model(
    cfg: RunCfg,
    served_model_id: str,
    test_samples: list[dict],
    output_json: Path,
    label: str,
) -> dict:
    image_long_edge = cfg.data.image_long_edge if cfg.data.image_long_edge > 0 else None
    print(f"\n=== {label}: evaluating model_id={served_model_id!r} on {len(test_samples)} samples ===")

    client = AsyncOpenAI(
        base_url=cfg.eval.base_url,
        api_key=cfg.eval.api_key,
        timeout=cfg.eval.timeout,
        max_retries=cfg.eval.max_retries,
    )
    semaphore = asyncio.Semaphore(cfg.eval.concurrency)
    predictions: list[str | None] = [None] * len(test_samples)
    latencies_ms: list[float] = [0.0] * len(test_samples)
    selected_cams: list[list[str]] = [[] for _ in test_samples]
    errors: list[str | None] = [None] * len(test_samples)

    async def process(index: int, sample: dict) -> None:
        try:
            selected = select_image_paths(
                image_paths=sample["image_paths"],
                camera_mode=cfg.data.camera_mode,
                nuscenes_root=cfg.data.nuscenes_dir,
            )
            if not selected:
                raise RuntimeError("No usable image paths.")
            selected_cams[index] = [camera for camera, _ in selected]

            async with semaphore:
                messages = build_chat_messages(selected, sample["question"], image_long_edge)
                start = time.perf_counter()
                response = await client.chat.completions.create(
                    model=served_model_id,
                    messages=messages,
                    max_tokens=cfg.model.max_new_tokens,
                    temperature=cfg.eval.temperature,
                )
                latency_ms = (time.perf_counter() - start) * 1000

            predictions[index] = response.choices[0].message.content.strip()
            latencies_ms[index] = latency_ms
        except Exception as exc:
            errors[index] = f"{type(exc).__name__}: {exc}"

    tasks = [process(i, s) for i, s in enumerate(test_samples)]
    completed = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        completed += 1
        if completed % 50 == 0 or completed == len(tasks):
            print(f"[{label}] Completed {completed}/{len(tasks)}", flush=True)

    records = []
    finals_pred: list[str] = []
    finals_ref: list[str] = []
    finals_lat: list[float] = []
    for index, sample in enumerate(test_samples):
        record = {
            "id": sample.get("frame_token"),
            "question_type": sample.get("qa_type"),
            "question": sample["question"],
            "reference": sample["answer"],
            "prediction": predictions[index],
            "selected_cameras": selected_cams[index],
            "latency_ms": round(latencies_ms[index], 2),
            "error": errors[index],
        }
        records.append(record)
        if predictions[index] is not None:
            finals_pred.append(predictions[index])
            finals_ref.append(sample["answer"])
            finals_lat.append(latencies_ms[index])

    metrics = compute_text_metrics(finals_pred, finals_ref) if finals_pred else {}
    latency_summary = {
        "mean_ms": round(sum(finals_lat) / len(finals_lat), 2) if finals_lat else 0,
        "min_ms": round(min(finals_lat), 2) if finals_lat else 0,
        "max_ms": round(max(finals_lat), 2) if finals_lat else 0,
    }
    output = {
        "config": config_dict(cfg),
        "label": label,
        "served_model_id": served_model_id,
        "num_samples": len(test_samples),
        "num_succeeded": len(finals_pred),
        "num_failed": sum(1 for e in errors if e),
        "rouge": metrics,
        "latency": latency_summary,
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[{label}] rouge={json.dumps(metrics)}  latency={json.dumps(latency_summary)}  failed={output['num_failed']}")
    print(f"[{label}] saved → {output_json}")
    return output


# ---------------------------------------------------------------------------
# Top-level: load dataset once, run base + lora as configured
# ---------------------------------------------------------------------------


async def run(cfg: RunCfg) -> None:
    if not (cfg.eval.run_base or cfg.eval.run_lora):
        print("Both eval.run_base and eval.run_lora are false — nothing to do.")
        return

    print("Loading DriveLM Dataset...")
    dataset = DriveLMDataset(str(cfg.data.nuscenes_dir), str(cfg.data.drivelm_json))
    test_samples = first_valid_samples(dataset, cfg.eval.num_samples, cfg.data.nuscenes_dir)

    if cfg.eval.run_base:
        await evaluate_model(
            cfg,
            served_model_id=cfg.model.model_id,
            test_samples=test_samples,
            output_json=cfg.eval.output_base_json,
            label="base",
        )

    if cfg.eval.run_lora:
        await evaluate_model(
            cfg,
            served_model_id=cfg.model.lora_model_id,
            test_samples=test_samples,
            output_json=cfg.eval.output_lora_json,
            label="lora",
        )


def main() -> None:
    cfg = load_config()
    print_config(cfg)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
