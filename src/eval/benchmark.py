import argparse
import json
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.pipeline import DriveLMDataset
from src.serve.qwen import (
    DEFAULT_QWEN_MODEL_ID,
    generate_qwen_answer,
    generate_qwen_answers_batch,
    load_qwen_model,
    load_qwen_processor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Qwen on DriveLM + nuScenes samples.")
    parser.add_argument("--model-id", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--nuscenes-dir", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--drivelm-json", type=Path, default=Path("data/drivelm/v1_1_train_nus.json"))
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of local usable samples to evaluate. Use 0 for all samples.",
    )
    parser.add_argument("--camera-mode", choices=["front", "front-arc", "all", "mosaic"], default="front-arc")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", choices=["auto", "none", "4bit", "8bit"], default="auto")
    parser.add_argument(
        "--image-long-edge",
        type=int,
        default=448,
        help="Resize each camera image to this max long edge before sending it to Qwen. Use 0 to disable.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Samples per model.generate call. 1 = original per-sample loop.",
    )
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/qwen_baseline_results.json"))
    return parser.parse_args()


def first_valid_samples(dataset: DriveLMDataset, limit: int, nuscenes_dir: Path):
    samples = []
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


def main():
    args = parse_args()
    print(f"Loading baseline VLM: {args.model_id}")

    processor = load_qwen_processor(args.model_id)
    model, device = load_qwen_model(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
    )
    model.eval()

    print("Loading DriveLM Dataset...")
    dataset = DriveLMDataset(str(args.nuscenes_dir), str(args.drivelm_json))
    test_samples = first_valid_samples(dataset, args.num_samples, args.nuscenes_dir)
    print(f"Running benchmark on {len(test_samples)} samples...")

    predictions = []
    references = []
    records = []
    latencies_ms = []
    batch_size = max(1, args.batch_size)
    print(f"Generation batch size: {batch_size}")

    processed = 0
    for batch_start in range(0, len(test_samples), batch_size):
        batch = test_samples[batch_start : batch_start + batch_size]
        start = time.perf_counter()
        if batch_size == 1:
            sample = batch[0]
            results = [
                generate_qwen_answer(
                    model=model,
                    processor=processor,
                    image_paths=sample["image_paths"],
                    question=sample["question"],
                    nuscenes_root=args.nuscenes_dir,
                    camera_mode=args.camera_mode,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    max_image_long_edge=args.image_long_edge,
                )
            ]
        else:
            results = generate_qwen_answers_batch(
                model=model,
                processor=processor,
                samples=batch,
                nuscenes_root=args.nuscenes_dir,
                camera_mode=args.camera_mode,
                device=device,
                max_new_tokens=args.max_new_tokens,
                max_image_long_edge=args.image_long_edge,
            )
        batch_latency_ms = (time.perf_counter() - start) * 1000
        per_sample_latency_ms = batch_latency_ms / len(batch)

        for offset, (sample, result) in enumerate(zip(batch, results)):
            pred = result.prediction
            predictions.append(pred)
            references.append(sample["answer"])
            latencies_ms.append(per_sample_latency_ms)
            records.append(
                {
                    "id": sample.get("frame_token"),
                    "question_type": sample.get("qa_type"),
                    "question": sample["question"],
                    "reference": sample["answer"],
                    "prediction": pred,
                    "selected_cameras": result.selected_cameras,
                    "latency_ms": round(per_sample_latency_ms, 2),
                }
            )
            global_index = batch_start + offset
            if global_index < 3:
                print(f"\n--- Example {global_index + 1} ---")
                print(f"Cameras: {', '.join(result.selected_cameras)}")
                print(f"Q: {sample['question']}")
                print(f"Ground Truth: {sample['answer']}")
                print(f"Prediction: {pred}")

        processed += len(batch)
        if processed % 10 == 0 or processed == len(test_samples) or processed < 10:
            print(f"Completed {processed}/{len(test_samples)} samples (batch latency {batch_latency_ms:.0f} ms)")

    results = compute_text_metrics(predictions, references)
    latency_summary = {
        "mean_ms": round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0,
        "min_ms": round(min(latencies_ms), 2) if latencies_ms else 0,
        "max_ms": round(max(latencies_ms), 2) if latencies_ms else 0,
    }
    output = {
        "model_id": args.model_id,
        "camera_mode": args.camera_mode,
        "num_samples": len(test_samples),
        "rouge": results,
        "latency": latency_summary,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\n--- Zero-Shot Benchmark Results (Qwen) ---")
    print(json.dumps({"rouge": results, "latency": latency_summary}, indent=2))
    print(f"Saved detailed results to {args.output_json}")

if __name__ == "__main__":
    main()
