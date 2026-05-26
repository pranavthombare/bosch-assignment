import argparse
import json
import time
import sys
from pathlib import Path
from peft import PeftModel

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.pipeline import DriveLMDataset
from src.eval.benchmark import compute_text_metrics, first_valid_samples
from src.serve.qwen import DEFAULT_QWEN_MODEL_ID, generate_qwen_answer, load_qwen_model, load_qwen_processor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen LoRA adapters on DriveLM.")
    parser.add_argument("--model-id", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--lora-path", type=Path, default=Path("models/qwen-lora"))
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
    parser.add_argument("--image-long-edge", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/qwen_finetuned_results.json"))
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading base Qwen model: {args.model_id}")
    processor = load_qwen_processor(args.model_id)
    base_model, device = load_qwen_model(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
    )

    print("Injecting Fine-Tuned LoRA Adapters...")
    if args.lora_path.exists():
        model = PeftModel.from_pretrained(
            base_model,
            str(args.lora_path),
            trust_remote_code=True
        ).eval()
    else:
        print(f"LoRA path {args.lora_path} not found. Run finetune.py first.")
        return

    print("Loading DriveLM Dataset for Evaluation...")
    dataset = DriveLMDataset(str(args.nuscenes_dir), str(args.drivelm_json))
    test_samples = first_valid_samples(dataset, args.num_samples, args.nuscenes_dir)
    
    predictions = []
    references = []
    records = []
    latencies_ms = []
    
    for index, sample in enumerate(test_samples):
        start = time.perf_counter()
        result = generate_qwen_answer(
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
        latency_ms = (time.perf_counter() - start) * 1000
        predictions.append(result.prediction)
        references.append(sample["answer"])
        latencies_ms.append(latency_ms)
        records.append(
            {
                "id": sample.get("frame_token"),
                "question_type": sample.get("qa_type"),
                "question": sample["question"],
                "reference": sample["answer"],
                "prediction": result.prediction,
                "selected_cameras": result.selected_cameras,
                "latency_ms": round(latency_ms, 2),
            }
        )
        if index < 3:
            print(f"\n--- Finetuned Example {index + 1} ---")
            print(f"Cameras: {', '.join(result.selected_cameras)}")
            print(f"Q: {sample['question']}")
            print(f"Ground Truth: {sample['answer']}")
            print(f"Prediction: {result.prediction}")
        if (index + 1) % 10 == 0 or index + 1 == len(test_samples):
            print(f"Completed {index + 1}/{len(test_samples)} samples")

    results = compute_text_metrics(predictions, references)
    latency_summary = {
        "mean_ms": round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0,
        "min_ms": round(min(latencies_ms), 2) if latencies_ms else 0,
        "max_ms": round(max(latencies_ms), 2) if latencies_ms else 0,
    }
    output = {
        "model_id": args.model_id,
        "lora_path": str(args.lora_path),
        "camera_mode": args.camera_mode,
        "num_samples": len(test_samples),
        "rouge": results,
        "latency": latency_summary,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\n--- Fine-Tuned Benchmark Results (Qwen + LoRA) ---")
    print(json.dumps({"rouge": results, "latency": latency_summary}, indent=2))
    print(f"Saved detailed results to {args.output_json}")

if __name__ == "__main__":
    main()
