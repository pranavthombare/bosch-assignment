from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.pipeline import DriveLMDataset
from src.eval.benchmark import compute_text_metrics, first_valid_samples
from src.serve.qwen import DEFAULT_QWEN_MODEL_ID, load_image, select_image_paths


SYSTEM_TEXT = (
    "You are answering autonomous-driving visual questions from synchronized "
    "nuScenes camera views. Use the camera labels when spatial context matters."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen through an OpenAI-compatible chat-completions endpoint.")
    parser.add_argument("--model-id", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--nuscenes-dir", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--drivelm-json", type=Path, default=Path("data/drivelm/v1_1_train_nus.json"))
    parser.add_argument("--num-samples", type=int, default=0, help="0 = all usable.")
    parser.add_argument("--camera-mode", choices=["front", "front-arc", "all", "mosaic"], default="front-arc")
    parser.add_argument("--image-long-edge", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/qwen_endpoint_results.json"))
    return parser.parse_args()


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


async def call_chat(client: AsyncOpenAI, args, messages) -> str:
    response = await client.chat.completions.create(
        model=args.model_id,
        messages=messages,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    return response.choices[0].message.content.strip()


async def run(args: argparse.Namespace) -> None:
    image_long_edge = args.image_long_edge if args.image_long_edge > 0 else None

    print("Loading DriveLM Dataset...")
    dataset = DriveLMDataset(str(args.nuscenes_dir), str(args.drivelm_json))
    test_samples = first_valid_samples(dataset, args.num_samples, args.nuscenes_dir)
    print(f"Running endpoint benchmark on {len(test_samples)} samples (concurrency={args.concurrency})...")

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    predictions: list[str | None] = [None] * len(test_samples)
    latencies_ms: list[float] = [0.0] * len(test_samples)
    selected_cams: list[list[str]] = [[] for _ in test_samples]
    errors: list[str | None] = [None] * len(test_samples)

    async def process(index: int, sample: dict) -> None:
        try:
            selected = select_image_paths(
                image_paths=sample["image_paths"],
                camera_mode=args.camera_mode,
                nuscenes_root=args.nuscenes_dir,
            )
            if not selected:
                raise RuntimeError("No usable image paths.")
            selected_cams[index] = [camera for camera, _ in selected]

            async with semaphore:
                messages = build_chat_messages(selected, sample["question"], image_long_edge)
                start = time.perf_counter()
                prediction = await call_chat(client, args, messages)
                latency_ms = (time.perf_counter() - start) * 1000

            predictions[index] = prediction
            latencies_ms[index] = latency_ms
        except Exception as exc:
            errors[index] = f"{type(exc).__name__}: {exc}"

    tasks = [process(i, s) for i, s in enumerate(test_samples)]
    completed = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        completed += 1
        if completed % 50 == 0 or completed == len(tasks):
            print(f"Completed {completed}/{len(tasks)}", flush=True)

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
        "model_id": args.model_id,
        "base_url": args.base_url,
        "camera_mode": args.camera_mode,
        "concurrency": args.concurrency,
        "num_samples": len(test_samples),
        "num_succeeded": len(finals_pred),
        "num_failed": sum(1 for e in errors if e),
        "rouge": metrics,
        "latency": latency_summary,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\n--- Endpoint Benchmark Results ---")
    print(json.dumps({"rouge": metrics, "latency": latency_summary, "failed": output["num_failed"]}, indent=2))
    print(f"Saved detailed results to {args.output_json}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
