from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_QWEN_MODEL_ID = os.getenv("DRIVELM_MODEL_ID", "Qwen/Qwen3.5-0.8B")

CAMERA_MODES = {
    "front": ["CAM_FRONT"],
    "front-arc": ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    "all": [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ],
}


@dataclass(frozen=True)
class GenerationResult:
    prediction: str
    selected_cameras: list[str]
    num_images: int


def resolve_device(requested_device: str | None = None) -> str:
    if requested_device and requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, requested_dtype: str | None = None) -> torch.dtype:
    if requested_dtype and requested_dtype != "auto":
        return getattr(torch, requested_dtype)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_qwen_processor(model_id: str = DEFAULT_QWEN_MODEL_ID):
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def load_qwen_model(
    model_id: str = DEFAULT_QWEN_MODEL_ID,
    device: str | None = None,
    dtype: str | None = None,
    quantization: str = "auto",
):
    resolved_device = resolve_device(device)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}

    if quantization == "auto":
        if resolved_device == "cuda":
            try:
                import bitsandbytes  # noqa: F401

                quantization = "4bit"
            except Exception:
                quantization = "none"
        else:
            quantization = "none"

    if quantization in {"4bit", "8bit"}:
        if resolved_device != "cuda":
            raise ValueError("4-bit/8-bit quantization requires a CUDA GPU and bitsandbytes.")
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["device_map"] = "auto"
        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["dtype"] = resolve_dtype(resolved_device, dtype)

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    if quantization == "none":
        model = model.to(resolved_device)
    return model, resolved_device


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if root is not None:
        return root / candidate
    return candidate


def select_image_paths(
    image_paths: dict[str, str],
    camera_mode: str,
    nuscenes_root: Path | None = None,
    composite_image_path: str | None = None,
) -> list[tuple[str, Path]]:
    if camera_mode == "mosaic" and composite_image_path:
        return [("MOSAIC", resolve_path(composite_image_path, nuscenes_root))]
    if camera_mode == "mosaic":
        camera_mode = "all"

    cameras = CAMERA_MODES.get(camera_mode)
    if cameras is None:
        raise ValueError(f"Unsupported camera mode: {camera_mode}. Choose one of front, front-arc, all, mosaic.")

    selected: list[tuple[str, Path]] = []
    for camera in cameras:
        if camera not in image_paths:
            continue
        path = resolve_path(image_paths[camera], nuscenes_root)
        if path.exists():
            selected.append((camera, path))
    return selected


def resize_to_long_edge(image: Image.Image, max_long_edge: int | None) -> Image.Image:
    if max_long_edge is None or max_long_edge <= 0:
        return image
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.BICUBIC)


@lru_cache(maxsize=128)
def _load_image_cached(path: str, mtime_ns: int, max_long_edge: int | None) -> Image.Image:
    del mtime_ns
    with Image.open(path) as image:
        return resize_to_long_edge(image.convert("RGB"), max_long_edge).copy()


def load_image(path: Path, use_cache: bool = True, max_long_edge: int | None = 448) -> Image.Image:
    if use_cache:
        stat = path.stat()
        return _load_image_cached(str(path), stat.st_mtime_ns, max_long_edge).copy()
    with Image.open(path) as image:
        return resize_to_long_edge(image.convert("RGB"), max_long_edge)


def build_qwen_messages(
    selected_images: list[tuple[str, Path]],
    question: str,
    answer: str | None = None,
    use_image_cache: bool = True,
    max_image_long_edge: int | None = 448,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are answering autonomous-driving visual questions from synchronized "
                "nuScenes camera views. Use the camera labels when spatial context matters."
            ),
        }
    ]
    for camera, path in selected_images:
        content.append({"type": "text", "text": f"Camera view: {camera}"})
        content.append(
            {
                "type": "image",
                "image": load_image(
                    path,
                    use_cache=use_image_cache,
                    max_long_edge=max_image_long_edge,
                ),
            }
        )
    content.append({"type": "text", "text": f"Question: {question}\nAnswer in one short sentence."})

    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def apply_qwen_chat_template(
    processor,
    messages: list[dict[str, Any]],
    device: str,
    add_generation_prompt: bool,
) -> dict[str, torch.Tensor]:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def generate_qwen_answer(
    model,
    processor,
    image_paths: dict[str, str],
    question: str,
    nuscenes_root: Path,
    camera_mode: str = "front-arc",
    composite_image_path: str | None = None,
    device: str = "cuda",
    max_new_tokens: int = 64,
    use_image_cache: bool = True,
    max_image_long_edge: int | None = 448,
) -> GenerationResult:
    selected_images = select_image_paths(
        image_paths=image_paths,
        camera_mode=camera_mode,
        nuscenes_root=nuscenes_root,
        composite_image_path=composite_image_path,
    )
    if not selected_images:
        raise FileNotFoundError("No usable image paths were found for this example.")

    messages = build_qwen_messages(
        selected_images,
        question,
        use_image_cache=use_image_cache,
        max_image_long_edge=max_image_long_edge,
    )
    inputs = apply_qwen_chat_template(
        processor,
        messages,
        device=device,
        add_generation_prompt=True,
    )

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_length = inputs["input_ids"].shape[-1]
    generated_ids = generated_ids[:, input_length:]
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return GenerationResult(
        prediction=prediction,
        selected_cameras=[camera for camera, _ in selected_images],
        num_images=len(selected_images),
    )


def generate_qwen_answers_batch(
    model,
    processor,
    samples: list[dict[str, Any]],
    nuscenes_root: Path,
    camera_mode: str = "front-arc",
    device: str = "cuda",
    max_new_tokens: int = 64,
    use_image_cache: bool = True,
    max_image_long_edge: int | None = 448,
) -> list[GenerationResult]:
    if not samples:
        return []

    selected_lists: list[list[tuple[str, Path]]] = []
    messages_batch: list[list[dict[str, Any]]] = []
    for sample in samples:
        selected = select_image_paths(
            image_paths=sample["image_paths"],
            camera_mode=camera_mode,
            nuscenes_root=nuscenes_root,
            composite_image_path=sample.get("composite_image_path"),
        )
        if not selected:
            raise FileNotFoundError("No usable image paths were found for a batch example.")
        selected_lists.append(selected)
        messages_batch.append(
            build_qwen_messages(
                selected,
                sample["question"],
                use_image_cache=use_image_cache,
                max_image_long_edge=max_image_long_edge,
            )
        )

    previous_padding_side = getattr(processor.tokenizer, "padding_side", "right")
    processor.tokenizer.padding_side = "left"
    try:
        inputs = processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    finally:
        processor.tokenizer.padding_side = previous_padding_side

    input_length = inputs["input_ids"].shape[-1]
    generated_only = generated_ids[:, input_length:]
    predictions = processor.batch_decode(generated_only, skip_special_tokens=True)

    return [
        GenerationResult(
            prediction=prediction.strip(),
            selected_cameras=[camera for camera, _ in selected],
            num_images=len(selected),
        )
        for prediction, selected in zip(predictions, selected_lists)
    ]


def freeze_vision_modules(model) -> list[str]:
    frozen: list[str] = []
    candidate_names = (
        "visual",
        "vision_model",
        "vision_tower",
        "vision_encoder",
        "image_encoder",
        "vpm",
    )
    for name in candidate_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = False
        frozen.append(name)
    return frozen


def common_lora_target_modules() -> list[str]:
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]