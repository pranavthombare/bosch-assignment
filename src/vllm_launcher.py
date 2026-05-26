from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def vllm_executable() -> str:
    sibling = Path(sys.executable).with_name("vllm")
    if sibling.exists():
        return str(sibling)
    return shutil.which("vllm") or "vllm"


def maybe_add(argv: list[str], flag: str, value: str | None) -> None:
    if value:
        argv.extend([flag, value])


def lora_adapter_path() -> Path:
    return env_path("DRIVELM_LORA_PATH", PROJECT_ROOT / "models" / "qwen-lora")


def adapter_is_available(path: Path) -> bool:
    return path.exists() and (path / "adapter_config.json").exists()


def lora_module_json(model_id: str, adapter_path: Path) -> str:
    return json.dumps(
        {
            "name": os.getenv("DRIVELM_LORA_NAME", "drivelm-lora"),
            "path": str(adapter_path),
            "base_model_name": model_id,
        },
        separators=(",", ":"),
    )


def build_vllm_argv() -> list[str]:
    model_id = os.getenv("DRIVELM_MODEL_ID", DEFAULT_MODEL_ID)
    image_limit = {
        "image": {
            "count": int(os.getenv("DRIVELM_VLLM_IMAGE_COUNT", "6")),
            "width": int(os.getenv("DRIVELM_VLLM_IMAGE_WIDTH", "336")),
            "height": int(os.getenv("DRIVELM_VLLM_IMAGE_HEIGHT", "336")),
        }
    }

    argv = [
        vllm_executable(),
        "serve",
        model_id,
        "--host",
        os.getenv("DRIVELM_VLLM_HOST", "0.0.0.0"),
        "--port",
        os.getenv("DRIVELM_VLLM_PORT", "8001"),
        "--dtype",
        os.getenv("DRIVELM_VLLM_DTYPE", "float16"),
        "--max-model-len",
        os.getenv("DRIVELM_VLLM_MAX_MODEL_LEN", "1024"),
        "--gpu-memory-utilization",
        os.getenv("DRIVELM_VLLM_GPU_MEMORY_UTILIZATION", "0.60"),
        "--mm-processor-cache-gb",
        os.getenv("DRIVELM_VLLM_MM_PROCESSOR_CACHE_GB", "1"),
        "--max-num-seqs",
        os.getenv("DRIVELM_VLLM_MAX_NUM_SEQS", "4"),
        "--max-num-batched-tokens",
        os.getenv("DRIVELM_VLLM_MAX_NUM_BATCHED_TOKENS", "2048"),
        "--limit-mm-per-prompt",
        json.dumps(image_limit, separators=(",", ":")),
    ]

    maybe_add(argv, "--attention-backend", os.getenv("DRIVELM_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"))

    if env_bool("DRIVELM_VLLM_ENFORCE_EAGER", True):
        argv.append("--enforce-eager")
    if env_bool("DRIVELM_VLLM_SKIP_MM_PROFILING", True):
        argv.append("--skip-mm-profiling")

    adapter_path = lora_adapter_path()
    use_lora = env_bool("DRIVELM_ENABLE_LORA", adapter_is_available(adapter_path))
    if use_lora:
        if not adapter_is_available(adapter_path):
            raise FileNotFoundError(
                f"DRIVELM_ENABLE_LORA is true, but no LoRA adapter was found at {adapter_path}."
            )
        argv.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                os.getenv("DRIVELM_VLLM_MAX_LORA_RANK", "8"),
                "--lora-modules",
                lora_module_json(model_id, adapter_path),
            ]
        )

    return argv


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    argv = build_vllm_argv()
    print("Starting DriveLM vLLM server")
    print("Base model:", os.getenv("DRIVELM_MODEL_ID", DEFAULT_MODEL_ID))
    print("LoRA adapter:", os.getenv("DRIVELM_LORA_NAME", "drivelm-lora") if "--enable-lora" in argv else "disabled")
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
