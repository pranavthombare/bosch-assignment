"""Typed runtime configuration sourced from environment variables.

Layering (lowest → highest precedence):

    code defaults  →  .env file (auto-loaded if present)  →  DRIVELM_* shell env

Env override naming: ``DRIVELM_<SECTION>__<FIELD>`` (two underscores between
section and field), e.g. ``DRIVELM_EVAL__NUM_SAMPLES=10``.

The effective config is printed at script start (``print_config``) and embedded
inside every artifact JSON (``config_dict``) so any historical run is
reproducible from its own output file.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_type_hints

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PREFIX = "DRIVELM_"


@dataclass(frozen=True)
class DataCfg:
    nuscenes_dir: Path = Path("data/nuscenes")
    drivelm_json: Path = Path("data/drivelm/v1_1_train_nus.json")
    camera_mode: str = "front-arc"
    image_long_edge: int = 448


@dataclass(frozen=True)
class ModelCfg:
    model_id: str = DEFAULT_QWEN_MODEL_ID
    lora_model_id: str = "drivelm-lora"
    quantization: str = "auto"
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 64


@dataclass(frozen=True)
class TrainCfg:
    num_samples: int = 1024
    epochs: int = 1
    lr: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    gradient_accumulation_steps: int = 2
    checkpoint_every: int = 200
    output_dir: Path = Path("models/qwen-lora")
    stratified: bool = False
    stratified_seed: int = 42
    sampling: str = "natural"  # natural | stratified | proportional


@dataclass(frozen=True)
class EvalCfg:
    base_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = "local"
    num_samples: int = 0
    temperature: float = 0.0
    concurrency: int = 8
    timeout: int = 180
    max_retries: int = 3
    run_base: bool = True
    run_lora: bool = True
    output_base_json: Path = Path("artifacts/baseline_front_arc_full.json")
    output_lora_json: Path = Path("artifacts/finetuned_front_arc_full.json")


@dataclass(frozen=True)
class RunCfg:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)


def _coerce(value: Any, target_type: Any) -> Any:
    if isinstance(value, target_type) and target_type is not bool:
        return value
    if target_type is Path:
        return Path(str(value)).expanduser()
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return value


def _apply_overrides(cfg: Any, overrides: dict[str, Any]) -> Any:
    if not overrides:
        return cfg
    type_hints = get_type_hints(type(cfg))
    new_values: dict[str, Any] = {}
    for f in fields(cfg):
        current = getattr(cfg, f.name)
        if is_dataclass(current):
            sub = overrides.get(f.name, {})
            if not isinstance(sub, dict):
                raise TypeError(f"expected dict for {f.name}, got {type(sub).__name__}")
            new_values[f.name] = _apply_overrides(current, sub)
        elif f.name in overrides:
            new_values[f.name] = _coerce(overrides[f.name], type_hints.get(f.name, type(current)))
    return replace(cfg, **new_values) if new_values else cfg


def _env_overrides(cfg: Any, prefix: str = ENV_PREFIX) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(cfg):
        current = getattr(cfg, f.name)
        if is_dataclass(current):
            sub_prefix = f"{prefix}{f.name.upper()}__"
            sub = _env_overrides(current, sub_prefix)
            if sub:
                out[f.name] = sub
        else:
            env_key = f"{prefix}{f.name.upper()}"
            if env_key in os.environ:
                out[f.name] = os.environ[env_key]
    return out


def load_config() -> RunCfg:
    """Return a RunCfg with `.env` and `DRIVELM_*` env vars applied to the defaults."""
    return _apply_overrides(RunCfg(), _env_overrides(RunCfg()))


def config_dict(cfg: Any) -> dict[str, Any]:
    """Serializable snapshot for embedding in artifact JSONs."""

    def _convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    return _convert(asdict(cfg))


def print_config(cfg: RunCfg) -> None:
    import json as _json

    print("Effective configuration:")
    print(_json.dumps(config_dict(cfg), indent=2))
