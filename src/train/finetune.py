import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RunCfg, load_config, print_config
from src.data.pipeline import DriveLMDataset, proportional_samples, stratified_samples
from src.eval.eval import first_valid_samples
from src.qwen import (
    apply_qwen_chat_template,
    build_qwen_messages,
    common_lora_target_modules,
    freeze_vision_modules,
    load_qwen_model,
    load_qwen_processor,
    select_image_paths,
)


def build_supervised_inputs(
    sample,
    processor,
    device: str,
    nuscenes_dir: Path,
    camera_mode: str,
    image_long_edge: int,
):
    selected_images = select_image_paths(
        image_paths=sample["image_paths"],
        camera_mode=camera_mode,
        nuscenes_root=nuscenes_dir,
    )
    if not selected_images:
        raise FileNotFoundError("No usable image paths were found for this training example.")

    full_messages = build_qwen_messages(
        selected_images=selected_images,
        question=sample["question"],
        answer=sample["answer"],
        use_image_cache=False,
        max_image_long_edge=image_long_edge,
    )
    prompt_messages = build_qwen_messages(
        selected_images=selected_images,
        question=sample["question"],
        answer=None,
        use_image_cache=False,
        max_image_long_edge=image_long_edge,
    )

    inputs = apply_qwen_chat_template(
        processor=processor,
        messages=full_messages,
        device=device,
        add_generation_prompt=False,
    )
    prompt_inputs = apply_qwen_chat_template(
        processor=processor,
        messages=prompt_messages,
        device=device,
        add_generation_prompt=True,
    )
    prompt_length = prompt_inputs["input_ids"].shape[-1]

    labels = inputs["input_ids"].clone()
    labels[:, :prompt_length] = -100
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


def run(cfg: RunCfg) -> None:
    print(f"Initializing Qwen LoRA training: {cfg.model.model_id}")
    processor = load_qwen_processor(cfg.model.model_id)
    model, device = load_qwen_model(
        model_id=cfg.model.model_id,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
        quantization=cfg.model.quantization,
    )
    if cfg.model.quantization in {"auto", "4bit", "8bit"} and device == "cuda":
        model = prepare_model_for_kbit_training(model)

    frozen_modules = freeze_vision_modules(model)
    if frozen_modules:
        print(f"Frozen vision modules: {', '.join(frozen_modules)}")

    lora_config = LoraConfig(
        r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        target_modules=common_lora_target_modules(),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading DriveLM Dataset...")
    dataset = DriveLMDataset(str(cfg.data.nuscenes_dir), str(cfg.data.drivelm_json))
    sampling = "stratified" if cfg.train.stratified else cfg.train.sampling
    if sampling == "stratified":
        train_samples = stratified_samples(dataset, cfg.data.nuscenes_dir, seed=cfg.train.stratified_seed)
        print(f"Training on {len(train_samples)} stratified samples (seed={cfg.train.stratified_seed})...")
    elif sampling == "proportional":
        train_samples = proportional_samples(dataset, cfg.data.nuscenes_dir, seed=cfg.train.stratified_seed)
        print(f"Training on {len(train_samples)} proportionally-sampled samples (seed={cfg.train.stratified_seed})...")
    else:
        train_samples = first_valid_samples(dataset, cfg.train.num_samples, cfg.data.nuscenes_dir)
        print(f"Training on {len(train_samples)} natural-distribution samples...")

    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.train.lr)
    global_step = 0

    for epoch in range(cfg.train.epochs):
        epoch_loss = 0.0
        print(f"Epoch {epoch + 1}/{cfg.train.epochs}")

        for step, sample in enumerate(train_samples, start=1):
            inputs = build_supervised_inputs(
                sample,
                processor,
                device,
                cfg.data.nuscenes_dir,
                cfg.data.camera_mode,
                cfg.data.image_long_edge,
            )
            outputs = model(**inputs)
            loss = outputs.loss / cfg.train.gradient_accumulation_steps

            loss.backward()
            if step % cfg.train.gradient_accumulation_steps == 0 or step == len(train_samples):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * cfg.train.gradient_accumulation_steps
            print(
                f"  step {step}/{len(train_samples)} "
                f"loss={loss.item() * cfg.train.gradient_accumulation_steps:.4f}"
            )

            if (
                cfg.train.checkpoint_every > 0
                and step % cfg.train.checkpoint_every == 0
                and step < len(train_samples)
            ):
                print(f"  checkpointing at step {step} to {cfg.train.output_dir}")
                cfg.train.output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(cfg.train.output_dir))

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / max(len(train_samples), 1):.4f}")

    print("Saving Qwen LoRA adapters...")
    cfg.train.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.train.output_dir))
    processor.save_pretrained(str(cfg.train.output_dir))
    print(f"Saved to {cfg.train.output_dir}")


def main() -> None:
    cfg = load_config()
    print_config(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
