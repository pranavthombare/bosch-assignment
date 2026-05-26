import argparse
import sys
from pathlib import Path
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.pipeline import DriveLMDataset
from src.eval.benchmark import first_valid_samples
from src.serve.qwen import (
    DEFAULT_QWEN_MODEL_ID,
    apply_qwen_chat_template,
    build_qwen_messages,
    common_lora_target_modules,
    freeze_vision_modules,
    load_qwen_model,
    load_qwen_processor,
    select_image_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen on DriveLM with LoRA/QLoRA.")
    parser.add_argument("--model-id", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--nuscenes-dir", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--drivelm-json", type=Path, default=Path("data/drivelm/v1_1_train_nus.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/qwen-lora"))
    parser.add_argument("--camera-mode", choices=["front", "front-arc", "all", "mosaic"], default="front-arc")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", choices=["auto", "none", "4bit", "8bit"], default="auto")
    parser.add_argument("--image-long-edge", type=int, default=448)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=200, help="Save adapter every N steps. 0 disables.")
    return parser.parse_args()


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

def main():
    args = parse_args()
    print(f"Initializing Qwen LoRA training: {args.model_id}")
    processor = load_qwen_processor(args.model_id)
    model, device = load_qwen_model(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
    )
    if args.quantization in {"auto", "4bit", "8bit"} and device == "cuda":
        model = prepare_model_for_kbit_training(model)
    
    frozen_modules = freeze_vision_modules(model)
    if frozen_modules:
        print(f"Frozen vision modules: {', '.join(frozen_modules)}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=common_lora_target_modules(),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Loading DriveLM Dataset...")
    dataset = DriveLMDataset(
        str(args.nuscenes_dir),
        str(args.drivelm_json),
    )
    train_samples = first_valid_samples(dataset, args.num_samples, args.nuscenes_dir)
    print(f"Training on {len(train_samples)} samples...")
    
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        for step, sample in enumerate(train_samples, start=1):
            inputs = build_supervised_inputs(
                sample,
                processor,
                device,
                args.nuscenes_dir,
                args.camera_mode,
                args.image_long_edge,
            )
            outputs = model(**inputs)
            loss = outputs.loss / args.gradient_accumulation_steps
            
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_samples):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            print(
                f"  step {step}/{len(train_samples)} "
                f"loss={loss.item() * args.gradient_accumulation_steps:.4f}"
            )

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0 and step < len(train_samples):
                print(f"  checkpointing at step {step} to {args.output_dir}")
                args.output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(args.output_dir))

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / max(len(train_samples), 1):.4f}")
        
    print("Saving Qwen LoRA adapters...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
