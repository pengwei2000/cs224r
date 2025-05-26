import torch
from torch import nn, optim
import sys
sys.path.append("..")
from transformers import AutoModelForCausalLM
from data.sft_data_pipeline import get_dataloader, tokenizer, MODEL_NAME
from tqdm import tqdm
import os
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse

run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

writer = SummaryWriter(log_dir=os.path.join("../output", "preference_sft", run_name))

def evaluate(model, dataloader, device, global_step):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i > 1000:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].size(0)
            count += batch["input_ids"].size(0)
    avg_loss = total_loss / count
    writer.add_scalar("Loss/eval", avg_loss, global_step)
    model.train()
    return avg_loss


def finetune_model():
    checkpoint_dir = f"../checkpoints/preference_sft_{args.save_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(checkpoint_dir, args.resume_from) if args.resume_from else MODEL_NAME
    if args.lora:
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        base_model.config.pad_token_id = tokenizer.pad_token_id
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.train()

    train_dataloader = get_dataloader(split="train", batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, debug_mode=args.debug_mode, max_length=args.max_length)
    eval_dataloader = get_dataloader(split="test", batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, debug_mode=args.debug_mode, max_length=args.max_length)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.GradScaler()
    num_epochs = args.num_epochs
    global_step = 0
    if args.resume_from and args.resume_from.startswith("step_"):
        try:
            global_step = int(args.resume_from.replace("step_", "").split("_")[0])
            print(f"Resuming training from global_step = {global_step}")
        except ValueError:
            print(f"Warning: Could not parse global_step from resume_from = {args.resume_from}")
    early_stop_patience = 20
    best_eval_loss = float("inf")
    early_stop_counter = 0
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
            if loss.isnan().any():
                print("Loss is NaN, global step:", global_step)
            scaler.scale(loss).backward()
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            pbar.set_postfix({"loss": loss.item()})
            if global_step % 1000 == 0:
                eval_loss = evaluate(model, eval_dataloader, device, global_step)
                print(f"Step {global_step} Eval Loss: {eval_loss:.4f}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stop_counter = 0

                    # Save checkpoint
                    model.save_pretrained(os.path.join(checkpoint_dir, f"step_{global_step}"))
                    # tokenizer.save_pretrained(os.path.join(checkpoint_dir, f"step_{global_step}"))
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print("Early stopping triggered.")
                        break

            global_step += 1

    model.save_pretrained(os.path.join(checkpoint_dir, f"step_{global_step}"))
    # tokenizer.save_pretrained(os.path.join(checkpoint_dir, f"step_{global_step}"))
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with SFT data.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading.")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode for small dataset.")
    parser.add_argument("--save_name", type=str, default=f'{run_name}', help="Name of the model to save.")
    parser.add_argument("--max_length", type=int, default=576, help="Maximum length of the input sequences.")
    parser.add_argument("--lora", action="store_true", help="Use LoRA for training.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to resume training from a checkpoint.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    finetune_model()
