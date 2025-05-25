import torch
from torch import nn, optim
import sys
sys.path.append("..")
from transformers import AutoModelForCausalLM
from data.rloo_data_verifier import get_dataloader, tokenizer, MODEL_NAME
from tqdm import tqdm
import os
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from utils import compute_score, centered_rewards

run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

writer = SummaryWriter(log_dir=os.path.join("../output", "verifier_rloo", run_name))

def evaluate(model, dataloader, device, global_step):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].size(0)
            count += batch["input_ids"].size(0)
    avg_loss = total_loss / count
    writer.add_scalar("Loss/eval", avg_loss, global_step)
    model.train()
    return avg_loss

def compute_weighted_loss(model, batch, tokenized_batch_result, rewards, device):

    all_losses = []
    for attention_mask, responses, reward_list in zip(batch['attention_mask'], tokenized_batch_result, rewards):
        if sum(reward_list) == 0:
            continue

        weights = torch.tensor(reward_list, device=device, dtype=torch.float)
        weights = centered_rewards(weights)

        losses = []
        for resp in responses:
            labels = resp.clone()
            assert len(labels.shape) == 1, "Input IDs should be a single sequence."
            # get input_ids where attention_mask is 1
            prompt_len = attention_mask.sum().item()
            labels[:prompt_len] = -100  # mask prompt portion
            new_attention_mask = torch.ones_like(resp, dtype=torch.long)
            # if the token id is 151643, then attention_mask should be 0
            new_attention_mask[resp == tokenizer.pad_token_id] = 0
            labels[new_attention_mask == 0] = -100  # mask padding tokens
            outputs = model(input_ids=resp.unsqueeze(0), attention_mask=new_attention_mask.unsqueeze(0), labels=labels.unsqueeze(0))
            losses.append(outputs.loss)

        loss_tensor = torch.stack(losses)
        weighted_loss = torch.dot(loss_tensor, weights)
        all_losses.append(weighted_loss)

    return torch.stack(all_losses).mean() if all_losses else torch.tensor(0.0, requires_grad=True).to(device)

def finetune_model():
    checkpoint_dir = f"../checkpoints/verifier_rloo_{args.save_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(checkpoint_dir, args.resume_from) if args.resume_from else '../checkpoints/verifier_sft_20250521-171405/step_1000'
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
    # eval_dataloader = get_dataloader(split="test", batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, debug_mode=args.debug_mode, max_length=args.max_length)
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
    # early_stop_patience = 20
    # best_eval_loss = float("inf")
    # early_stop_counter = 0
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            for k, v in batch.items():
                if k != "numbers" and k != "target":
                    batch[k] = v.to(device)
            # batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                with torch.no_grad():
                    tokenized_batch_result = []
                    for i in range(len(batch["input_ids"])):
                        # result is a 2d array tensor
                        result = model.generate(input_ids=batch["input_ids"][i].unsqueeze(0), attention_mask = batch['attention_mask'][i].unsqueeze(0), max_new_tokens=args.max_length, do_sample=True, num_return_sequences=args.num_return_sequence, pad_token_id = tokenizer.pad_token_id)
                        tokenized_batch_result.append(result)
                        # after batch decode, the return is a list of strings
                    completions = [tokenizer.batch_decode(r, add_special_tokens=False, truncation=True, max_length=args.max_length, padding=False) for r in tokenized_batch_result]  # a nested list
                    rewards = [[compute_score(r, t, n) for r in rs] for rs, t, n in zip(completions, batch['target'], batch['numbers'])]
                    # print(rewards)
                loss = compute_weighted_loss(model, batch, tokenized_batch_result, rewards, device)
            if loss.isnan().any():
                print("Loss is NaN, global step:", global_step)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar('rewards/mean', torch.tensor(rewards).float().mean(), global_step)
            pbar.set_postfix({"loss": loss.item(), "reward": torch.tensor(rewards).float().mean().item()})
            if global_step % 1000 == 0:
            #     eval_loss = evaluate(model, eval_dataloader, device, global_step)
            #     print(f"Step {global_step} Eval Loss: {eval_loss:.4f}")

            #     if eval_loss < best_eval_loss:
            #         best_eval_loss = eval_loss
            #         early_stop_counter = 0

            #   Save checkpoint
                model.save_pretrained(os.path.join(checkpoint_dir, f"step_{global_step}"))
            #     else:
            #         early_stop_counter += 1
            #         if early_stop_counter >= early_stop_patience:
            #             print("Early stopping triggered.")
            #             break
            global_step += 1

    model.save_pretrained(os.path.join(checkpoint_dir, f"step_{global_step}"))
    # tokenizer.save_pretrained(os.path.join(checkpoint_dir, f"step_{global_step}"))
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with rloo data.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading.")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode for small dataset.")
    parser.add_argument("--save_name", type=str, default=f'{run_name}', help="Name of the model to save.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of the input sequences.")
    parser.add_argument("--lora", action="store_true", help="Use LoRA for training.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to resume training from a checkpoint.")
    parser.add_argument("--num_return_sequence", type=int, default=2, help="Number of sequences to return during generation.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    finetune_model()
