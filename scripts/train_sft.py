import torch
from torch import nn, optim
import sys
sys.path.append("..")
from transformers import AutoModelForCausalLM
from data.sft_data_pipeline import get_dataloader, tokenizer, MODEL_NAME
from tqdm import tqdm
import os
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
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            total_loss += outputs.loss.item() * batch["input_ids"].size(0)
            count += batch["input_ids"].size(0)
    avg_loss = total_loss / count
    writer.add_scalar("Loss/eval", avg_loss, global_step)
    model.train()
    return avg_loss

def finetune_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device)
    model.train()

    train_dataloader = get_dataloader(split="train", batch_size=args.batch_size, num_workers=4)
    eval_dataloader = get_dataloader(split="test", batch_size=args.batch_size, shuffle=False, num_workers=4)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    global_step = 0
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1

            pbar.set_postfix({"loss": loss.item()})

            # Run evaluation after each epoch
        eval_loss = evaluate(model, eval_dataloader, device, global_step)
        print(f"Evaluation loss after epoch {epoch+1}: {eval_loss:.4f}")

    output_dir = "./qwen2.5-finetuned"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    writer.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with SFT data.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    finetune_model()
