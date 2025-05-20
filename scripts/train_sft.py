import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def evaluate(model, dataloader, device, global_step):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].size(0)
            count += batch["input_ids"].size(0)
    avg_loss = total_loss / count
    if dist.get_rank() == 0:
        writer.add_scalar("Loss/eval", avg_loss, global_step)
    model.train()
    return avg_loss


def finetune_model():
    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    train_dataloader = get_dataloader(split="train", batch_size=4, shuffle=True, num_workers=args.num_workers, debug_mode=args.debug_mode)
    eval_dataloader = get_dataloader(split="test", batch_size=4, shuffle=False, num_workers=args.num_workers, debug_mode=args.debug_mode)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    num_epochs = args.num_epochs
    global_step = 0
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=local_rank != 0)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss.mean()
            if loss.isnan().any():
                print("Loss is NaN, global step:", global_step)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if local_rank == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                pbar.set_postfix({"loss": loss.item()})
            global_step += 1

        if local_rank == 0:
            eval_loss = evaluate(model, eval_dataloader, device, global_step)
            print(f"Epoch {epoch+1} Eval Loss: {eval_loss:.4f}")

    if local_rank == 0:
        model.module.save_pretrained("./preference_sft")
        tokenizer.save_pretrained("./preference_sft")
        writer.close()

    cleanup()

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with SFT data.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading.")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode for small dataset.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    finetune_model()
