import torch
import sys
sys.path.append("..")
from transformers import AutoModelForCausalLM
from data.dpo_ultrafeedback_data_pipeline import get_dataloader, tokenizer
from tqdm import tqdm
from utils import sequence_log_prob
import argparse

def finetune_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_model_path = '../checkpoints/preference_sft_20250520-041117/step_55000'
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, trust_remote_code=True)
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.to(device)
    ref_model.eval()
    train_dataloader = get_dataloader(split="train_prefs", batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    global_step = 0
    confidences = []
    prompt_ids = []
    pbar = tqdm(train_dataloader)
    for batch in pbar:
        prompt_ids.append(batch['prompt_id'][0])
        batch = {k: v.to(device) for k, v in batch.items() if k != "prompt_id"}
        ref_log_prob_chosen = sequence_log_prob(ref_model, batch['input_ids_chosen'], batch['attention_mask_chosen'], batch['labels_chosen'])
        ref_log_prob_rejected = sequence_log_prob(ref_model, batch['input_ids_rejected'], batch['attention_mask_rejected'], batch['labels_rejected'])
        ref_confidence = ref_log_prob_chosen - ref_log_prob_rejected
        confidences.append(ref_confidence.item())
        pbar.set_postfix({"confidence": ref_confidence.item()})
        global_step += 1
        if global_step == 1:
            print(prompt_ids, confidences)
        output_dir = f"../output/confidences_ref.txt"
        with open(output_dir, 'w') as f:
            for prompt_id, confidence in zip(prompt_ids, confidences):
                f.write(f"{prompt_id}\t{confidence}\n")
    print(f"Confidences saved to {output_dir}")
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with DPO data.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training and evaluation.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    finetune_model()
