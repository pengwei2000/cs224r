import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Dict

# Constants
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Custom dataset class
class UltraFeedbackDataset(Dataset):
    def __init__(self, split: str = "train_prefs", debug_mode: bool = False):
        self.dataset = load_dataset(DATASET_NAME, split=split)
        if debug_mode:
            self.dataset = self.dataset.select(range(2000))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        chosen = self.dataset[idx]['chosen']
        rejected = self.dataset[idx]['rejected']
        # Find the first assistant response and use previous messages as prompt
        response = None
        prompt = self.dataset[idx]['prompt']
        assert len(chosen) == 2, f"Expected 2 chosen responses, got {len(chosen)}"
        assert len(rejected) == 2, f"Expected 2 rejected responses, got {len(rejected)}"
        chosen_response = chosen[1]['content']
        rejected_response = rejected[1]['content']
        assert chosen[1]['role'] == "assistant", f"Expected assistant role, got {chosen[1]['role']}"
        assert rejected[1]['role'] == "assistant", f"Expected assistant role, got {rejected[1]['role']}"

        # Tokenize the prompt and full text
        prompt_tokens = tokenizer(prompt, truncation=True, max_length=968, add_special_tokens=False)["input_ids"]
        chosen_tokens = tokenizer(chosen_response, truncation=True, max_length=989-len(prompt_tokens), padding=False)["input_ids"]
        rejected_tokens = tokenizer(rejected_response, truncation=True, max_length=989 - len(prompt_tokens), padding=False)["input_ids"]

        labels_chosen = [-100] * len(prompt_tokens) + chosen_tokens
        labels_rejected = [-100] * len(prompt_tokens) + rejected_tokens

        return {
            "input_ids_chosen": torch.tensor(prompt_tokens+chosen_tokens, dtype=torch.long),
            "attention_mask_chosen": torch.tensor([1] * len(prompt_tokens+chosen_tokens), dtype=torch.long),   # ensure the information is not flowed from pad
            "labels_chosen": torch.tensor(labels_chosen, dtype=torch.long),
            "input_ids_rejected": torch.tensor(prompt_tokens+rejected_tokens, dtype=torch.long),
            "attention_mask_rejected": torch.tensor([1] * len(prompt_tokens+rejected_tokens), dtype=torch.long),   # ensure the information is not flowed from pad
            "labels_rejected": torch.tensor(labels_rejected, dtype=torch.long),
        }

# Collate function with dynamic padding
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Convert batch of dicts to dict of lists
    batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
    # Pad each field
    input_ids_chosen = torch.nn.utils.rnn.pad_sequence(batch_dict["input_ids_chosen"], batch_first=True, padding_value=tokenizer.pad_token_id)   # batch_first if True, the output will be in B x T x * format, T x B x * otherwise.
    attention_mask_chosen = torch.nn.utils.rnn.pad_sequence(batch_dict["attention_mask_chosen"], batch_first=True, padding_value=0)
    labels_chosen = torch.nn.utils.rnn.pad_sequence(batch_dict["labels_chosen"], batch_first=True, padding_value=-100)   # -100 is the ignore index for loss calculation
    input_ids_rejected = torch.nn.utils.rnn.pad_sequence(batch_dict["input_ids_rejected"], batch_first=True, padding_value=tokenizer.pad_token_id)   # batch_first if True, the output will be in B x T x * format, T x B x * otherwise.
    attention_mask_rejected = torch.nn.utils.rnn.pad_sequence(batch_dict["attention_mask_rejected"], batch_first=True, padding_value=0)
    labels_rejected = torch.nn.utils.rnn.pad_sequence(batch_dict["labels_rejected"], batch_first=True, padding_value=-100)   # -100 is the ignore index for loss calculation

    return {
        "input_ids_chosen": input_ids_chosen,
        "attention_mask_chosen": attention_mask_chosen,
        "labels_chosen": labels_chosen,
        "input_ids_rejected": input_ids_rejected,
        "attention_mask_rejected": attention_mask_rejected,
        "labels_rejected": labels_rejected,
    }

# Function to get DataLoader
def get_dataloader(split="train_prefs", batch_size=64, shuffle=True, num_workers=4, debug_mode=False):
    dataset = UltraFeedbackDataset(split=split, debug_mode=debug_mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return dataloader

if __name__ == "__main__":
    dl = get_dataloader()
    for batch in dl:
        print(batch)
        break
