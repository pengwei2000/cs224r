import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Dict

# Constants
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "HuggingFaceTB/smol-smoltalk"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Custom dataset class
class SmolTalkDataset(Dataset):
    def __init__(self, split: str = "train", max_length: int = 576):
        self.dataset = load_dataset(DATASET_NAME, split=split)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]['messages']
        # Find the first assistant response and use previous messages as prompt
        response = None
        prompt = None
        for msg in sample:
            if msg["role"] == "assistant":
                response = msg["content"]
            elif msg["role"] == "user":
                prompt = msg["content"]
            if response is not None and prompt is not None:
                break
        full_text = prompt + response

        # Tokenize the prompt and full text
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_tokens = tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False)["input_ids"]
        labels = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]
        labels = labels[:self.max_length]  # Truncate labels to max_length
        # print(torch.tensor(labels, dtype=torch.long).shape, torch.tensor(full_tokens, dtype=torch.long).shape)
        return {
            "input_ids": torch.tensor(full_tokens, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(full_tokens), dtype=torch.long),   # ensure the information is not flowed from pad
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# Collate function with dynamic padding
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Convert batch of dicts to dict of lists
    batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
    # Pad each field
    input_ids = torch.nn.utils.rnn.pad_sequence(batch_dict["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)   # batch_first if True, the output will be in B x T x * format, T x B x * otherwise.
    attention_mask = torch.nn.utils.rnn.pad_sequence(batch_dict["attention_mask"], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(batch_dict["labels"], batch_first=True, padding_value=-100)   # -100 is the ignore index for loss calculation
    # print(labels.shape, input_ids.shape, attention_mask.shape)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# Function to get DataLoader
def get_dataloader(split="train", batch_size=64, shuffle=True, num_workers=4):
    dataset = SmolTalkDataset(split=split)
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
