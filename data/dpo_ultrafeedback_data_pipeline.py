import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Dict
import json
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
        self.split = split
        # load json file as the ref_generation
        with open("C:\\All files\\Stanford\\PhD Courses\\cs224r\\project\\data\\ref_outputs_trainset.json", "r") as f:
            self.ref_generation = json.load(f)  # a list, the idx is aligned with the original dataset
        if debug_mode:
            self.dataset = self.dataset.select(range(2000))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        chosen = self.dataset[idx]['chosen']
        rejected = self.dataset[idx]['rejected']
        # Find the first assistant response and use previous messages as prompt
        if self.split == "train_prefs":
            ref_response = self.ref_generation[idx]['response']
            ref_prompt_id = self.ref_generation[idx]['prompt_id']
            assert ref_prompt_id == self.dataset[idx]['prompt_id'], f"Prompt ID mismatch: {ref_prompt_id} != {self.dataset[idx]['prompt_id']}"

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
        if self.split == "train_prefs":
            ref_gen_tokens = tokenizer(ref_response, truncation=True, max_length=989 - len(prompt_tokens), padding=False)["input_ids"]
        labels_chosen = [-100] * len(prompt_tokens) + chosen_tokens
        labels_rejected = [-100] * len(prompt_tokens) + rejected_tokens
        labels_gen = [-100] * len(prompt_tokens) + ref_gen_tokens
        if self.split != "train_prefs":
            return {
            "input_ids_chosen": torch.tensor(prompt_tokens+chosen_tokens, dtype=torch.long),
            "attention_mask_chosen": torch.tensor([1] * len(prompt_tokens+chosen_tokens), dtype=torch.long),   # ensure the information is not flowed from pad
            "labels_chosen": torch.tensor(labels_chosen, dtype=torch.long),
            "input_ids_rejected": torch.tensor(prompt_tokens+rejected_tokens, dtype=torch.long),
            "attention_mask_rejected": torch.tensor([1] * len(prompt_tokens+rejected_tokens), dtype=torch.long),   # ensure the information is not flowed from pad
            "labels_rejected": torch.tensor(labels_rejected, dtype=torch.long),
            "prompt_id": self.dataset[idx]['prompt_id'],  # Keep track of the prompt ID
        }
        return {
            "input_ids_chosen": torch.tensor(prompt_tokens+chosen_tokens, dtype=torch.long),
            "attention_mask_chosen": torch.tensor([1] * len(prompt_tokens+chosen_tokens), dtype=torch.long),   # ensure the information is not flowed from pad
            "labels_chosen": torch.tensor(labels_chosen, dtype=torch.long),
            "input_ids_rejected": torch.tensor(prompt_tokens+rejected_tokens, dtype=torch.long),
            "attention_mask_rejected": torch.tensor([1] * len(prompt_tokens+rejected_tokens), dtype=torch.long),   # ensure the information is not flowed from pad
            "labels_rejected": torch.tensor(labels_rejected, dtype=torch.long),
            "prompt_id": self.dataset[idx]['prompt_id'],  # Keep track of the prompt ID
            "ref_gen": torch.tensor(prompt_tokens+ref_gen_tokens, dtype=torch.long),  # Reference generation tokens
            "ref_gen_attention_mask": torch.tensor([1] * len(prompt_tokens+ref_gen_tokens), dtype=torch.long),  # Reference generation attention mask
            "ref_gen_labels": torch.tensor(labels_gen, dtype=torch.long),  # Reference generation labels
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
    ref_gen = torch.nn.utils.rnn.pad_sequence(batch_dict["ref_gen"], batch_first=True, padding_value=tokenizer.pad_token_id)   # Reference generation tokens
    ref_gen_attention_mask = torch.nn.utils.rnn.pad_sequence(batch_dict["ref_gen_attention_mask"], batch_first=True, padding_value=0)
    ref_gen_labels = torch.nn.utils.rnn.pad_sequence(batch_dict["ref_gen_labels"], batch_first=True, padding_value=-100)   # Reference generation labels
    return {
        "input_ids_chosen": input_ids_chosen,
        "attention_mask_chosen": attention_mask_chosen,
        "labels_chosen": labels_chosen,
        "input_ids_rejected": input_ids_rejected,
        "attention_mask_rejected": attention_mask_rejected,
        "labels_rejected": labels_rejected,
        "prompt_id": [d["prompt_id"] for d in batch],  # Keep track of the prompt IDs
        "ref_gen": ref_gen,  # Reference generation tokens
        "ref_gen_attention_mask": ref_gen_attention_mask,  # Reference generation attention mask
        "ref_gen_labels": ref_gen_labels,  # Reference generation labels
    }

# Function to get DataLoader
def get_dataloader(split="train_prefs", batch_size=8, shuffle=True, num_workers=1, debug_mode=False):
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
