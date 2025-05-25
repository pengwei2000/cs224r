import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Dict
from transformers import AutoModelForCausalLM

# Constants
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = 'Jiayi-Pan/Countdown-Tasks-3to4'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
assert tokenizer.pad_token_id == 151643, f"Expected pad token ID to be 151643, got {tokenizer.pad_token_id}"
# Custom dataset class
class CountDownDataset(Dataset):
    def __init__(self, split: str = "train", max_length: int = 10240, debug_mode: bool = False):
        self.dataset = load_dataset(DATASET_NAME, split=split)
        if debug_mode:
            self.dataset = self.dataset.select(range(200))
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        target = self.dataset[idx]['target']
        numbers = self.dataset[idx]['nums']
        prompt = f'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.'

        # Tokenize the prompt and full text
        prompt_tokens = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=self.max_length, padding=False)["input_ids"]

        return {
            "input_ids": torch.tensor(prompt_tokens, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(prompt_tokens), dtype=torch.long),   # ensure the information is not flowed from pad
            "target": torch.tensor([target], dtype=torch.long),  # Store target for potential use
            "numbers": torch.tensor(numbers, dtype=torch.long),  # Store numbers for potential use
        }

# Collate function with dynamic padding
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Convert batch of dicts to dict of lists
    batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
    # Pad each field
    input_ids = torch.nn.utils.rnn.pad_sequence(batch_dict["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')   # batch_first if True, the output will be in B x T x * format, T x B x * otherwise.
    attention_mask = torch.nn.utils.rnn.pad_sequence(batch_dict["attention_mask"], batch_first=True, padding_value=0, padding_side='left')
    # print(labels.shape, input_ids.shape, attention_mask.shape)
    # print(batch_dict["target"], batch_dict["numbers"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target": batch_dict["target"],
        "numbers": batch_dict["numbers"],
    }

# Function to get DataLoader
def get_dataloader(split="train", batch_size=4, shuffle=True, num_workers=4, debug_mode=False, max_length=10240):
    dataset = CountDownDataset(split=split, max_length=max_length, debug_mode=debug_mode)
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
        print(batch['input_ids'].shape, batch['attention_mask'].shape)
        for i in range(batch['input_ids'].shape[0]):
            if 151643 in batch['input_ids'][i]:
                result = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
                print(result)
        # model = AutoModelForCausalLM.from_pretrained('../checkpoints/verifier_sft_20250521-171405/step_1000', trust_remote_code=True)
        # model.eval()
        # result = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_new_tokens=1024, do_sample=True, num_return_sequences=2)
        # result = tokenizer.batch_decode(result, skip_special_tokens=True)
        # print(result.shape)
        break
