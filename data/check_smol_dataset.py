from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

DATASET_NAME = "HuggingFaceTB/smol-smoltalk"
dataset = load_dataset(DATASET_NAME, split="train")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", use_fast=True)

def get_token_length(example):     
    sample = example['messages']
    # Find the first assistant response and use previous messages as prompt
    prompt_parts = []
    response = None
    prompt = None
    for msg in sample:
        if msg["role"] == "assistant":
            response = msg["content"]
        elif msg["role"] == "user":
            prompt = msg["content"]
        if response is not None and prompt is not None:
            break

    # Combine prompt and response
    full_text = prompt + response
    return {"token_length": len(tokenizer.encode(full_text))}

dataset = dataset.map(get_token_length)
lengths = dataset["token_length"]
percentile_95 = np.percentile(lengths, 95)
print(f"95th percentile of token lengths: {percentile_95}") # 576