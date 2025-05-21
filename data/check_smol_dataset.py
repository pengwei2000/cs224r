from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

DATASET_NAME = "HuggingFaceTB/smol-smoltalk"
DATASET_NAME = "Asap7772/cog_behav_all_strategies"
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

def get_token_length_verifier(example):     
    prompt = example['query']
    response = example['completion']

    full_text = prompt + response
    return {"token_length": len(tokenizer.encode(full_text))}

dataset = dataset.map(get_token_length_verifier)
lengths = dataset["token_length"]
percentile_95 = np.percentile(lengths, 95)
median_length = np.median(lengths)
print(f"Median token length: {median_length}") # 576
print(f"95th percentile of token lengths: {percentile_95}") # 576