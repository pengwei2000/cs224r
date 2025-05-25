from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

DATASET_NAME = "HuggingFaceTB/smol-smoltalk"
DATASET_NAME = "Asap7772/cog_behav_all_strategies"
DATASET_NAME = 'Jiayi-Pan/Countdown-Tasks-3to4'
dataset = load_dataset(DATASET_NAME, split="train")
# DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
# dataset = load_dataset(DATASET_NAME, split="train_prefs")
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
def get_token_length_countdown(example):
    target = example['target']
    numbers = example['nums']
    prompt = f'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.'
    full_text = prompt
    return {"token_length": len(tokenizer.encode(full_text))}

def get_token_length_ultrafeedback(example):  
       
    chosen = example['chosen']
    rejected = example['rejected']
    # Find the first assistant response and use previous messages as prompt

    prompt = example['prompt']
    assert len(chosen) == 2, f"Expected 2 chosen responses, got {len(chosen)}"
    assert len(rejected) == 2, f"Expected 2 rejected responses, got {len(rejected)}"
    chosen_response = chosen[1]['content']
    rejected_response = rejected[1]['content']
    
    # Combine prompt and response
    full_text_chosen = prompt + chosen_response
    full_text_rejected = prompt + rejected_response

    return {"token_length_chosen": len(tokenizer.encode(full_text_chosen)), "token_length_rejected": len(tokenizer.encode(full_text_rejected))}

dataset = dataset.map(get_token_length_countdown)
lengths = dataset["token_length"]
percentile_95 = np.percentile(lengths, 95)
median_length = np.median(lengths)
print(f"Median token length: {median_length}") # 576
print(f"95th percentile of token lengths: {percentile_95}") # 576
print(f"Max token length: {max(lengths)}") # 576
# chosen_lengths = dataset["token_length_chosen"]
# rejected_lengths = dataset["token_length_rejected"]
# percentile_95_chosen = np.percentile(chosen_lengths, 95)
# median_length_chosen = np.median(chosen_lengths)
# percentile_95_rejected = np.percentile(rejected_lengths, 95)
# median_length_rejected = np.median(rejected_lengths)
# print(f"Median token length chosen: {median_length_chosen}") # 378
# print(f"95th percentile of token lengths chosen: {percentile_95_chosen}") # 989 
# print(f"Median token length rejected: {median_length_rejected}") # 327
# print(f"95th percentile of token lengths rejected: {percentile_95_rejected}") # 967
