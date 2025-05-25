import torch
from torch import nn, optim
import sys
sys.path.append("..")
from transformers import AutoModelForCausalLM
from data.rloo_data_verifier import get_dataloader, tokenizer, MODEL_NAME
from tqdm import tqdm
import os
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from utils import compute_score, extract_solution
import re

# model = AutoModelForCausalLM.from_pretrained('../checkpoints/verifier_sft_20250521-171405/step_1000', trust_remote_code=True)
# model.eval()
# ground_truth = {"target": 25, "numbers": [40, 9, 56]}
# target = 25
# numbers = [40, 9, 56]
# prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers [40, 9, 56], create an equation that equals 25. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step."
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# result = model._sample
# result = model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True, num_return_sequences=2)
# print(result)
# print(result.shape)
# result = tokenizer.batch_decode(result, skip_special_tokens=True)
# print(result)
# # score = [compute_score(r, ground_truth) for r in result]

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

response = client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-70b-reward",
    messages=[
        {"role": "user", "content": "I am going to Paris, what should I see?"},
        {"role": "assistant", "content": "Ah, Paris, the City of Light! There are so many amazing things to see and do..."}
    ],
)

print("Nemotron Reward Score as str:", response.choices[0].message.content.split(":")[-1].strip())
