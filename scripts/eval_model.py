import argparse
import time
from datasets import load_dataset
from vllm import LLM, SamplingParams
from openai import OpenAI
from tqdm import tqdm

### === Configurable Constants === ###
API_BASE = "https://integrate.api.nvidia.com/v1"
REWARD_MODEL = "nvidia/llama-3.1-nemotron-70b-reward"

### === Nemotron Reward Client === ###
def create_reward_client(api_key):
    return OpenAI(base_url=API_BASE, api_key=api_key)

def get_reward_score(client, prompt, response):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    try:
        result = client.chat.completions.create(model=REWARD_MODEL, messages=messages)
        return float(result.choices[0].message.content.split(":")[-1].strip())
    except Exception as e:
        print(f"Nemotron API error: {e}")
        return None

### === Evaluation Function === ###
def evaluate_models(your_model_path, ref_model_path, api_key, max_prompts=100):
    print("Loading dataset...")
    prompts = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")["prompt"][:max_prompts]

    print("Loading models with VLLM...")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=989)

    your_llm = LLM(model=your_model_path, dtype="auto", gpu_memory_utilization=0.3)
    ref_llm = LLM(model=ref_model_path, dtype="auto", gpu_memory_utilization=0.3)

    print("Generating responses...")
    your_outputs = your_llm.generate(prompts, sampling_params)
    ref_outputs = ref_llm.generate(prompts, sampling_params)
    if not args.debug_mode:
        client = create_reward_client(api_key)
    win_labels = []

    print("Scoring with Nemotron...")
    for i in tqdm(range(len(prompts))):
        prompt = prompts[i]
        your_response = your_outputs[i].outputs[0].text.strip()
        ref_response = ref_outputs[i].outputs[0].text.strip()
        print(prompt)
        print(f"Your response: {your_response}")
        print(f"Reference response: {ref_response}")
        if not args.debug_mode:
            r1 = get_reward_score(client, prompt, your_response)
            r2 = get_reward_score(client, prompt, ref_response)
        else:
            r1 = None
            r2 = None
        if r1 is not None and r2 is not None:
            win = int(r1 > r2)
            win_labels.append(win)

        time.sleep(0.5)  # to avoid hitting API rate limits

    win_rate = sum(win_labels) / len(win_labels)
    print(f"\nFinal Win Rate (your model > reference): {win_rate:.3f} ({sum(win_labels)} / {len(win_labels)})")

### === CLI Parser === ###
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RLOO-trained model vs. baseline using Nemotron reward.")
    parser.add_argument("--your_model", type=str, default='../checkpoints/preference_dpo_20250525-005852/step_6000', help="Path to your fine-tuned model (RLOO).")
    parser.add_argument("--ref_model", type=str, default="../checkpoints/preference_sft_20250520-041117/step_55000", help="Baseline model path or HF hub ID.")
    parser.add_argument("--api_key", type=str, default="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")
    parser.add_argument("--max_prompts", type=int, default=1000, help="Number of prompts to evaluate.")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode for small dataset.")
    return parser.parse_args()

### === Main Entrypoint === ###
if __name__ == "__main__":
    args = parse_args()
    evaluate_models(
        your_model_path=args.your_model,
        ref_model_path=args.ref_model,
        api_key=args.api_key,
        max_prompts=args.max_prompts
    )
