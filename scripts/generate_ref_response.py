import argparse
import time
from datasets import load_dataset
from vllm import LLM, SamplingParams
from openai import OpenAI
from tqdm import tqdm
import json

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
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    prompts = dataset["prompt"][:max_prompts]
    prompt_ids = dataset["prompt_id"][:max_prompts]
    assert len(prompts) == len(prompt_ids), "Mismatch in number of prompts and prompt IDs."

    print("Loading models with VLLM...")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=989)

    if args.stage == "your_model":
        print("Generating responses from your model...")
        llm = LLM(model=your_model_path, dtype="auto")
        outputs = llm.generate(prompts, sampling_params)
        your_responses = [o.outputs[0].text.strip() for o in outputs]
        with open("your_outputs.json", "w") as f:
            json.dump(your_responses, f)
        print("Saved your model responses to your_outputs.json")

    elif args.stage == "ref_model":
        print("Generating responses from reference model...")
        llm = LLM(model=ref_model_path, dtype="auto")
        outputs = llm.generate(prompts, sampling_params)
        ref_responses = []
        assert len(outputs) == len(prompts), "Mismatch in number of outputs and prompts."
        for p, p_id, o in zip(prompts, prompt_ids, outputs):
            ref_responses.append({
                "prompt_id": p_id,
                "prompt": p,
                "response": o.outputs[0].text.strip()
            })

        with open("ref_outputs_trainset.json", "w") as f:
            json.dump(ref_responses, f)
        print("Saved reference model responses to ref_outputs.json")

    elif args.stage == "score":
        print("Scoring model outputs with Nemotron...")
        with open("your_outputs.json") as f:
            your_responses = json.load(f)
        with open("ref_outputs.json") as f:
            ref_responses = json.load(f)

        assert len(your_responses) == len(ref_responses) == len(prompts), "Mismatch in number of prompts/responses."

        client = create_reward_client(args.api_key)
        win_labels = []

        for i in tqdm(range(len(prompts))):
            prompt = prompts[i]
            r1 = get_reward_score(client, prompt, your_responses[i])
            r2 = get_reward_score(client, prompt, ref_responses[i])
            if r1 is not None and r2 is not None:
                win_labels.append(int(r1 > r2))
            # time.sleep(0.5) 
        win_rate = sum(win_labels) / len(win_labels)
        print(f"\nFinal Win Rate (your model > reference): {win_rate:.3f} ({sum(win_labels)} / {len(win_labels)})")

    else:
        raise ValueError(f"Invalid stage: {args.stage}")

### === CLI Parser === ###
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RLOO-trained model vs. baseline using Nemotron reward.")
    parser.add_argument("--your_model", type=str, default='../checkpoints/preference_dpo_20250525-005852/step_6000', help="Path to your fine-tuned model (RLOO).")
    parser.add_argument("--ref_model", type=str, default="../checkpoints/preference_sft_20250525_pref_sft_grad_acc_length_600/step_55000", help="Baseline model path or HF hub ID.")
    parser.add_argument("--api_key", type=str, default="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")
    parser.add_argument("--max_prompts", type=int, default=100000, help="Number of prompts to evaluate.")
    # parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode for small dataset.")
    parser.add_argument("--stage", type=str, choices=["your_model", "ref_model", "score"], required=True,
                        help="Stage of evaluation: 'generate_your_model', 'generate_ref_model', or 'score'.")
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
