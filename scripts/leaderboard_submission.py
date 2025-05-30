import json
from vllm import LLM, SamplingParams

prompts = []
with open("ultrafeedback_leaderboard.json", 'r') as f:
    for line in f:
        data = json.loads(line)
        # print(data['prompt'])
        prompts.append(data['prompt'])

sampling_params = SamplingParams(temperature=1, max_tokens=1024)
llm = LLM(model='../checkpoints/preference_sft_20250525_pref_sft_grad_acc_length_600/step_55000')
outputs = llm.generate(prompts, sampling_params)
your_responses = [o.outputs[0].text.strip() for o in outputs]
# print(your_response)
with open("leaderboard_submission.json", "w") as f:
    for prompt, response in zip(prompts, your_responses):
        result = {}
        result['prompt'] = prompt
        result['response'] = response
        f.write(json.dumps(result) + '\n')
print("Saved your model responses to leaderboard_submission.json")
