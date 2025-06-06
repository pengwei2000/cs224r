import json
from vllm import LLM, SamplingParams

prompts = []
with open("ultrafeedback_leaderboard.json", 'r') as f:
    for line in f:
        data = json.loads(line)
        # print(data['prompt'])
        prompts.append(data['prompt'])

sampling_params = SamplingParams(temperature=0.7, max_tokens=989)
llm = LLM(model='../checkpoints/extension_20250529_unll_alpha0dot1_modelreward_linearweight/step_17000')
outputs = llm.generate(prompts, sampling_params)
your_responses = [o.outputs[0].text.strip() for o in outputs]
# print(your_response)
with open("leaderboard_submission_ext.json", "w") as f:
    for prompt, response in zip(prompts, your_responses):
        result = {}
        result['prompt'] = prompt
        result['response'] = response
        f.write(json.dumps(result) + '\n')
print("Saved your model responses to leaderboard_submission_ext.json")
