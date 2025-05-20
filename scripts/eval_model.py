from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./preference_sft", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./preference_sft", trust_remote_code=True)

prompt = "Explain the concept of reinforcement learning:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
