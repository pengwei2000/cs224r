import torch
import torch.nn.functional as F
import re
import random


def sequence_log_prob(model, input_ids, attention_mask, labels):

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    assert logits.shape[:-1] == labels.shape, f"Logits shape {logits.shape} does not match labels shape {labels.shape}"
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    seq_log_probs = (token_log_probs * loss_mask).sum(dim=-1)
    return seq_log_probs  # shape: (batch,)


def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    input_ids_chosen = batch["input_ids_chosen"]
    attention_mask_chosen = batch["attention_mask_chosen"]
    labels_chosen = batch["labels_chosen"]

    input_ids_rejected = batch["input_ids_rejected"]
    attention_mask_rejected = batch["attention_mask_rejected"]
    labels_rejected = batch["labels_rejected"]

    pi_log_prob_chosen = sequence_log_prob(policy_model, input_ids_chosen, attention_mask_chosen, labels_chosen)
    pi_log_prob_rejected = sequence_log_prob(policy_model, input_ids_rejected, attention_mask_rejected, labels_rejected)

    ref_log_prob_chosen = sequence_log_prob(ref_model, input_ids_chosen, attention_mask_chosen, labels_chosen)
    ref_log_prob_rejected = sequence_log_prob(ref_model, input_ids_rejected, attention_mask_rejected, labels_rejected)

    logratios = beta * ((pi_log_prob_chosen - ref_log_prob_chosen) - (pi_log_prob_rejected - ref_log_prob_rejected))
    loss = -F.logsigmoid(logratios).mean()
    return loss

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s÷]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        if '÷' in equation_str:
            equation_str = equation_str.replace('÷', '/')
        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, target, numbers, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score 
    
def centered_rewards(reward_list):
    k = len(reward_list)
    rewards = torch.tensor(reward_list, dtype=torch.float)
    centered = []
    for i in range(k):
        other_mean = (rewards.sum() - rewards[i]) / (k - 1)
        centered.append(rewards[i] - other_mean)
    return torch.stack(centered)