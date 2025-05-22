import torch
import torch.nn.functional as F

def compute_log_probs(logits, labels):
    # logits: (B, T, V), labels: (B, T)
    log_probs = F.log_softmax(logits, dim=-1)
    labels = labels.clone()
    labels[labels == -100] = 0  # avoid indexing errors
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs


def sequence_log_prob(model, input_ids, attention_mask, labels):

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    token_log_probs = compute_log_probs(logits, labels)
    seq_log_probs = (token_log_probs * (labels != -100)).sum(dim=-1)
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
