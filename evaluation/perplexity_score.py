import argparse
import json
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
import qwen2


def calculate_batch_perplexity(model, tokenizer, input_texts):
    inputs = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    print(input_ids.shape, attention_mask.shape)

    # Pass the input batch through the model to get logits
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Shift the logits and input_ids to align targets correctly
    # Logits dimensions are: (batch_size, seq_length, vocab_size) 
    shift_logits = logits[:, :-1, :]  # Ignore the last token's logits
    shift_labels = input_ids[:, 1:]   # Skip the first token in the labels

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather the log probabilities for the correct tokens
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out positions corresponding to padding tokens
    target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.dtype)

    # Compute the mean negative log-likelihood for each sequence
    negative_log_likelihood = -target_log_probs.sum(dim=-1) / attention_mask[:, 1:].sum(dim=-1)

    # Compute perplexity for each sequence
    perplexities = torch.exp(negative_log_likelihood)
    perplexities = perplexities.tolist()

    return perplexities

if __name__ == "__main__":
    with open("../result.json", 'r', encoding='utf-8') as file:
        stylized_res = json.load(file)
    MODEL = "../models/Qwen1.5-14B-Chat"

    
    stylized_responses = [item['daiyu_answer'] for item in stylized_res]

    # memory required is high, seperate for testing
    stylized_responses = stylized_responses[180:220] 
    # stylized_responses = stylized_responses[100:110] 

    tokenizer = qwen2.Qwen2Tokenizer.from_pretrained(MODEL)
    model = qwen2.Qwen2ForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"
    model.to(device)
    # perplexity scores
    perplexities = calculate_batch_perplexity(model, tokenizer, stylized_responses)
    mean_perplexity_score = np.mean(perplexities)
    print(mean_perplexity_score.shape)
    print(f"Mean Perplexity Score: {mean_perplexity_score}")

    # fluency scores
    fluency_scores = [1 / (1 + np.log(ppl)) for ppl in perplexities]
    mean_fluency_score = np.mean(fluency_scores)
    print(f"Mean Fluency Score: {mean_fluency_score}")

# python perplexity_score.py 




