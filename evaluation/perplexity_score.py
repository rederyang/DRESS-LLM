import argparse
import json
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
import qwen2


model_path = '../models/Qwen1.5-14B-Chat'
tokenizer = qwen2.Qwen2Tokenizer.from_pretrained(model_path)
model = qwen2.Qwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

ppls = []
fss = []

with open("../result_selected_50_DRESS_ORI.json", 'r', encoding='utf-8') as file:
    stylized_res = json.load(file)
    
stylized_responses = [item['daiyu_answer'] for item in stylized_res]

for index, question in enumerate(stylized_responses):
    print(index)
    inputs = tokenizer(question, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # note only one sentence is in a batch so we use [0]
    # Hence no padding, all attion_mask should be 1
    inputs["input_ids"] = torch.tensor([[model.config.bos_token_id]+inputs["input_ids"][0].tolist()]).to(device) # append a start token to the input
    # print(inputs["input_ids"].shape)
    inputs["attention_mask"] = torch.tensor([[1]+inputs["attention_mask"][0].tolist()]).to(device)
    # print(inputs["attention_mask"].shape)
    # if torch.all(inputs["attention_mask"] == 1):
    #     print("Attention mask contains only 1s")
    
    with torch.no_grad():
        outputs = model(**inputs) ## ** unpack dictionary so the values of "matched" keys are passed as to the corresponding arguements
        logits = outputs.logits
        logits = logits.view(logits.size(1), logits.size(2)) # size = (seq_len, vocab_size)
        # print(logits.size())
        prob = torch.nn.functional.log_softmax(logits, dim=1) 
        prob_list = []
        for i in range(prob.size(0)-1):
            prob_list.append(prob[i,inputs["input_ids"][0][i+1]].item()) # only consider the "correct" next word probability as the perplexity
        if len(prob_list) == 0:
            ppl = 1.0
        else:
            # standard way of perplexity
            prob_list = np.array(prob_list) 
            sum_p = np.sum(prob_list)
            ppl = np.exp(-1/prob_list.size * sum_p)
            fs = 1 / (1 + np.log(ppl))
        
        ppls.append(ppl)
        fss.append(fs)

ppls = np.array(ppls)
print(f"Total num of samples: {ppls.shape}")

ppl_mean_all = np.mean(ppls)
if ppl_mean_all == float('inf'):
    ppl_mean_all = "infinity"
print(f"Mean_perplexity_All: {ppl_mean_all}")
print(f"FluencyScore_All: {1 / np.log(1 + ppl_mean_all)}")

ppl_mean_dy = np.mean(ppls[0:200])
if ppl_mean_dy == float('inf'):
    ppl_mean_dy = "infinity"
print(f"Mean_perplexity_First_half: {ppl_mean_dy}")
print(f"FluencyScore_FirstHalf: {1 / np.log(1 + ppl_mean_dy)}")


ppl_mean_md = np.mean(ppls[-200:])
if ppl_mean_md == float('inf'):
    ppl_mean_md = "infinity"
print(f"Mean_perplexity_SecondHalf: {ppl_mean_md}")
print(f"FluencyScore_SecondHalf: {1 / np.log(1 + ppl_mean_md)}")


fss = np.array(fss)
print(f"Total num of samples: {fss.shape}")

fs_mean_all = np.mean(fss)
if fs_mean_all == float('inf'):
    fs_mean_all = "infinity"
print(f"FluencyScore_All: {fs_mean_all}")

fs_mean_dy = np.mean(fss[0:200])
if fs_mean_dy == float('inf'):
    fs_mean_dy = "infinity"
print(f"FluencyScore_FirstHalf: {fs_mean_dy}")


fs_mean_md = np.mean(fss[-200:])
if fs_mean_md == float('inf'):
    fs_mean_md = "infinity"
print(f"FluencyScore_SecondHalf: {fs_mean_md}")

# #python perplexity_score.py






####### original perplexity score by author ###############

# for index, question in enumerate(stylized_responses):
#     print(index)
#     inputs = tokenizer(question, return_tensors='pt')
#     # Move all inputs to GPU
#     inputs = {k: v.to(device) for k, v in inputs.items()}
    
#     # Create new tensors directly on GPU
#     inputs["input_ids"] = torch.tensor([[model.config.bos_token_id]+inputs["input_ids"][0].tolist()], device=device)
#     inputs["attention_mask"] = torch.tensor([[1]+inputs["attention_mask"][0].tolist()], device=device)
    
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         logits = logits.view(logits.size(1), logits.size(2))
#         prob = torch.nn.functional.softmax(logits, dim=1)
#         prob_list = []
#         for i in range(prob.size(0)-1):
#             prob_list.append(prob[i,inputs["input_ids"][0][i+1]].item())
#         if len(prob_list) == 0:
#             ppl = 1.0
#             fs = 1.0  # Maximum fluency for empty sequence
#         else:
#             prob_list = np.array(prob_list)
#             prob_list = prob_list ** (1.0/(prob.size(0)-1))
#             prob_list = 1.0 / prob_list
#             ppl = np.prod(prob_list)
#             fs = 1 / (1 + np.log(ppl))  # Calculate fluency score
        
#         ppls.append(ppl)
#         fss.append(fs)  # Store fluency scores

# ppls = np.array(ppls)
# print(ppls.shape)

# ppl_mean_all = np.mean(ppls)
# if ppl_mean_all == float('inf'):
#     ppl_mean_all = "infinity"
# print(ppl_mean_all)
# # ppl_med_all = np.median(ppls)
# # if ppl_med_all == float('inf'):
# #     ppl_med_all = "infinity"
# # print(ppl_med_all)

# ppl_mean_dy = np.mean(ppls[0:200])
# if ppl_mean_dy == float('inf'):
#     ppl_mean_dy = "infinity"
# print(ppl_mean_dy)
# # ppl_med_dy = np.median(ppls[0:200])
# # if ppl_med_dy == float('inf'):
# #     ppl_med_dy = "infinity"
# # print(ppl_med_dy)

# ppl_mean_md = np.mean(ppls[-200:])
# if ppl_mean_md == float('inf'):
#     ppl_mean_md = "infinity"
# print(ppl_mean_md)
# # ppl_med_md = np.median(ppls[-200:])
# # if ppl_med_md == float('inf'):
# #     ppl_med_md = "infinity"
# # print(ppl_med_md)

# fss = np.array(fss)
# print("Fluency Scores:")
# print(f"Total samples: {fss.shape}")

# fs_mean_all = np.mean(fss)
# print(f"Mean Fluency Score (All): {fs_mean_all}")

# fs_mean_dy = np.mean(fss[0:200])
# print(f"Mean Fluency Score (First Half): {fs_mean_dy}")

# fs_mean_md = np.mean(fss[-200:])
# print(f"Mean Fluency Score (Second Half): {fs_mean_md}")


