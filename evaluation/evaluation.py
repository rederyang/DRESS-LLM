'''
Before running the evaluation code, the following files need to be prepared:
    temp_result.json:   The result generated in the previous steps consists of "question", "transferred_answer", "modern_answer", and "model_path" for each piece of data
    /PretainedModels/chinese-bert-wwm-ext-trained: Trained classifier
    /PretainedModels/Qwen1.5-14B-Chat: The unedited original model, used to calculate perplexity
    /PretainedModels/bge-large-en-v1.5: Embedding model
'''
import json
import torch
import numpy as np
import random
import os
import math
from FlagEmbedding import FlagAutoModel  # activate only when calculating semantic preservation and the env should be turned to contain transformer=4.42.0
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
import qwen2


################generate the temp_result from result.json######################

# file_path = './temp_result.json'
# with open(file_path, 'r', encoding='utf-8') as file:
#     data_list = json.load(file)

# # To add the "modern_answer" 
# file_path = '../dataset/Valid_DRC.json'
# with open(file_path, 'r', encoding='utf-8') as file:
#     groundtruth = json.load(file)


# # Update the key name to "transferred_answer"
# for index,sample in enumerate(data_list):
#     sample["transferred_answer"] = sample.pop("daiyu_answer") # pop will remove the original key
#     sample["modern_answer"] = groundtruth[index]["incorrect_answers"][0]

# # save to json
# with open('./temp_result.json', 'w', encoding='utf-8') as file:
#     json.dump(data_list, file, ensure_ascii=False, indent=4)
##############################################################################




d_answer_list = []
file_path = 'temp_result.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data_list = json.load(file)

read_model_path = data_list[0]["model_path"]
read_examples = data_list[:]
for QA in read_examples:
    del QA["model_path"]

for QA in data_list:
    d_answer_list.append(QA["transferred_answer"])

#########################################################################################
#                                    Style Intensity                                    #
#########################################################################################

# from transformers import BertTokenizer
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# from transformers import BertForSequenceClassification
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

# if torch.cuda.is_available():  
#     device = torch.device("cuda")

# tokenizer = BertTokenizer.from_pretrained("/PretainedModels/chinese-bert-wwm-ext-trained")

# input_ids = []
# for sent in d_answer_list:
#     encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
#     input_ids.append(encoded_sent)

# MAX_LEN = 150
# input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
#                           value=0, truncating="post", padding="post")

# attention_masks = []
# for sent in input_ids:
#     att_mask = [int(token_id > 0) for token_id in sent]
#     attention_masks.append(att_mask)

# labels = [1] * len(input_ids)

# input_ids = torch.tensor(input_ids)
# attention_masks = torch.tensor(attention_masks)
# labels = torch.tensor(labels)

# batch_size = 10
# dataset = TensorDataset(input_ids, attention_masks, labels)
# sampler = SequentialSampler(dataset)
# dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)


# model = BertForSequenceClassification.from_pretrained(
#     "/PretainedModels/chinese-bert-wwm-ext-trained",
#     num_labels = 2,
#     output_attentions = False,
#     output_hidden_states = False,
# )
# model.cuda()

# def flat_accuracy(preds, labels):
#     global cnt
#     print(preds)
#     for i in range(len(preds)):
#         exps = np.exp(preds[i])
#         sum = np.sum(exps)
#         pred = exps / sum
#         read_examples[cnt]['tss_score'] = float(pred[1])
#         print(read_examples[cnt]['tss_score'])
#         cnt += 1
#     pred_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return np.sum(pred_flat == labels_flat) / len(labels_flat)

# seed_val = 42
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

# model.eval()

# nb_eval_steps = 0
# tss = 0
# cnt = 0
# for batch in dataloader:
        
#     batch = tuple(t.to(device) for t in batch)
    
#     b_input_ids, b_input_mask, b_labels = batch
    
#     with torch.no_grad():        
#         outputs = model(b_input_ids, 
#                         token_type_ids=None, 
#                         attention_mask=b_input_mask)
        
#     logits = outputs[0]
#     logits = logits.detach().cpu().numpy()
#     label_ids = b_labels.to('cpu').numpy()
    
#     tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
#     # Accumulate the total accuracy.
#     tss += tmp_eval_accuracy
#     # Track the number of batches
#     nb_eval_steps += 1
# # Report the final accuracy for this validation run.
# tss_result = tss/nb_eval_steps
# print(tss_result)

#########################################################################################
#                                    Fluency Score                                      #
#########################################################################################


# model_path = '../models/Qwen1.5-14B-Chat'
# tokenizer = qwen2.Qwen2Tokenizer.from_pretrained(model_path)
# model = qwen2.Qwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# ppls = []
# fss = []

# for index, question in enumerate(d_answer_list):
#     print(index)
#     inputs = tokenizer(question, return_tensors='pt')
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     # note only one sentence is in a batch so we use [0]
#     # Hence no padding, all attion_mask should be 1
#     inputs["input_ids"] = torch.tensor([[model.config.bos_token_id]+inputs["input_ids"][0].tolist()]).to(device) # append a start token to the input
#     # print(inputs["input_ids"].shape)
#     inputs["attention_mask"] = torch.tensor([[1]+inputs["attention_mask"][0].tolist()]).to(device)
#     # print(inputs["attention_mask"].shape)
#     # if torch.all(inputs["attention_mask"] == 1):
#     #     print("Attention mask contains only 1s")
    
    # with torch.no_grad():
    #     outputs = model(**inputs) ## ** unpack dictionary so the values of "matched" keys are passed as to the corresponding arguements
    #     logits = outputs.logits
    #     logits = logits.view(logits.size(1), logits.size(2)) # size = (seq_len, vocab_size)
    #     # print(logits.size())
    #     prob = torch.nn.functional.log_softmax(logits, dim=1) 
    #     prob_list = []
    #     for i in range(prob.size(0)-1):
    #         prob_list.append(prob[i,inputs["input_ids"][0][i+1]].item()) # only consider the "correct" next word probability as the perplexity
    #     if len(prob_list) == 0:
    #         ppl = 1.0
    #     else:
    #         # standard way of perplexity
    #         prob_list = np.array(prob_list) 
    #         sum_p = np.sum(prob_list)
    #         ppl = np.exp(-1/prob_list.size * sum_p)
    #         fs = 1 / (1 + np.log(ppl))
        
    #     ppls.append(ppl)
    #     fss.append(fs)

# # ppls = np.array(ppls)
# # print(f"Total num of samples: {ppls.shape}")

# # ppl_mean_all = np.mean(ppls)
# # if ppl_mean_all == float('inf'):
# #     ppl_mean_all = "infinity"
# # print(f"Mean_perplexity_All: {ppl_mean_all}")
# # print(f"FluencyScore_All: {1 / np.log(1 + ppl_mean_all)}")

# # ppl_mean_dy = np.mean(ppls[0:200])
# # if ppl_mean_dy == float('inf'):
# #     ppl_mean_dy = "infinity"
# # print(f"Mean_perplexity_First_half: {ppl_mean_dy}")
# # print(f"FluencyScore_FirstHalf: {1 / np.log(1 + ppl_mean_dy)}")


# # ppl_mean_md = np.mean(ppls[-200:])
# # if ppl_mean_md == float('inf'):
# #     ppl_mean_md = "infinity"
# # print(f"Mean_perplexity_SecondHalf: {ppl_mean_md}")
# # print(f"FluencyScore_SecondHalf: {1 / np.log(1 + ppl_mean_md)}")


# fss = np.array(fss)
# print(f"Total num of samples: {fss.shape}")

# fs_mean_all = np.mean(fss)
# if fs_mean_all == float('inf'):
#     fs_mean_all = "infinity"
# print(f"FluencyScore_All: {fs_mean_all}")

# fs_mean_dy = np.mean(fss[0:200])
# if fs_mean_dy == float('inf'):
#     fs_mean_dy = "infinity"
# print(f"FluencyScore_FirstHalf: {fs_mean_dy}")


# fs_mean_md = np.mean(fss[-200:])
# if fs_mean_md == float('inf'):
#     fs_mean_md = "infinity"
# print(f"FluencyScore_SecondHalf: {fs_mean_md}")

# for i, QA in enumerate(read_examples):
#     QA['ppl_score'] = ppls[i]




#########################################################################################
#                                 Semantic Preservation                                 #
#########################################################################################

model = FlagAutoModel.from_finetuned('BAAI/bge-small-zh-v1.5',
                                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                      use_fp16=True)
sp_scores_firstHalf = []
sp_scores_secondHalf = []
for i, QA in enumerate(read_examples):
    cands = [QA["transferred_answer"]]
    refs = [QA["modern_answer"]]

    embeddings_1 = model.encode(cands)
    embeddings_2 = model.encode(refs)
    embeddings_1 = embeddings_1.flatten()
    embeddings_2 = embeddings_2.flatten()
    similarity = np.dot(embeddings_1, embeddings_2) / (np.linalg.norm(embeddings_1) * np.linalg.norm(embeddings_2))
    QA["BGE"] = float(similarity)
    if i < len(read_examples) // 2:
        sp_scores_firstHalf.append(QA["BGE"])
    else:
        sp_scores_secondHalf.append(QA["BGE"])
print(len(sp_scores_firstHalf))
print(len(sp_scores_secondHalf))
print("Mean Semantic Preservation Score First Half: ", np.mean(sp_scores_firstHalf))
print("Mean Semantic Preservation Score Second Half: ", np.mean(sp_scores_secondHalf))





# with open("result.json", 'w', encoding='utf-8') as new_file:
#     json.dump(data_list, new_file, ensure_ascii=False, indent=4)
# os.remove("temp_result.json")