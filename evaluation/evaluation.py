'''
Before running the evaluation code, the following files need to be prepared:
    temp_result.json:   The result generated in the previous steps consists of "question", "transferred_answer", "modern_answer", and "model_path" for each piece of data
    /PretainedModels/chinese-bert-wwm-ext-trained: Trained classifier
    /PretainedModels/Qwen1.5-14B-Chat: The unedited original model, used to calculate perplexity
    /PretainedModels/bge-large-en-v1.5: Embedding model
'''
import json
import qwen2
import torch
import numpy as np
import random
import os
import math
from FlagEmbedding import FlagModel
from tqdm import tqdm


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

from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

if torch.cuda.is_available():  
    device = torch.device("cuda")

tokenizer = BertTokenizer.from_pretrained("/PretainedModels/chinese-bert-wwm-ext-trained")

input_ids = []
for sent in d_answer_list:
    encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
    input_ids.append(encoded_sent)

MAX_LEN = 150
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                          value=0, truncating="post", padding="post")

attention_masks = []
for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

labels = [1] * len(input_ids)

input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)

batch_size = 10
dataset = TensorDataset(input_ids, attention_masks, labels)
sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)


model = BertForSequenceClassification.from_pretrained(
    "/PretainedModels/chinese-bert-wwm-ext-trained",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)
model.cuda()

def flat_accuracy(preds, labels):
    global cnt
    print(preds)
    for i in range(len(preds)):
        exps = np.exp(preds[i])
        sum = np.sum(exps)
        pred = exps / sum
        read_examples[cnt]['tss_score'] = float(pred[1])
        print(read_examples[cnt]['tss_score'])
        cnt += 1
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

model.eval()

nb_eval_steps = 0
tss = 0
cnt = 0
for batch in dataloader:
        
    batch = tuple(t.to(device) for t in batch)
    
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():        
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
        
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
    # Accumulate the total accuracy.
    tss += tmp_eval_accuracy
    # Track the number of batches
    nb_eval_steps += 1
# Report the final accuracy for this validation run.
tss_result = tss/nb_eval_steps
print(tss_result)

#########################################################################################
#                                    Fluency Score                                      #
#########################################################################################


model_path = '/PretainedModels/Qwen1.5-14B-Chat'
tokenizer = qwen2.Qwen2Tokenizer.from_pretrained(model_path)
model = qwen2.Qwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")

ppls = []

for index, question in enumerate(d_answer_list):
    print(index)
    inputs = tokenizer(question, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs["input_ids"] = torch.tensor([[model.config.bos_token_id]+inputs["input_ids"][0].tolist()])
    inputs["attention_mask"] = torch.tensor([[1]+inputs["attention_mask"][0].tolist()])

    with torch.no_grad():
        outputs = model(**inputs)

        logits = outputs.logits
        logits = logits.view(logits.size(1), logits.size(2))
        prob = torch.nn.functional.softmax(logits, dim=1)
        prob_list = []
        for i in range(prob.size(0)-1):
            prob_list.append(prob[i,inputs["input_ids"][0][i+1]].item())
        if len(prob_list) == 0:
            ppl = 1.0
        else:
            prob_list = np.array(prob_list)
            prob_list = prob_list ** (1.0/(prob.size(0)-1))
            prob_list = 1.0 / prob_list
            ppl = np.prod(prob_list)
        
        ppls.append(ppl)

ppls = np.array(ppls)
print(ppls.shape)
ppl_mean_all = np.mean(ppls)
if ppl_mean_all == float('inf'):
    ppl_mean_all = "infinity"
print(ppl_mean_all)
ppl_med_all = np.median(ppls)
if ppl_med_all == float('inf'):
    ppl_med_all = "infinity"
print(ppl_med_all)
ppl_mean_dy = np.mean(ppls[0:200])
if ppl_mean_dy == float('inf'):
    ppl_mean_dy = "infinity"
print(ppl_mean_dy)
ppl_med_dy = np.median(ppls[0:200])
if ppl_med_dy == float('inf'):
    ppl_med_dy = "infinity"
print(ppl_med_dy)
ppl_mean_md = np.mean(ppls[-200:])
if ppl_mean_md == float('inf'):
    ppl_mean_md = "infinity"
print(ppl_mean_md)
ppl_med_md = np.median(ppls[-200:])
if ppl_med_md == float('inf'):
    ppl_med_md = "infinity"
print(ppl_med_md)

for i, QA in enumerate(read_examples):
    QA['ppl_score'] = ppls[i]




#########################################################################################
#                                 Semantic Preservation                                 #
#########################################################################################

model = FlagModel('/PretainedModels/bge-large-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)
sp_scores = []
for i, QA in enumerate(read_examples):
    QA['ppl_score'] = ppls[i]
    cands = [QA["transferred_answer"]]
    refs = [QA["modern_answer"]]

    embeddings_1 = model.encode(cands)
    embeddings_2 = model.encode(refs)
    embeddings_1 = embeddings_1.flatten()
    embeddings_2 = embeddings_2.flatten()
    similarity = np.dot(embeddings_1, embeddings_2) / (np.linalg.norm(embeddings_1) * np.linalg.norm(embeddings_2))

    QA["BGE"] = float(similarity)
    sp_scores.append(QA["BGE"])



with open("result.json", 'r', encoding='utf-8') as file:
    data_list = json.load(file)
new_dict = {}
new_dict["sp_score"] = sum(sp_scores) / len(sp_scores)
new_dict["tss_score"] = tss_result
ppl_dict = {}
ppl_dict["mean_all"] = ppl_mean_all
ppl_dict["median_all"] = ppl_med_all
ppl_dict["mean_modern"] = ppl_mean_dy
ppl_dict["median_modern"] = ppl_med_dy
ppl_dict["mean_Shakespeare"] = ppl_mean_md
ppl_dict["median_Shakespeare"] = ppl_med_md
new_dict["ppl_score"] = ppl_dict

new_dict["examples"] = read_examples
data_list[0][read_model_path] = new_dict
with open("result.json", 'w', encoding='utf-8') as new_file:
    json.dump(data_list, new_file, ensure_ascii=False, indent=4)
# os.remove("temp_result.json")