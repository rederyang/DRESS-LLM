import os
import torch
import numpy as np
import pickle
import sys
sys.path.append('../')
from utils import get_llama_activations_bau, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions
import llama
import qwen2
import argparse
import json
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--feature_path', type=str)
parser.add_argument('--asset_path', type=str)
parser.add_argument('--optimize_K', action='store_true', default=False)
parser.add_argument('--variance_threshold', type=float, default=0.95)
parser.add_argument('--default_K', type=int, default=64)
args = parser.parse_args()

a = 'top_'
b = '_heads_alpha_'
index_a = args.model_path.find(a)
index_b = args.model_path.find(b)
K = args.model_path[index_a + len(a): index_b]
alpha = args.model_path[index_b + len(b): ]
dump_path = K + '_' + alpha
print(dump_path)

# 从预训练模型加载tokenizer和模型
tokenizer = qwen2.Qwen2Tokenizer.from_pretrained(args.model_path)
model = qwen2.Qwen2ForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")

# 准备问题
questions = []
with open("dataset/Valid_DRC.json", 'r', encoding='utf-8') as file:
    data_list = json.load(file)
for QA in data_list:
    questions.append(QA["question"])

answers = []

# 读取基于数据集计算并保存好的转向向量
np.load.__defaults__=(None, True, True, 'ASCII')
probes = np.load(args.asset_path + "/probes_" + dump_path + ".npy")
top_heads = np.load(args.asset_path + "/top_heads_" + dump_path + ".npy")
np.load.__defaults__=(None, False, True, 'ASCII')
with open(args.asset_path + "/activations_" + dump_path + ".pkl", 'rb') as f:
    activations_dict = pickle.load(f)
num_heads = model.config.num_attention_heads
activations = np.load(args.feature_path + "/Qwen1.5-14B-Chat_DRC_head_wise.npy")
activations = rearrange(activations, 'b l (h d) -> b l h d', h = num_heads)

svd_s_dict = {}
svd_Vh_dict = {}
svd_K_dict = {}

def svd_decomposition(layer_no, head_no, X, optimize_K=False, variance_threshold=0.95, default_K=64):
    from scipy.linalg import svd
    U, s, Vh = svd(X, full_matrices=False)
    '''
    X: (N, 128), N个正负样本对之差
    U: (N, 128)
    s: (128, ), sigma矩阵的主对角线元素(奇异值降序)
    Vh: (128, 128)
    '''
    # 保存s, Vh
    key = 'L' + str(layer_no) + 'H' + str(head_no)
    svd_s_dict[key] = s
    svd_Vh_dict[key] = Vh

    # 确定最优的K
    if optimize_K:
        N, D = X.shape
        
        # 阶段1: 基于解释方差分析来划定搜索空间
        # 计算每个奇异值的解释方差比例
        var_explained = (s**2) / np.sum(s**2)
        # 计算累积解释方差
        cumulative_var_explained = np.cumsum(var_explained)
        # 找到满足阈值的最小K值
        K_var = np.argmax(cumulative_var_explained >= variance_threshold) + 1
        
        # 定义搜索范围
        # K_min = max(1, int(K_var/2))
        # K_max = min(s.shape[0], int(1.5 * K_var))
        K_min = 1
        K_max = s.shape[0] - 1
        
        # 阶段2: 使用贝叶斯信息准则(BIC)在搜索范围内找到最优K
        best_K = K_min
        best_BIC = float('inf')
        
        for K in range(K_min, K_max + 1):
            # 使用前K个奇异值和向量重建数据
            X_reconstructed = U[:, :K] @ np.diag(s[:K]) @ Vh[:K, :]
            
            # 计算均方误差
            MSE = np.mean((X - X_reconstructed) ** 2)
            
            # 计算BIC
            # BIC = N * D * log(MSE) + K * (N + D + 1) * log(N * D)
            BIC = N * D * np.log(MSE) + K * (N + D + 1) * np.log(N * D)
            
            if BIC < best_BIC:
                best_BIC = BIC
                best_K = K
        
        # 如果最优K还是搜索范围的边界，可能需要扩大搜索范围
        if best_K == K_min or best_K == K_max:
            print(f"Warning: Optimal K ({best_K}) is at the boundary of search range for Layer {layer_no}, Head {head_no}. Consider expanding the search range.")
        
        # 保存最优K值
        svd_K_dict[key] = best_K
        
        print(f"Layer {layer_no}, Head {head_no}: Optimal K = {best_K}, Variance explained = {cumulative_var_explained[best_K-1]:.4f}")
    else:
        # 如果不优化K，默认使用固定值64（与原始DRESS框架一致）
        svd_K_dict[key] = default_K
    
    return


def get_steering_vector(layer_no, head_no, vector, cur_activations):
    # vector: 参考steering vector
    # cur_activations: 当前layer, head的activation (bias=0情况下)
    key = 'L' + str(layer_no) + 'H' + str(head_no)
    K = svd_K_dict[key]
    s = svd_s_dict[key]
    Vh = svd_Vh_dict[key]
    Vh = Vh[:K, :]
    x = vector
    V = Vh.T
    w = np.dot(Vh, x.T)
    w2 = np.dot(Vh, cur_activations.T)
    head_activations = activations[:,layer_no,head_no,:]
    correct_activations = head_activations[::2, :]
    correct_activations = np.mean(correct_activations, axis=0)
    w4 = np.dot(Vh, correct_activations.T)
    w *= (1.0 + 0.5 * np.sign(w) * (w4 - w2))
    xx = np.dot(V, w)
    return xx


def get_activations(question):
    # prompt = tokenizer_1(question, return_tensors = 'pt').input_ids
    # 在不同的layer, head上计算question的activation与probe的相似度
    # bias=0时计算activation
    with torch.no_grad():
        for layer_no, heads in activations_dict.items():
            displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
            device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
            displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
            bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
            model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)

    prompt = question
    all_head_wise_activations = []
    device = "cuda"
    layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
    all_head_wise_activations.append(head_wise_activations[:,-1,:])
    head_wise_activations = rearrange(all_head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

    weights = []
    with torch.no_grad():
        for layer_no, heads in activations_dict.items():
            displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
            for head_no, vector in activations_dict[layer_no].items():
                cur_activations = head_wise_activations[:,layer_no,head_no,:].flatten()

                s_vector = get_steering_vector(layer_no, head_no, vector, cur_activations)
                displacement[head_no] = s_vector
                
            device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
            displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
            bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
            model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)
    return



def my_generate(w0, q_tokens, inputs):
    generated = inputs["input_ids"]
    sequence = []
    max_length = 600
    layer_num = 40
    avg_weights = [w0]
    for i in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            token = torch.tensor([probs.argmax().item()]).unsqueeze(0).to('cuda:0')
            generated = torch.cat((generated, token), dim=1)
            q_tokens = torch.cat((q_tokens, token), dim=1)
            sequence.append(token.cpu().numpy()[0][0])
            get_activations(q_tokens)

            if token.cpu().numpy()[0][0] == 151643 or token.cpu().numpy()[0][0] == 151644 or token.cpu().numpy()[0][0] == 151645: 
                break
    return sequence


print("为所有head上的转向向量作svd分解并保存")
for layer_no, heads in activations_dict.items():
        for head_no, vector in activations_dict[layer_no].items():
            head_activations = activations[:,layer_no,head_no,:]
            correct_activations = head_activations[::2, :]
            incorrect_activations = head_activations[1::2, :]
            correct_activations = correct_activations - incorrect_activations
            svd_decomposition(layer_no, head_no, correct_activations, optimize_K=args.optimize_K, variance_threshold=args.variance_threshold, default_K=args.default_K)
print("分解完毕")


for index, question in enumerate(questions):
    
    q_tokens = tokenizer(question, return_tensors = 'pt').input_ids
    w0 = get_activations(q_tokens)
    
    question = "请你对下面的语句作出回应：\n" + question + "\n好的，我的回答如下：\n"
    
    inputs = tokenizer(question, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()} 
    
    # my_generate()
    sequence = my_generate(w0, q_tokens.to('cuda:0'), inputs)
    # print(sequence)
    answer = tokenizer.decode(sequence, skip_special_tokens=True)
    print(index, answer)
    answers.append(answer)
    


output_data = []
for i in range(len(questions)):
    dict = {}
    dict["question"] = questions[i]
    dict["daiyu_answer"] = answers[i]
    dict["model_path"] = args.model_path
    output_data.append(dict)
###########################
with open("result_testing.json", 'w', encoding='utf-8') as new_file:
    json.dump(output_data, new_file, ensure_ascii=False, indent=4)

# python generate.py --model_path "/scratch/eecs556w25_class_root/eecs556w25_class/haojd/edited_model/Qwen1.5-14B-Chat_dataset_DRC_seed_42_top_64_heads_alpha_3.0" --asset_path "/scratch/eecs556w25_class_root/eecs556w25_class/haojd/features" --feature_path "/scratch/eecs556w25_class_root/eecs556w25_class/haojd/features" --optimize_K
