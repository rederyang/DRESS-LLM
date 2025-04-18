import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import FlagAutoModel

if __name__ == "__main__":
    with open("../dataset/Valid_DRC.json", 'r', encoding='utf-8') as file:
        original_res = json.load(file)
    with open("../result.json", 'r', encoding='utf-8') as file:
        stylized_res = json.load(file)

    print(len(original_res)) # 400 for DRC
    print(len(stylized_res)) # 400 for DRC
    # print(original_res[200]['incorrect_answers'])
    # print(stylized_res[200]['daiyu_answer'])

    # load BGE embedding model

    model = FlagAutoModel.from_finetuned('BAAI/bge-large-zh-v1.5',
                                      query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                      use_fp16=True)
    
    # append the second half data and their stylized results to list
    sentence1 = []
    sentence2 = []
    for i in range(len(original_res)//2, len(original_res)):
        sentence1.append(original_res[i]['incorrect_answers'][0]) 
        sentence2.append(stylized_res[i]['daiyu_answer'])
    
    # use BGE to turn sentences into embeddings
    embeddings_1 = model.encode(sentence1)
    embeddings_2 = model.encode(sentence2)
    print(embeddings_1.shape)
    print(embeddings_2.shape)   
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity.diagonal())
    print(len(similarity.diagonal()))
    print("semantic preservation score: ", similarity.diagonal().mean())
    