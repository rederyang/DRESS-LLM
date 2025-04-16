# python get_activations.py Qwen1.5-14B-Chat DRC --model_dir "/data/CharacterAI/PretainedModels/Qwen1.5-14B-Chat"
import os
import torch
import numpy as np
import pickle
from utils import get_llama_activations_bau, tokenized_tqa, tokenized_tqa_gen_DRC, tokenized_tqa_gen_Shakespeare
import llama
import qwen2
import argparse
import json
from tqdm import tqdm

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='Qwen1.5-14B-Chat')
    parser.add_argument('dataset_name', type=str, default='Daiyu')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--save_dir", type=str, default=None, help='local directory to save activations')
    args = parser.parse_args()

    MODEL = args.model_dir

    tokenizer = qwen2.Qwen2Tokenizer.from_pretrained(MODEL)
    model = qwen2.Qwen2ForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"

    if args.dataset_name == "DRC": 
        with open("dataset/Train_DRC.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        formatter = tokenized_tqa_gen_DRC
    elif args.dataset_name == "Shakespeare": 
        with open("dataset/Train_Shakespeare.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        formatter = tokenized_tqa_gen_Shakespeare
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    print(len(dataset))
    prompts, labels = formatter(dataset, tokenizer)
    print(len(prompts), len(labels))

    # interleave structure:
    # prompts = [(qeury + correct_answer), (query + incorrect_answer), (query + correct_answer), (query + incorrect_answer), ...]
    
    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    import gc
    for prompt in tqdm(prompts):
        ''' Input (5120)
        1. Self-Attention Block:
        → Split to heads (40 × 128)
        → Attention
        → Concatenate (5120)
        → Linear projection (5120) ← head_wise_hidden_states
        → Add & LayerNorm
        ↓
        2. MLP Block:
        → FFN
        → Linear projection
        → Add & LayerNorm ← hidden_states
        '''

        # output for each layer:layer_wise_activations [num_layers, sequence_length, hidden_size]
        # output for each head: head_wise_activations [num_heads, sequence_length, head_dim * num_heads = hidden_size] 
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
        layer_wise_activations_wanted = layer_wise_activations[:,-1,:].copy() # only last token [num_layers, hidden_size]
        head_wise_activations_wanted = head_wise_activations[:,-1,:].copy() # only last token [num_heads, hidden_size]
        del layer_wise_activations, head_wise_activations, _
        all_layer_wise_activations.append(layer_wise_activations_wanted)
        all_head_wise_activations.append(head_wise_activations_wanted) # [num_prompts, num_heads, hidden_size]

        gc.collect()

    os.makedirs(f'{args.save_dir}', exist_ok=True)

    print("Saving labels")
    np.save(os.path.join(args.save_dir, f'{args.model_name}_{args.dataset_name}_labels.npy'), labels)

    print("Saving layer wise activations")
    np.save(os.path.join(args.save_dir, f'{args.model_name}_{args.dataset_name}_layer_wise.npy'), all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(os.path.join(args.save_dir, f'{args.model_name}_{args.dataset_name}_head_wise.npy'), all_head_wise_activations)
    

if __name__ == '__main__':
    main()