#!/bin/bash


config_name="edited_model_cluster"

python edit_weight.py \
    --model_name Qwen1.5-14B-Chat \
    --dataset_name DRC \
    --activation_path "/root/autodl-tmp/features/Qwen1.5-14B-Chat_DRC_head_wise.npy" \
    --label_path "/root/autodl-tmp/features/Qwen1.5-14B-Chat_DRC_labels.npy" \
    --model_dir "/root/autodl-tmp/models/Qwen1.5-14B-Chat" \
    --num_heads 64 \
    --alpha 3 \
    --selection_method cluster \
    --save_dir "/root/autodl-tmp/${config_name}/"

python generate.py \
    --model_path /root/autodl-tmp/${config_name}/Qwen1.5* \
    --dataset_path "/root/DRESS-LLM/dataset/Valid_DRC_mini.json" \
    --feature_path "/root/autodl-tmp/features/" \
    --asset_path "/root/autodl-tmp/${config_name}/" \
    --default_K 64 \
    # --optimize_K
