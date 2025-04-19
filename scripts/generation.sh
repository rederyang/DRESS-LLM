#!/bin/bash

# generation

# option 1: editing with fixed K
python generate.py \
    --model_path "/root/autodl-tmp/${config_name}/Qwen1.5-14B-Chat_dataset_DRC_seed_42_top_64_heads_alpha_3.0" \
    --feature_path "/root/autodl-tmp/features/" \
    --asset_path "/root/autodl-tmp/${config_name}/" \
    --default_K 64

# option 2: generate with head-specific K
python generate.py \
    --model_path "/root/autodl-tmp/${config_name}/Qwen1.5-14B-Chat_dataset_DRC_seed_42_top_64_heads_alpha_3.0" \
    --feature_path "/root/autodl-tmp/features/" \
    --asset_path "/root/autodl-tmp/${config_name}/" \
    --optimize_K \
    --variance_threshold 0.95 \
    --default_K 64