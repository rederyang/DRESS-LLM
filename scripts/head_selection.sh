#!/bin/bash

# head selection

# option 1: select heads by individual probing
config_name = "edited_model_baseline"
python edit_weight.py \
    --model_name Qwen1.5-14B-Chat \
    --dataset_name DRC \
    --activation_path "/root/autodl-tmp/features/Qwen1.5-14B-Chat_DRC_head_wise.npy" \
    --label_path "/root/autodl-tmp/features/Qwen1.5-14B-Chat_DRC_labels.npy" \
    --model_dir "/root/autodl-tmp/models/Qwen1.5-14B-Chat" \
    --num_heads 64 \
    --alpha 3 \
    --save_dir "/root/autodl-tmp/${config_name}/"

# option 2: select heads by collaborative selection
config_name = "edited_model_cs_l0_5_ln_25_svd_64_l1_0_gr_0.05"
python edit_weight.py \
    --model_name Qwen1.5-14B-Chat \
    --dataset_name DRC \
    --activation_path "/root/autodl-tmp/features/Qwen1.5-14B-Chat_DRC_head_wise.npy" \
    --label_path "/root/autodl-tmp/features/Qwen1.5-14B-Chat_DRC_labels.npy" \
    --model_dir "/root/autodl-tmp/models/Qwen1.5-14B-Chat" \
    --num_heads 64 \
    --alpha 3 \
    --collaborative_selection \
    --l0_layer 5 \
    --ln_layer 25 \
    --select_svd_components 64 \
    --l1_reg 0.0 \
    --group_reg 0.05 \
    --save_dir "/root/autodl-tmp/${config_name}/"
# num_heads not used for collaborative selection

# option 3: select heads by heuristic
config_name = "edited_model_heuristic_64_3"
python edit_weight.py \
    --model_name Qwen1.5-14B-Chat \
    --dataset_name DRC \
    --activation_path "/root/autodl-tmp/features/Qwen1.5-14B-Chat_DRC_head_wise.npy" \
    --label_path "/root/autodl-tmp/features/Qwen1.5-14B-Chat_DRC_labels.npy" \
    --model_dir "/root/autodl-tmp/models/Qwen1.5-14B-Chat" \
    --num_heads 64 \
    --alpha 3 \
    --selection_method heuristic \
    --save_dir "/root/autodl-tmp/${config_name}/"