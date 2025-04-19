#!/bin/bash

# feature extraction

python get_activations.py \
    Qwen1.5-14B-Chat \
    DRC \
    --model_dir "/root/autodl-tmp/models/Qwen1.5-14B-Chat" \
    --save_dir "/root/autodl-tmp/features/"