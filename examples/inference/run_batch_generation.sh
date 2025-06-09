#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python ../../src/generate_batch.py \
    --model_name_or_path /data/sjtu/wrx/model_weights/Qwen2-7B-Instruct/ \
    --template qwen \
    # --model_name_or_path /data/sjtu/wrx/model_weights/glm-4-9b-chat/ \
    # --template chatglm3 \
    
    
    
    # --finetuning_type lora
    # --adapter_name_or_path ../../saves/LLaMA2-7B/lora/sft \
