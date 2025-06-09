    #!/bin/bash

CUDA_VISIBLE_DEVICES=5 python ../../src/eval_psychbench.py \
    --model_name_or_path /data/sjtu/wrx/model_weights/Qwen2-7B-Instruct/ \
    --adapter_name_or_path /data/sjtu/wrx/llamafactory/psychgpt-sft-qwen2-7b-lora-all-256-8192-1227 \
    --template qwen \
    --finetuning_type lora \
    --temperature 0.1 \

    # --model_name_or_path /data/sjtu/wrx/model_weights/Baichuan2-7B-Chat/ \
    # --template baichuan2 \
    # --temperature 0 \
