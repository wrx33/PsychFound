import json
import re
import pandas as pd
import os
from tqdm import tqdm
from openai import OpenAI
import ollama


path2 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task5-management-f2x-instruction.jsonl'
path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0213/diagnosis_1000.jsonl'

data = []
with open(path1, 'r') as f1:
    for line in f1:
        newline = eval(line)
        data.append(newline)
f1.close()

# cnt = 0
for item in tqdm(data):
    conversation = item['conversations']
    
    rejected_content = item['rejected']['value']
    chosen_content = item['chosen']['value'].replace(rejected_content, "")

    # conversation = []
    # conversation.append({'from': 'human', 'value': input_info})

    chosen = {"from": "gpt", "value": chosen_content}
    rejected = {"from": "gpt", "value": rejected_content}

    conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
    
    # if cnt >= 2000:
    #   break
    # cnt += 1
    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0213/diagnosis_1000_diff.jsonl', 'a') as f2:
        json_str = json.dumps(conversations, ensure_ascii=False)
        f2.write(json_str + '\n')
        f2.flush()