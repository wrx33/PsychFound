import pandas as pd
import json
import re
import numpy as np
import ollama
from tqdm import tqdm
import os

path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/A3'
file_list = os.listdir(path)

for file in file_list:
    file_path = os.path.join(path, file)

    data = []
    with open(file_path, 'r') as f1:
        for line in f1:
            newline = eval(line)
            data.append(newline)
    f1.close()

    for item in tqdm(data):
        for q in range(len(item['QA_pairs'])):
            question = item['description'] + '\n' + item['QA_pairs'][q]['question'] + ':\n' + str(item['QA_pairs'][q]['option'])
            anwser = item['QA_pairs'][q]['answer'] + "\n解释：" + item['QA_pairs'][q]['explanation']

            conversation = []
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': anwser})

            conversations = {'conversations': conversation}

            with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0114/task-all-shuffled-half.jsonl', 'a') as f2:
                json_str = json.dumps(conversations, ensure_ascii=False)
                f2.write(json_str + '\n')
                f2.flush()