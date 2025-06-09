import pandas as pd
import json
import re 
import os
from tqdm import tqdm
import json
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

class psychgpt_api():
    def __init__(self):
        self.api_base_url = "http://localhost:{}/v1".format(os.environ.get("API_PORT", 8002))
        self.client = OpenAI(
            api_key="{}".format(os.environ.get("API_KEY", "0")),
            base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8002)),
        )
        self.header = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }
        self.system = [
            {
                "role": "system", 
                "content": "你是由北京安定医院开发的精神心理领域专精大模型PsychGPT。你可以辅助精神科医生完成各种临床工作。"
            }
        ]
        self.history = []
    
    def clear_history(self):
        self.history = []
    
    def chat(self, query, history=None, stream=True):
        message = []
        if history:
            message.extend(history)
        else:
            message.append(
                {
                    "role": "user",
                    "content": query
                }
            )
        completion = self.client.chat.completions.create(
            model='psychgpt',
            messages=message,
            stream=stream,
        )
        if stream:
            return completion
        else:
            print(completion.choices[0].message.content)
            return completion.choices[0].message.content

api = psychgpt_api()

# 测试knowledge，在A3和knowledge选择题上
# Knowledge 选择题
path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_refine_0314/license_test.json'

with open(path, 'r') as file:
    data = json.load(file)
file.close()

print(len(data))

correct_cnt = []
for item in tqdm(data):
    input_info = item['conversations'][0]['value']
    gt = item['conversations'][1]['value']
    ans = item['answer']

    psychgpt_ans = api.chat(input_info, stream=False)

    if ans in psychgpt_ans:
        correct_cnt.append(1)
    else:
        correct_cnt.append(0)

    conversation = []
    conversation.append({'from':'human','value': input_info})
    conversation.append({'from':'gpt','value': psychgpt_ans})
    conversations = {"conversations": conversation}

    item = conversations    
    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/psychgpt-0317/test_license.jsonl', 'a') as file:
        # for item in conversations:
        # 将每个字典转换为JSON字符串并写入文件
        json_str = json.dumps(item, ensure_ascii=False)
        file.write(json_str + '\n')
        file.flush()

print(np.sum(correct_cnt)/len(correct_cnt))