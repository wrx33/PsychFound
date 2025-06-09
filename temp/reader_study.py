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

class qwen2_api():
    def __init__(self):
        self.api_base_url = "http://localhost:{}/v1".format(os.environ.get("API_PORT", 8001))
        self.client = OpenAI(
            api_key="{}".format(os.environ.get("API_KEY", "0")),
            base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8001)),
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
        cutoff = 0
        if history:
            message.extend(history)
            # while len("".join([content['content'] for content in message])) > 2048:
            #     message.pop(0)
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
    

qwen_api = qwen2_api()

path= '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/reader_study/psychgpt_psychiatrists.jsonl'
data = []
with open(path, 'r', encoding='utf-8') as f1:
    for line in f1:
        newline = eval(line)
        data.append(newline)
f1.close()

scores1 = []
scores2 = []
scores3 = []
scores4 = []

for item in data:
    patient_info = item['patient_info']
    psychgpt_ans = f"""诊断意见：{item['psychgpt_diagnosis']}\n 鉴别分析：{item['psychgpt_differential']} \n 用药建议：{item['psychgpt_medication']}"""
    junior_ans = f"""诊断意见：{item['junior_diagnosis']}\n 鉴别分析：{item['junior_differential']} \n 用药建议：{item['junior_medication']}"""
    intermediate_ans = f"""诊断意见：{item['intermediate_diagnosis']}\n 鉴别分析：{item['intermediate_differential']} \n 用药建议：{item['intermediate_medication']}"""
    senior_ans = f"""诊断意见：{item['senior_diagnosis']}\n 鉴别分析：{item['senior_differential']} \n 用药建议：{item['senior_medication']}"""

    instruction1 = f"""请作为一位精神科临床专家，从诊断准确性，鉴别分析全面性，用药准确性和全面性等方面，对下面【A,B,C,D】给出的病例分析的内容进行偏好排序。按照偏好由高到低的顺序进行输出。只输出最终排序，不要输出其他内容。
    ###
    病例信息：{patient_info}

    ###
    A给出的病例分析：{intermediate_ans}
    B给出的病例分析：{psychgpt_ans}
    C给出的病例分析：{junior_ans}
    D给出的病例分析：{senior_ans}
    """

    # instruction2 = f"""请作为一位精神科临床专家，从诊断准确性，鉴别分析全面性，用药准确性和全面性等方面，对下面医生给出的病例分析的内容进行评分。以5分制进行评分，1分表示最差，5分表示最好。只输出最终评分，不要输出其他内容。
    # ###
    # 病例信息：{patient_info}

    # ###
    # 病例分析：{junior_ans}
    # """


    # instruction3 = f"""请作为一位精神科临床专家，从诊断准确性，鉴别分析全面性，用药准确性和全面性等方面，对下面医生给出的病例分析的内容进行评分。以5分制进行评分，1分表示最差，5分表示最好。只输出最终评分，不要输出任何其他内容。
    # ###
    # 病例信息：{patient_info}

    # ###
    # 病例分析：{intermediate_ans}
    # """

    # instruction4 = f"""请作为一位精神科临床专家，从诊断准确性，鉴别分析全面性，用药准确性和全面性等方面，对下面医生给出的病例分析的内容进行评分。以5分制进行评分，1分表示最差，5分表示最好。只输出最终评分，不要输出任何其他内容。
    # ###
    # 病例信息：{patient_info}

    # ###
    # 病例分析：{senior_ans}
    # """

    score1 = qwen_api.chat(instruction1, stream=False)
    # score2 = qwen_api.chat(instruction2, stream=False)
    # score3 = qwen_api.chat(instruction3, stream=False)
    # score4 = qwen_api.chat(instruction4, stream=False)

    item = {"偏好排序": score1}
    print(score1)
    # item = {"PsychGPT": score1, "Junior": score2, "Intermediate": score3, "Senior": score4}
    # print(f"Scores: \n PsychGPT: {score1}, Junior: {score2}, Intermediate: {score3}, Senior: {score4}")

    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/reader_study/reader_scores.jsonl', 'a') as f2:
        json_str = json.dumps(item, ensure_ascii=False)
        f2.write(json_str + '\n')
        f2.flush()

    scores1.append(score1)
    # scores2.append(score2)
    # scores3.append(score3)
    # scores4.append(score4)



print(scores1)
# print(scores2)
# print(scores3)
# print(scores4)



