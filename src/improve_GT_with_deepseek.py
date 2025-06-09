import pandas as pd
import json
import re
import numpy as np
import ollama
from tqdm import tqdm
import threading
from ollama import Client

def augment_with_deepseek(input_path, output_path):

    # path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task2-diagnosis-f3x.jsonl'

    data = []
    with open(input_path, 'r') as file:
        for line in file:
            newline = eval(line)
            data.append(newline)
    file.close()

    path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113_deepseek/task2-diagnosis-deepseek-f3x.jsonl'

    chosen_data = []
    with open(path1, 'r') as file1:
        for line in file1:
            newline = eval(line)
            chosen_data.append(newline)
    file1.close()

    idx = 0
    for item in tqdm(data[::7]):
        input_info = "\n".join(item['conversations'][0]['value'].split("\n")[1:])
        output_info = item['conversations'][1]['value']

        # instruction = instruction
        instruction = f"""请扮演一位专业的精神科临床医生，根据下述患者信息，按照ICD-10诊断标准给出主要诊断以及共病诊断（若有）的ICD-10代码以及疾病名称（精确到亚型）。仅需要给出诊断的ICD代码及疾病名称，无需进行分析。\n输出格式为：\n主要诊断：ICD10代码及疾病名称\n精神科共病诊断：ICD10代码及疾病名称，若无则填“无”     
（可选的ICD10代码及其对应的诊断为：\nF20.0 偏执型精神分裂症 \nF20.1 青春型精神分裂症 \nF20.2 紧张型精神分裂症 \nF20.3 未分化型精神分裂症 \nF20.4 精神分裂症后抑郁 \nF20.5 残留型精神分裂症 \nF20.6 单纯型精神分裂症 \nF20.8 其它精神分裂症 \nF20.9 精神分裂症，未特定 \nF30.001 轻躁狂 \nF30.101 不伴有精神病性症状的躁狂发作 \nF30.201 伴有精神病性症状的躁狂发作 \nF30.802 兴奋状态 \nF30.901 躁狂发作 \nF30.902 躁狂状态 \nF31.001 双相情感障碍,目前为轻躁狂发作 \nF31.201 双相情感障碍,目前为伴有精神病性症状的躁狂发作 \nF31.101 双相情感障碍,目前为不伴有精神病性症状的躁狂发作 \nF31.302 双相情感障碍,目前为轻度抑郁发作 \nF31.303 双相情感障碍,目前为不伴有躯体症状的轻度抑郁发作 \nF31.304 双相情感障碍,目前为中度抑郁发作 \nF31.305 双相情感障碍,目前为不伴有躯体症状的中度抑郁发作 \nF31.311 双相情感障碍,目前为伴有躯体症状的轻度抑郁发作 \nF31.312 双相情感障碍,目前为伴有躯体症状的中度抑郁发作 \nF31.401 双相情感障碍,目前为不伴有精神病性症状的重度抑郁发作 \nF31.501 双相情感障碍,目前为伴有精神病性症状的重度抑郁发作 \nF31.601 双相情感障碍,目前为混合性发作 \nF31.701 双相情感障碍,目前为缓解状态 \nF31.901 双相情感障碍 \nF32.001 轻度抑郁发作 \nF32.002 不伴有躯体症状的轻度抑郁发作 \nF32.011 伴有躯体症状的轻度抑郁发作 \nF32.101 中度抑郁发作 \nF32.102 不伴有躯体症状的中度抑郁发作 \nF32.111 伴有躯体症状的中度抑郁发作 \nF32.201 不伴有精神病性症状的重度抑郁发作 \nF32.301 伴有精神病性症状的重度抑郁发作 \nF32.901 抑郁发作 \nF32.902 抑郁状态 \nF33.001 复发性抑郁障碍,目前为轻度发作 \nF33.002 复发性抑郁障碍,目前为伴有躯体症状的轻度发作 \nF33.011 复发性抑郁障碍,目前为不伴有躯体症状的轻度发作 \nF33.101 复发性抑郁障碍,目前为中度发作 \nF33.102 复发性抑郁障碍,目前为伴有躯体症状的中度发作 \nF33.111 复发性抑郁障碍,目前为不伴有躯体症状的中度发作 \nF33.201 复发性抑郁障碍,目前为不伴有精神病性症状的重度发作 \nF33.301 复发性抑郁障碍,目前为伴有精神病性症状的重度发作 \nF33.401 复发性抑郁障碍,目前为缓解状态 \nF33.901 复发性抑郁障碍 \nF00.- 阿尔茨海默病性痴呆 \nF06.7 轻度认知障碍 \nF10.- 酒精所致的精神和行为障碍 \nF13.- 使用镇静催眠剂所致的精神和行为障碍 \nF34.0 环性心境 \nF34.1 恶劣心境 \nF34.8 其它持续性心境（情感）障碍 \nF34.9 持续性心境（情感）障碍，未特定 \nF41.1 广泛性焦虑障碍 \nF42.- 强迫性障碍 \nF44.- 分离性障碍 \nF70 轻度精神发育迟滞 \nF71 中度精神发育迟滞 \nF72 重度精神发育迟滞 \nF73 极重度精神发育迟滞 \nF78 其它精神发育迟滞 \nF79 未特定的精神发育迟滞 \nF90 注意缺陷与多动障碍 \nF95 抽动障碍 ）\n，请根据你的知识给出专业详细的回答。
        
患者信息：{input_info}\n
        """

        # response = ollama.chat(model='deepseek-r1:70b', messages=[
        #     {
        #         'role': 'user',
        #         'content': instruction
        #     },
        # ],
        # # options = {"temperature":0.1}
        # )
        # print(response['message']['content'])

        # conversation = []
        # conversation.append({'from': 'human', 'value': item['conversations'][0]['value']})
        # conversation.append({'from': 'gpt', 'value': response['message']['content']})

        # item = {'conversations': conversation}

        # with open(output_path, 'a') as file:
        #     json_str = json.dumps(item, ensure_ascii=False)
        #     file.write(json_str + '\n')
        #     file.flush()
        
        item_chosen = chosen_data[idx]
        idx += 1
        chosen_input = item_chosen['conversations'][0]['value']
        chosen_output = item_chosen['conversations'][1]['value']

        conversation = []
        conversation.append({'from': 'human', 'value': chosen_input})
        chosen = {"from": "gpt", "value": chosen_output}
        rejected = {"from": "gpt", "value": output_info}
        conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}

        with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0213/diagnosis_deepseek_vs_qwen.jsonl', 'a') as f2:
            json_str = json.dumps(conversations, ensure_ascii=False)
            f2.write(json_str + '\n')
            f2.flush()

input_path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task2-diagnosis-f3x.jsonl'
output_path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113_deepseek/task2-diagnosis-deepseek-f3x.jsonl'
augment_with_deepseek(input_path1, output_path1)

# input_path2 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task2-diagnosis-f2x.jsonl'
# output_path2 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113_deepseek/task2-diagnosis-f2x.jsonl'
# augment_with_deepseek(input_path2, output_path2)


def augment_management_with_deepseek(input_path, output_path):
    client = Client(
        host='127.0.0.1:11435',
    )
    # path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task2-diagnosis-f3x.jsonl'

    data = []
    with open(input_path, 'r') as file:
        for line in file:
            newline = eval(line)
            data.append(newline)
    file.close()

    for item in tqdm(data[::2]):
        conversation = []
        for i in range(len(item['conversations'])//2):
            input_info = "\n".join(item['conversations'][2*i]['value'].split("\n")[1:])
            output_info = item['conversations'][2*i+1]['value']

            # instruction = instruction
            instruction = f"""请根据下面的患者病情进展记录（包括患者当前用药，不良反应，精神检查，辅助检查等信息），首先解读患者的化验检查等结果，对于患者病情管理有何提示，然后分析讨论应该如何调整下一步治疗方案，越具体越详细越好。
            患者住院记录：{input_info}            
            """

            response = ollama.chat(model='deepseek-r1:70b', messages=[
                {
                    'role': 'user',
                    'content': instruction
                },
            ],
            
            # options = {"temperature":0.1}
            )
            print(response['message']['content'])

        
            conversation.append({'from': 'human', 'value': item['conversations'][0]['value']})
            conversation.append({'from': 'gpt', 'value':response['message']['content']})

        item = {'conversations': conversation}

        with open(output_path, 'a') as file:
            json_str = json.dumps(item, ensure_ascii=False)
            file.write(json_str + '\n')
            file.flush()


# input_path3 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task5-management-f2x-instruction.jsonl'
# output_path3 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113_deepseek/task5-management-f2x-instruction.jsonl'
# augment_management_with_deepseek(input_path3, output_path3)

# input_path4 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task5-management-f3x-instruction.jsonl'
# output_path4 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113_deepseek/task5-management-deepseek-f3x-instruction.jsonl'
# augment_management_with_deepseek(input_path4, output_path4)
