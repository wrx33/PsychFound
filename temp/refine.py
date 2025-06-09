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

# path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_refine_0314/knowledge_test.json'

# with open(path, 'r') as file:
#     data = json.load(file)
# file.close()

# print(len(data))

# for item in tqdm(data):
#     input_info = item['question']
#     output = f"<think>{item['reasoning_0']}</think>\n\n<answer>{item['answer_0']}</answer>"
#     pred = item['answer_0']
#     gt = item['conversations'][1]['value']
#     answer = item['answer']

#     pattern = re.compile(r'[A-Z]')
#     matches = pattern.findall(pred)
    
#     if matches == answer:
#         conversation = []
#         conversation.append({'from':'human','value': input_info})
#         conversation.append({'from':'gpt','value': output})
#         conversations = {"conversations": conversation}
#     else:
#         conversation = []
#         conversation.append({'from':'human','value': input_info})
#         conversation.append({'from':'gpt','value': gt})
#         conversations = {"conversations": conversation}
    
#     item = conversations    
#     with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0313_refine/task5_selection.jsonl', 'a') as file:
#         # for item in conversations:
#         # 将每个字典转换为JSON字符串并写入文件
#         json_str = json.dumps(item, ensure_ascii=False)
#         file.write(json_str + '\n')
#         file.flush()


# path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_refine_0314/knowledge_test.json'

# with open(path, 'r') as file:
#     data = json.load(file)
# file.close()

# print(len(data))

# for item in tqdm(data):
#     input_info = item['conversations'][0]['value']
#     gt = item['conversations'][1]['value']

#     conversation = []
#     conversation.append({'from':'human','value': input_info})
#     conversation.append({'from':'gpt','value': gt})
#     conversations = {"conversations": conversation}

#     item = conversations    
#     with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0313_refine/task_knowledge.jsonl', 'a') as file:
#         # for item in conversations:
#         # 将每个字典转换为JSON字符串并写入文件
#         json_str = json.dumps(item, ensure_ascii=False)
#         file.write(json_str + '\n')
#         file.flush()




code_convert = {
    '伴有精神病性症状的重度抑郁发作': 'F32.3 重度抑郁发作，伴精神病性症状',
    '复发性抑郁障碍,目前为中度发作': 'F33.1 复发性抑郁障碍，当前为中度发作',
    '复发性抑郁障碍,目前为伴有躯体症状的轻度发作': 'F33.0 复发性抑郁障碍，当前为轻度发作',
    '双相情感障碍,目前为混合性发作': 'F31.6 双相情感障碍，当前为混合发作',
    '复发性抑郁障碍,目前为缓解状态': 'F33.4 复发性抑郁障碍，当前为缓解状态',
    '双相情感障碍,目前为伴有躯体症状的中度抑郁发作': 'F31.3 双相情感障碍，当前为轻度或中度抑郁发作',
    '抑郁状态': 'F32.9 抑郁发作，未特指',
    '躁狂发作': 'F30.9 躁狂发作，未特指',
    '躁狂状态': 'F30.9 躁狂发作，未特指',
    '双相情感障碍,目前为不伴有躯体症状的中度抑郁发作': 'F31.3 双相情感障碍，当前为轻度或中度抑郁发作',
    '伴有精神病性症状的躁狂发作': 'F30.2 躁狂，伴精神病性症状',
    '伴有精神病性症状的躁狂': 'F30.2 躁狂，伴精神病性症状',
    '双相情感障碍,目前为伴有精神病性症状的躁狂发作': 'F31.2 双相情感障碍，当前为伴精神病性症状的躁狂发作',
    '双相情感障碍，目前为伴有精神病性症状的躁狂发作': 'F31.2 双相情感障碍，当前为伴精神病性症状的躁狂发作',
    '复发性抑郁障碍,目前为不伴有精神病性症状的重度发作': 'F33.2 复发性抑郁障碍，当前为不伴精神病性症状的重度发作',
    '双相情感障碍,目前为轻度抑郁发作': 'F31.3 双相情感障碍，当前为轻度或中度抑郁发作',
    '伴有躯体症状的轻度抑郁发作': 'F32.0 轻度抑郁发作',
    '双相情感障碍,目前为轻躁狂发作': 'F31.0 双相情感障碍，当前为轻躁狂发作',
    '双相情感障碍,目前为不伴有精神病性症状的重度抑郁发作': 'F31.4 双相情感障碍，当前为不伴精神病性症状的重度抑郁发作',
    '双相情感障碍，目前为不伴有精神病性症状的重度抑郁发作': 'F31.4 双相情感障碍，当前为不伴精神病性症状的重度抑郁发作',
    '双相情感障碍,目前为不伴有精神病性症状的躁狂发作': 'F31.1 双相情感障碍，当前为不伴精神病性症状的躁狂',
    '复发性抑郁障碍,目前为伴有躯体症状的中度发作': 'F33.1 复发性抑郁障碍，当前为中度发作',
    '双相情感障碍,目前为不伴有躯体症状的轻度抑郁发作': 'F31.3 双相情感障碍，当前为轻度或中度抑郁发作',
    '双相情感障碍,目前为中度抑郁发作': 'F31.3 双相情感障碍，当前为轻度或中度抑郁发作',
    '复发性抑郁障碍,目前为不伴有躯体症状的轻度发作': 'F33.0 复发性抑郁障碍，当前为轻度发作',
    '不伴有躯体症状的中度抑郁发作': 'F32.1 中度抑郁发作',
    '不伴有精神病性症状的躁狂发作': 'F30.1 躁狂，伴精神病性症状',
    '双相情感障碍': 'F31.9 双相情感障碍，未特指',
    '抑郁发作': 'F32.9 抑郁发作，未特指',
    '双相情感障碍,目前为伴有精神病性症状的重度抑郁发作': 'F31.5 双相情感障碍，当前为伴精神病性症状的重度抑郁发作',
    '双相情感障碍，目前为伴有精神病性症状的重度抑郁发作': 'F31.5 双相情感障碍，当前为伴精神病性症状的重度抑郁发作',
    '双相情感障碍,目前为伴有躯体症状的轻度抑郁发作': 'F31.3 双相情感障碍，当前为轻度或中度抑郁发作',
    '中度抑郁发作': 'F32.1 中度抑郁发作',
    '兴奋状态': 'F30.0 轻躁狂',
    '不伴有躯体症状的轻度抑郁发作': 'F32.0 轻度抑郁发作',
    '轻度抑郁发作': 'F32.0 轻度抑郁发作',
    '复发性抑郁障碍,目前为轻度发作': 'F33.0 复发性抑郁障碍，当前为轻度发作',
    '复发性抑郁障碍,目前为伴有精神病性症状的重度发作': 'F33.3 复发性抑郁障碍，当前为伴精神病性症状的重度发作',
    '复发性抑郁障碍，目前为伴有精神病性症状的重度发作': 'F33.3 复发性抑郁障碍，当前为伴精神病性症状的重度发作',
    '复发性抑郁障碍': 'F33.9 复发性抑郁障碍，未特指',
    '复发性抑郁障碍,目前为不伴有躯体症状的中度发作': 'F33.1 复发性抑郁障碍，当前为中度发作',
    '双相情感障碍,目前为缓解状态': 'F31.7 双相情感障碍，当前为缓解状态',
    '不伴有精神病性症状的重度抑郁发作': 'F32.2 重度抑郁发作，不伴精神病性症状',
    '伴有躯体症状的中度抑郁发作': 'F32.1 中度抑郁发作',
    '伴有躯体症状中度抑郁发作': 'F32.1 中度抑郁发作',
    '偏执性精神障碍': 'F20.0 偏执型精神分裂症',
    '偏执型精神分裂症': 'F20.0 偏执型精神分裂症',
    '混合性焦虑障碍': 'F41.3 其他混合性焦虑障碍',
    '急性应激反应': 'F43.0 急性应激反应',
    '轻度精神发育迟滞,显著的行为缺陷,需要加以关注或治疗': 'F70.0 轻度精神发育迟滞',
    '单纯型精神分裂症': 'F20.6 单纯型精神分裂症',
    '甲状腺功能减退所致精神障碍': 'F06.3 器质性心境（情感）障碍',
    '严重应激反应': 'F43.0 急性应激反应',
    '伴有精神分裂症症状的急性精神病性障碍': 'F23.2 急性精神分裂症样精神病性障碍',
    '抽动秽语综合征': 'F95.9 抽动障碍，未特定',
    '中度精神发育迟滞,显著的行为缺陷,需要加以关注或治疗': 'F71.0 中度精神发育迟滞',
    '中度精神发育迟缓，需要加以关注或治疗的显著行为缺陷': 'F71.0 中度精神发育迟滞',
    '强迫性障碍': 'F42.9 强迫性障碍，未特指',
    '分裂情感性障碍混合发作': 'F25.2 分裂情感性障碍，混合型',
    '分裂情感性障碍': 'F25.9 分裂情感性障碍，未特指',
    '焦虑状态': 'F41.9 焦虑障碍，未特指',
    '大麻类物质所致的精神和行为障碍': 'F12.0 使用大麻类物质所致的精神和行为障碍',
    '缄默状态': 'F94.0 选择性缄默症',
    '精神分裂症': 'F20.9 精神分裂症，未特指',
    '难治性精神分裂症': 'F20.9 精神分裂症，未特指',
    '急性精神分裂样精神病性障碍,不伴急性应激反应': 'F25.9 分裂情感性障碍，未特指',
    '产后抑郁': 'F53.9 产褥期精神障碍，未特定',
    '通常起病于童年和少年期的行为与情绪障碍': 'F94.9 童年社会功能障碍，未特定',
    '通常在童年和青少年期发病的行为和情绪障碍': 'F94.9 童年社会功能障碍，未特定',
    '未分化型精神分裂症': 'F20.3 未分化型精神分裂症',
    '焦虑障碍': 'F41.9 焦虑障碍，未特指',
    '青春型精神分裂症': 'F20.1 青春型精神分裂症',
    '急性精神分裂样精神病性障碍': 'F25.9 分裂情感性障碍，未特指',
    '心境[情感]障碍': 'F39 未特指的心境[情感]障碍',
    '心境［情感］障碍': 'F39 未特指的心境[情感]障碍',
    '混合性焦虑和抑郁障碍': 'F39 未特指的心境[情感]障碍',
    '妄想状态': 'F22.0 妄想性障碍',
    '分裂型障碍': 'F25.9 分裂情感性障碍，未特指',
    '紧张型精神分裂症': 'F20.2 紧张型精神分裂症',
    '分裂情感性障碍抑郁发作': 'F25.1 分裂情感性障碍，抑郁型',
    '非器质性精神障碍': 'F29 未特指的非器质性精神病',
    '非器质性睡眠障碍': 'F29 未特指的非器质性精神病',
    '器质性精神障碍': "F09 未特指的器质性或症状性精神障碍",
    '器质性焦虑障碍': "F09 未特指的器质性或症状性精神障碍",
    '器质性心境［情感］障碍': "F09 未特指的器质性或症状性精神障碍",
    '急性而短暂的精神病性障碍': 'F23.9 急性而短暂的精神病性障碍，未特定',
    '酒精所致的精神和行为障碍': 'F10.9 酒精引起的精神和行为障碍，未特指',
    '使用酒精引起的精神性障碍': 'F10.9 酒精引起的精神和行为障碍，未特指',
    '使用酒精引起的精神和行为障碍': 'F10.9 酒精引起的精神和行为障碍，未特指',
    '注意缺陷与多动障碍': 'F90.9 多动性障碍，未特定',
    '童年情绪障碍': 'F93.9 童年情绪障碍，未特定',
    '童年社交性焦虑障碍': 'F93.9 童年情绪障碍，未特定',
    '残留型精神分裂症': 'F20.5 残留型精神分裂症',
    '分裂情感性障碍躁狂发作': 'F25.0 分裂情感性障碍，躁狂型',
    '以妄想为主的急性精神病性障碍': 'F23.3 其他急性以妄想为主的精神病性障碍',
    '持久的妄想性障碍': 'F22.0 妄想性障碍',
    '分离[转换]性障碍': 'F44.9 分离[转换]性障碍，未特指',
    '分离［转换］性障碍': 'F44.9 分离[转换]性障碍，未特指',
    '躯体化障碍': 'F45.0 躯体化障碍',
    '躯体形式障碍': 'F45.0 躯体化障碍',
    '广泛性焦虑障碍': 'F41.1 广泛性焦虑障碍',
    '适应障碍': 'F44.9 分离[转换]性障碍，未特指',
    '创伤后应激障碍': "F43.1 创伤后应激障碍",
    '惊恐障碍［间歇发作性焦虑］': "F41.0 惊恐障碍[间歇性发作性焦虑]",
}

class psychgpt_api():
    def __init__(self):
        self.api_base_url = "http://localhost:{}/v1".format(os.environ.get("API_PORT", 8009))
        self.client = OpenAI(
            api_key="{}".format(os.environ.get("API_KEY", "0")),
            base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8009)),
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



# class qwen2_api():
#     def __init__(self):
#         self.api_base_url = "http://localhost:{}/v1".format(os.environ.get("API_PORT", 8001))
#         self.client = OpenAI(
#             api_key="{}".format(os.environ.get("API_KEY", "0")),
#             base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8001)),
#         )
#         self.header = {
#             'accept': 'application/json',
#             'Content-Type': 'application/json',
#         }
#         self.system = [
#             {
#                 "role": "system", 
#                 "content": "你是由北京安定医院开发的精神心理领域专精大模型PsychGPT。你可以辅助精神科医生完成各种临床工作。"
#             }
#         ]
#         self.history = []
    
#     def clear_history(self):
#         self.history = []
    
#     def chat(self, query, history=None, stream=True):
#         message = []
#         cutoff = 0
#         if history:
#             message.extend(history)
#             # while len("".join([content['content'] for content in message])) > 2048:
#             #     message.pop(0)
#         else:
#             message.append(
#                 {
#                     "role": "user",
#                     "content": query
#                 }
#             )
#         completion = self.client.chat.completions.create(
#             model='psychgpt',
#             messages=message,
#             stream=stream,
#         )
#         if stream:
#             return completion
#         else:
#             print(completion.choices[0].message.content)
#             return completion.choices[0].message.content


api = psychgpt_api()
# qwen_api = qwen2_api()

# path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0313_refine/task2.jsonl'

# data = []
# with open(path, 'r', encoding='utf-8') as f1:
#     for line in f1:
#         newline = eval(line)
#         data.append(newline)
# f1.close()

# print(len(data))
# cnt = 0
# for item in tqdm(data):
#     input_info = item['conversations'][0]['value']
#     output_ori = item['conversations'][1]['value']

#     psychgpt_ans = api.chat(input_info, stream=False)

#     conversation = []
#     conversation.append({'from': 'human', 'value': input_info})
#     chosen = {"from": "gpt", "value": output_ori}
#     rejected = {"from": "gpt", "value": psychgpt_ans}
#     conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
    
#     with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_refine_task2/task2_dpo.jsonl', 'a') as f2:
#         json_str = json.dumps(conversations, ensure_ascii=False)
#         f2.write(json_str + '\n')
#         f2.flush()


# 对CME和CMB进行优化

# path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/cme_cmb/CMB_psych.json'
path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/cme_cmb/CMExam_psych.json'

with open(path, 'r') as file:
    data = json.load(file)
file.close()
acc = []
for item in tqdm(data):
    # input_info = f"你是一名精神科医生，请完成下面的精神科医师规培结业考试题目。不需要做任何分析和解释，直接输出答案选项。\n 题目：{item['question']}：{item['option']}"
    # gt = f"{item['answer']} {item['option'][item['answer']]}"

    input_info = f"你是一名精神科医生，请完成下面的精神科相关考试题目。不需要做任何分析和解释，直接输出答案选项。\n 题目：{item['QA_pairs'][0]['question']}：{item['QA_pairs'][0]['option']}"
    gt = f"{item['QA_pairs'][0]['answer']} {item['QA_pairs'][0]['option'][item['QA_pairs'][0]['answer']]}"

    psychgpt_ans = api.chat(input_info, stream=False)
    try:
        final_ans = re.search(r"<answer>(.*?)</answer>", psychgpt_ans, re.DOTALL).group(1).strip()
    except:
        final_ans = psychgpt_ans
    
    
    if gt in final_ans or final_ans in gt:
        conversation = []
        conversation.append({'from': 'human', 'value': input_info})
        chosen = {"from": "gpt", "value": psychgpt_ans}
        rejected = {"from": "gpt", "value": gt}
        conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
    else:
        conversation = []
        conversation.append({'from': 'human', 'value': input_info})
        chosen = {"from": "gpt", "value": gt}
        rejected = {"from": "gpt", "value": psychgpt_ans}
        conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}

    

    item = conversations    
    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0417_refine/CME.jsonl', 'a') as file:
        # for item in conversations:
        # 将每个字典转换为JSON字符串并写入文件
        json_str = json.dumps(item, ensure_ascii=False)
        file.write(json_str + '\n')
        file.flush()

# print(np.sum(acc)/len(acc))

path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/cme_cmb/CMB_psych.json'
# path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/cme_cmb/CMExam_psych.json'

with open(path, 'r') as file:
    data = json.load(file)
file.close()
acc = []
for item in tqdm(data):
    input_info = f"你是一名精神科医生，请完成下面的精神科医师规培结业考试题目。不需要做任何分析和解释，直接输出答案选项。\n 题目：{item['question']}：{item['option']}"
    gt = f"{item['answer']} {item['option'][item['answer']]}"

    # input_info = f"你是一名精神科医生，请完成下面的精神科相关考试题目。不需要做任何分析和解释，直接输出答案选项。\n 题目：{item['QA_pairs'][0]['question']}：{item['QA_pairs'][0]['option']}"
    # gt = f"{item['QA_pairs'][0]['answer']} {item['QA_pairs'][0]['option'][item['QA_pairs'][0]['answer']]}"

    psychgpt_ans = api.chat(input_info, stream=False)
    try:
        final_ans = re.search(r"<answer>(.*?)</answer>", psychgpt_ans, re.DOTALL).group(1).strip()
    except:
        final_ans = psychgpt_ans
    
    if gt in final_ans or final_ans in gt:
        conversation = []
        conversation.append({'from': 'human', 'value': input_info})
        chosen = {"from": "gpt", "value": psychgpt_ans}
        rejected = {"from": "gpt", "value": gt}
        conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
    else:
        conversation = []
        conversation.append({'from': 'human', 'value': input_info})
        chosen = {"from": "gpt", "value": gt}
        rejected = {"from": "gpt", "value": psychgpt_ans}
        conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}

    item = conversations    
    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0417_refine/CMB.jsonl', 'a') as file:
        # for item in conversations:
        # 将每个字典转换为JSON字符串并写入文件
        json_str = json.dumps(item, ensure_ascii=False)
        file.write(json_str + '\n')
        file.flush()

# print(np.sum(acc)/len(acc))