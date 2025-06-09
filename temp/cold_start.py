import json
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

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


api = psychgpt_api()
qwen_api = qwen2_api()

path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/psychbench-0219/0shot/task2_deepseek-r1.json'

with open(path, 'r') as file:
    data = json.load(file)
file.close()

print(len(data))
cnt = 0
for item in tqdm(data):
    input_info = item['query']
    output_ori = item['conversations'][1]['value']
    output_reasoning = item['reasoning_0']
    output_answer = item['answer_0']

    # print(output_ori)
    diag = re.search(r'主要诊断：(.*?)精神科共病诊断', output_ori, re.DOTALL).group(1).strip().strip('。')
    diag_full = code_convert[diag]
    diag_code = code_convert[diag].split(' ')[0]
    diag_name = code_convert[diag].split(' ')[1]

    psychgpt_ans = api.chat(input_info, stream=False)

    conversation = []
    conversation.append({'from': 'human', 'value': query})
    chosen = {"from": "gpt", "value": output}
    rejected = {"from": "gpt", "value": psychgpt_ans}
    conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
    
    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0307/firstcourse_missing_300_refine.jsonl', 'a') as f2:
        json_str = json.dumps(conversations, ensure_ascii=False)
        f2.write(json_str + '\n')
        f2.flush()


    if diag_code in output_answer or diag_name in output_answer:
        conversation = []
        conversation.append({'from':'human','value': input_info})
        conversation.append({'from':'gpt','value': f"""<think>{output_reasoning}</think>\n<answer>{output_answer}</answer>"""})
        cnt += 1
        # conversation = []
        # conversation.append({'from': 'human', 'value': input_info})
        # chosen = {"from": "gpt", "value": f"""<think>{output_reasoning}</think>\n<answer>{output_answer}</answer>"""}
        # rejected = {"from": "gpt", "value": psychgpt_ans}
        conversations = {"conversations": conversation}
        
        item = conversations
            
        with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0313_refine_task2/task2.jsonl', 'a') as file:
            # for item in conversations:
            # 将每个字典转换为JSON字符串并写入文件
            json_str = json.dumps(item, ensure_ascii=False)
            file.write(json_str + '\n')
            file.flush()
    
    else:
        patient_info = re.search(r"请根据你的知识给出专业详细的回答。\n(.*?)答：", input_info, re.DOTALL).group(1)
        prompt = f"""下面这位患者的正确诊断结果为{diag_full}，请根据患者病历信息，严格按照下面的示例的格式，撰写一份诊断分析。
        患者病历信息：\n {patient_info} \n
        
        诊断分析示例：
        <think>好的，我需要根据提供的患者信息和ICD-10标准来给出主要诊断和可能的共病诊断。首先，患者是一名13岁女性，主诉包括持续一年的情绪低落、睡眠问题、注意力不集中、自杀意念以及自残行为。此外，还有食欲下降、兴趣丧失、主动性减退，这些症状符合抑郁发作的特征。\n\n从现病史来看，患者有明显的抑郁症状，如持续的心情差、兴趣丧失、睡眠障碍、食欲下降、自杀企图，并且对舍曲林治疗反应不佳。这些症状持续存在且严重，需要考虑重度抑郁发作。根据ICD-10，如果抑郁发作达到重度且不伴有精神病性症状，应使用F32.2。但需要排除双相情感障碍的可能，因为患者是否有过躁狂或轻躁狂发作的病史。根据病史中提到的“无明显的情感高涨表现”，可以排除当前躁狂发作，因此主要诊断可能是重度抑郁发作。\n\n另外，患者有时出现突然的胸闷、手抖等躯体症状，这可能提示焦虑症状。但根据给出的选项，广泛性焦虑障碍（F41.1）是可能的共病诊断。不过需要确认是否符合广泛性焦虑的诊断标准，如持续的担忧、紧张等。患者有焦虑的躯体症状，但主要诊断是抑郁，所以可能作为共病存在。\n\n精神检查显示患者情感低落，有自残行为，无幻觉妄想，思维连贯，自知力不全。这些进一步支持抑郁发作的诊断，而非精神分裂症或其他精神病性障碍。因此，主要诊断应为F32.201（不伴有精神病性症状的重度抑郁发作），共病诊断可能是F41.1（广泛性焦虑障碍）。\n</think>\n<answer>\n\n主要诊断：F32.201 不伴有精神病性症状的重度抑郁发作  \n精神科共病诊断：F41.1 广泛性焦虑障碍</answer>
        
        诊断分析：
        <think>"""

        generated_ans = qwen_api.chat(query=prompt, stream=False)

        conversation = []
        conversation.append({'from':'human','value': input_info})
        conversation.append({'from':'gpt','value': generated_ans})
        cnt += 1
        # conversation = []
        # conversation.append({'from': 'human', 'value': input_info})
        # chosen = {"from": "gpt", "value": f"""<think>{output_reasoning}</think>\n<answer>{output_answer}</answer>"""}
        # rejected = {"from": "gpt", "value": psychgpt_ans}
        conversations = {"conversations": conversation}
        
        item = conversations
            
        with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0313_refine_task2/task2.jsonl', 'a') as file:
            # for item in conversations:
            # 将每个字典转换为JSON字符串并写入文件
            json_str = json.dumps(item, ensure_ascii=False)
            file.write(json_str + '\n')
            file.flush()





print(cnt)

