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

'''
消融实验和补充实验
1. RL思维增强 vs 无RL思维增强
2. 知识注入+思维增强 vs 直接思维增强
3. 输入患者信息变化对模型输入的影响
4. 对模型犯错的case进行error analysis，分析出错原因
5. 模型对harm的控制，人为评价。
'''

code_convert = {
    '伴有精神病性症状的重度抑郁发作': 'F32.3 重度抑郁发作，伴精神病性症状',
    '重度抑郁发作，伴精神病性症状': 'F32.3 重度抑郁发作，伴精神病性症状',
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
    '躁狂，伴精神病性症状': 'F30.2 躁狂，伴精神病性症状',
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
    '不伴有精神病性症状的躁狂发作': 'F30.1 躁狂，不伴精神病性症状',
    '躁狂，不伴精神病性症状': 'F30.1 躁狂，不伴精神病性症状',
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

class psychgpt_r1_api():
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
            # print(completion.choices[0].message.content)
            return completion.choices[0].message.content


class psychgpt_api():
    def __init__(self):
        self.api_base_url = "http://localhost:{}/v1".format(os.environ.get("API_PORT", 8004))
        self.client = OpenAI(
            api_key="{}".format(os.environ.get("API_KEY", "0")),
            base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8004)),
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
            # print(completion.choices[0].message.content)
            return completion.choices[0].message.content


psychgpt_r1_api = psychgpt_r1_api()
psychgpt_api = psychgpt_api()

def ablation_reasoning():
    '''
    关于有/无RL思维增强的模型对比
    '''

    path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_refine_0314/task2.jsonl'

    data = []
    with open(path, 'r', encoding='utf-8') as f1:
        for line in f1:
            newline = eval(line)
            data.append(newline)
    f1.close()

    print(len(data))
    psychgpt_r1_cnt = []
    psychgpt_cnt = []
    cnt = 98
    error_cases = []
    wrong_cases = []

    for item in tqdm(data[99:]):
        cnt +=1

        input_info = item['conversations'][0]['value']
        output_ori = item['conversations'][1]['value']
        try:
            gt = re.search(r"<answer>(.*?)</answer>", output_ori, re.DOTALL).group(1).strip()
        except:
            gt = output_ori
        try:
            gt = re.search(r"主要诊断：(.*?)\n", gt, re.DOTALL).group(1).strip()
        except:
            try:
                gt = re.search(r"主要诊断：(.*)", gt, re.DOTALL).group(1).strip()
            except:
                pass

        try:
            gt = code_convert[gt.split(' ')[1]]
        except:
            pass

        psychgpt_r1_ans = psychgpt_r1_api.chat(input_info, stream=False)
        try:
            psychgpt_r1_diag = re.search(r"<answer>(.*?)</answer>", psychgpt_r1_ans, re.DOTALL).group(1).strip()
        except:
            psychgpt_r1_diag = psychgpt_r1_ans

        try:
            psychgpt_r1_diag = re.search(r"主要诊断：(.*?)\n", psychgpt_r1_diag, re.DOTALL).group(1).strip()
        except:
            try:
                psychgpt_r1_diag = re.search(r"主要诊断：(.*)", psychgpt_r1_diag, re.DOTALL).group(1).strip()
            except:
                pass

        try:
            psychgpt_r1_diag = code_convert[psychgpt_r1_diag.split(' ')[1]]
        except:
            pass

        psychgpt_ans = psychgpt_api.chat(input_info, stream=False)
        try:
            psychgpt_diag = re.search(r"诊断：(.*?)诊断及鉴别诊断分析", psychgpt_ans, re.DOTALL).group(1).strip().replace('。', '')
        except:
            psychgpt_diag = psychgpt_ans
        try:
            psychgpt_diag = code_convert[psychgpt_diag]
        except:
            pass
        
        print(gt)
        print(psychgpt_r1_diag)
        print(psychgpt_diag)


        try:
            if gt == psychgpt_r1_diag or gt.split(' ')[1] == psychgpt_r1_diag.split(' ')[1]:
                psychgpt_r1_cnt.append(1)
            else:
                psychgpt_r1_cnt.append(0)
                error_cases.append(cnt)
        except:
            psychgpt_r1_cnt.append(0)
            error_cases.append(cnt)
        
        try:
            if gt == psychgpt_diag or gt.split(' ')[1] == psychgpt_diag.split(' ')[1]:
                psychgpt_cnt.append(1)
            else:
                psychgpt_cnt.append(0)
                # error_cases.append(cnt)
        except:
            psychgpt_cnt.append(0)

        
        print(psychgpt_r1_cnt)
        print(psychgpt_cnt)
        print(error_cases)
        conversation = []
        conversation.append({'from': 'human', 'value': input_info})
        conversation.append({'from': 'gpt', 'value': gt})
        psychgpt_r1 = {"from": "gpt", "value": psychgpt_r1_ans}
        psychgpt = {"from": "gpt", "value": psychgpt_ans}
        conversations = {"conversations": conversation, "psychgpt_r1": psychgpt_r1, "psychgpt": psychgpt}
        
        with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/ablation/psychgpt-r1_vs_psychgpt_task2_1.jsonl', 'a') as f2:
            json_str = json.dumps(conversations, ensure_ascii=False)
            f2.write(json_str + '\n')
            f2.flush()


    # print(cnt)
    print(f"error_cases: {error_cases}")
    # print(f"wrong_cases: {wrong_cases}")

    print(np.sum(psychgpt_r1_cnt)/len(psychgpt_r1_cnt))
    print(np.sum(psychgpt_cnt)/len(psychgpt_cnt))
    accuracy = {"psychgpt_r1": np.sum(psychgpt_r1_cnt)/len(psychgpt_r1_cnt), "psychgpt": np.sum(psychgpt_cnt)/len(psychgpt_cnt), "error_cases": error_cases}
    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/ablation/psychgpt-r1_vs_psychgpt_task2_1.jsonl', 'a') as f2:
            json_str = json.dumps(accuracy, ensure_ascii=False)
            f2.write(json_str + '\n')
            f2.flush()

# ablation_reasoning()

def ablation_knowledge_injection():

    '''
    关于在推理强化前有/无进行专业知识注入
    '''
    
    # psychgpt_r1_api = psychgpt_r1_api()
    # psychgpt_api = psychgpt_api()

    path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/ablation/psychgpt-r1_vs_psychgpt_task2.jsonl'
    data1 = []
    with open(path1, 'r', encoding='utf-8') as f1:
        for line in f1:
            newline = eval(line)
            data1.append(newline)
    f1.close()

    path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_refine_0314/task2.jsonl'

    data = []
    with open(path, 'r', encoding='utf-8') as f1:
        for line in f1:
            newline = eval(line)
            data.append(newline)
    f1.close()

    print(len(data))
    psychgpt_r1_cnt = []
    psychgpt_cnt = []
    cnt = 0
    error_cases = []
    wrong_cases = []

    for idx, item in enumerate(tqdm(data[6:])):
        
        input_info = item['conversations'][0]['value'].replace("仅需要给出诊断的ICD代码及疾病名称，无需进行分析。", "")

        output_ori = item['conversations'][1]['value']
        try:
            gt = re.search(r"<answer>(.*?)</answer>", output_ori, re.DOTALL).group(1).strip()
            try:
                gt = re.search(r"主要诊断：(.*?)\n", gt, re.DOTALL).group(1).strip()
            except:
                try:
                    gt = re.search(r"主要诊断：(.*)", gt, re.DOTALL).group(1).strip()
                except:
                    gt = output_ori

        except:
            gt = output_ori

        # psychgpt_r1_ans = psychgpt_r1_api.chat(input_info, stream=False)
        # psychgpt_r1_ans = data1[idx]['psychgpt_r1']['value']
        # psychgpt_r1_diag = re.search(r"<answer>(.*?)</answer>", psychgpt_r1_ans, re.DOTALL).group(1).strip()

        # try:
        #     psychgpt_r1_diag = re.search(r"主要诊断：(.*?)\n", psychgpt_r1_diag, re.DOTALL).group(1).strip()
        # except:
        #     psychgpt_r1_diag = re.search(r"主要诊断：(.*)", psychgpt_r1_diag, re.DOTALL).group(1).strip()

        # try:
        #     psychgpt_r1_diag = code_convert[psychgpt_r1_diag.split(' ')[1]]
        # except:
        #     pass

        psychgpt_ans = psychgpt_api.chat(input_info, stream=False)

        # try:
        #     psychgpt_diag = re.search(r"主要诊断：(.*?)\n", psychgpt_diag, re.DOTALL).group(1).strip()
        # except:
        #     psychgpt_diag = re.search(r"主要诊断：(.*)", psychgpt_diag, re.DOTALL).group(1).strip()

        # try:
        #     psychgpt_diag = code_convert[psychgpt_diag]
        # except:
        #     pass

        # if gt == psychgpt_r1_diag or gt.split(' ')[1] == psychgpt_r1_diag.split(' ')[1]:
        #     psychgpt_r1_cnt.append(1)
        # else:
        #     psychgpt_r1_cnt.append(0)
        #     error_cases.append(cnt)
        
        # if gt == psychgpt_diag or gt.split(' ')[1] in psychgpt_diag.split(' ')[1]:
        #     psychgpt_cnt.append(1)
        # else:
        #     psychgpt_cnt.append(0)
        #     error_cases.append(cnt)

        
        # print(psychgpt_r1_cnt)
        # print(psychgpt_cnt)
        conversation = []
        conversation.append({'from': 'human', 'value': input_info})
        conversation.append({'from': 'gpt', 'value': gt})
        # psychgpt_r1 = {"from": "gpt", "value": "pass"}
        psychgpt = {"from": "gpt", "value": psychgpt_ans}
        conversations = {"conversations": conversation,  "psychgpt": psychgpt}
        
        with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/ablation/w_wo_KnowledgeInjection_task2_2.jsonl', 'a') as f2:
            json_str = json.dumps(conversations, ensure_ascii=False)
            f2.write(json_str + '\n')
            f2.flush()


    # print(cnt)
    # print(f"error_cases: {error_cases}")
    # print(f"wrong_cases: {wrong_cases}")

    # print(np.sum(psychgpt_r1_cnt)/len(psychgpt_r1_cnt))
    # print(np.sum(psychgpt_cnt)/len(psychgpt_cnt))
    # accuracy = {"psychgpt_r1": np.sum(psychgpt_r1_cnt)/len(psychgpt_r1_cnt), "psychgpt": np.sum(psychgpt_cnt)/len(psychgpt_cnt)}
    # with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/ablation/w_wo_KnowledgeInjection_task2.jsonl', 'a') as f2:
    #         json_str = json.dumps(accuracy, ensure_ascii=False)
    #         f2.write(json_str + '\n')
    #         f2.flush()

# ablation_knowledge_injection()

def ablation_reaction():
    '''
    测试模型对于输入患者病情的变化的响应
    '''
    test_case1 = "阅读下面的患者病历信息，诊断患者可能的精神疾病，并分析诊断依据，同时完成鉴别诊断。患者病历信息：患者男性，19岁。\n现病史:（部分病史家属回忆不清）2020年10月患者无明显诱因下出现情绪差，没兴趣，心烦，后逐渐出现凭空闻声，看不见人的时候可以听见声音，听到路上有人说“你活着有什么用”等；敏感多疑，认为有人在监视自己，周围人在嘲笑自己、议论自己，自己不管做什么都会被别人知道，情绪差加重，不断哭泣，自残，遂休学，于当地治疗，诊断不详，予舍曲林、奥氮平、利培酮等药物治疗（具体剂量不详），服药后2月中体重增长15kg，病情无好转。同年底，患者跟母亲争吵后跳楼，导致左锁骨、右腓骨远端、跟骨骨折，L4压缩性骨折，外固定后好转，未遗留肢体活动不利。2021-01-27患者骨折等好转后，因仍情绪差及凭空闻声，于重庆医科大学附属医院治疗，诊断“抑郁发作”，予草酸艾司西酞普兰20mg qd+阿立哌唑 10mg qd治疗，情绪较前好转，凭空闻声频率减少，好转出院。出院后患者规律服药，生活基本正常，可自理，在家休养数月后，同年8月复学，复学后可考年级前10。10月起患者再次出现情绪差，看到老师、同学都紧张害怕、心慌，在学校多次自残自伤，凭空闻声，敏感多疑，在老师建议下遂再次休学，休学在家玩手机，与家人交流少，反复向母亲要钱，不给钱就以死相逼，遂于北医六院、当地医院等治疗，诊断不详，调整治疗为布南色林8mg bid、伏硫西汀10mg qd、文拉法辛75mg qd、喹硫平200mg qd，规律用药，病情无明显改善。 2022年1月，患者与母亲争吵后，在家烧炭自杀，未昏迷，患者家属报警，警察来后，患者又企图跳楼，被阻止，未诊治。2月患者母亲给患者口服中药及朱砂（每日约3g）治疗，患者仍情绪不稳，伴有轻生观念，02-11首次就诊于我院，完善CT提示，胃内多发不规则状高密度影，遂就诊于北大第一医院，行腹部CT检查，阑尾、结直肠内散在点状高密度，结合临床，家属表示医生考虑进一步观察即可，遂02-15于在家属陪同下来我院就诊，急诊以“抑郁状态”第1次收入院。本次为非自愿入院。入院前2周，患者饮食、睡眠、二便无殊、体重无明显变化。2022-02-08患者着凉后发热，Tmax38.8℃，服用连花清瘟胶囊后退热，未再发热，否认咳嗽、腹泻表现。患者为非自愿，家属已签署非自愿相关协议书。\n既往史：左锁骨、右腓骨远端、跟骨骨折，L4压缩性骨折后；\n家族史：无殊\n查体，辅助检查及精神检查：无殊；\n  8.精神检查：意识清，定向力完整；主被动接触可，对答切题，语速语量适中，承认既往凭空闻声，内容多为嘲笑自己，现在已几乎消失；承认既往关系妄想，认为周围人都在议论自己等，目前已消失；诉持续的情绪差、没兴趣、觉得活着没意思，诉起病前曾有近1年情绪不稳定，有时有较为高兴的时候，很爱说话，很想加入同学的聊天；睡眠、饮食、二便大致正常，部分自知力。：意识清，定向力完整；接触、对答可，目前未引出持续感知觉障碍，未引出思维形式及内容障碍；诉自起病起情绪持续不稳定，起初为持续心情差，没兴趣，高兴不起来，近半年则以情绪不稳定突出，一天之内情绪像做过山车，好的时候充满干劲，认真听课、认真做作业，没有原因的就想笑，不好的时候悲伤，觉得自己做什么都不行，有时心情不好的时候会出现幻听，大约1周会出现1次，内容为声音说自己不行；对既往2次自杀行为，诉第一次跳楼是跟母亲吵架后冲动下出现的，第二次烧炭自杀是计划了几天，就算不跟母亲吵架，自己也会落实，但对2次行为均不感到后悔；最近2周仍是情绪低落为主，没办法继续上课，觉得活着没意思，在病房内多独处，卧床睡觉，一般情况食大致正常，部分自知力。\n入院日期：2022年02月15日\n"
    # 双相情感障碍,目前为伴有精神病性症状的重度抑郁发作

    test_case2 = "阅读下面的患者病历信息，诊断患者可能的精神疾病，并分析诊断依据，同时完成鉴别诊断。患者病历信息：患者男性，19岁。\n现病史:（部分病史家属回忆不清）2020年10月患者无明显诱因下出现情绪差，没兴趣，心烦；情绪差加重，不断哭泣，自残，遂休学，于当地治疗，诊断不详，予舍曲林、奥氮平、利培酮等药物治疗（具体剂量不详），服药后2月中体重增长15kg，病情无好转。同年底，患者跟母亲争吵后跳楼，导致左锁骨、右腓骨远端、跟骨骨折，L4压缩性骨折，外固定后好转，未遗留肢体活动不利。2021-01-27患者骨折等好转后，因仍情绪差，于重庆医科大学附属医院治疗，诊断“抑郁发作”，予草酸艾司西酞普兰20mg qd+阿立哌唑 10mg qd治疗，情绪较前好转出院。出院后患者规律服药，生活基本正常，可自理，在家休养数月后，同年8月复学，复学后可考年级前10。10月起患者再次出现情绪差，看到老师、同学都紧张害怕、心慌，在学校多次自残自伤，敏感，在老师建议下遂再次休学，休学在家玩手机，与家人交流少，反复向母亲要钱，不给钱就以死相逼，遂于北医六院、当地医院等治疗，诊断不详，调整治疗为布南色林8mg bid、伏硫西汀10mg qd、文拉法辛75mg qd、喹硫平200mg qd，规律用药，病情无明显改善。 2022年1月，患者与母亲争吵后，在家烧炭自杀，未昏迷，患者家属报警，警察来后，患者又企图跳楼，被阻止，未诊治。2月患者母亲给患者口服中药及朱砂（每日约3g）治疗，患者仍情绪不稳，伴有轻生观念，02-11首次就诊于我院，完善CT提示，胃内多发不规则状高密度影，遂就诊于北大第一医院，行腹部CT检查，阑尾、结直肠内散在点状高密度，结合临床，家属表示医生考虑进一步观察即可，遂02-15于在家属陪同下来我院就诊，急诊以“抑郁状态”第1次收入院。本次为非自愿入院。入院前2周，患者饮食、睡眠、二便无殊、体重无明显变化。2022-02-08患者着凉后发热，Tmax38.8℃，服用连花清瘟胶囊后退热，未再发热，否认咳嗽、腹泻表现。患者为非自愿，家属已签署非自愿相关协议书。\n既往史：左锁骨、右腓骨远端、跟骨骨折，L4压缩性骨折后；\n家族史：无殊\n查体，辅助检查及精神检查：无殊；\n  8.精神检查：意识清，定向力完整；主被动接触可，对答切题，语速语量适中；诉持续的情绪差、没兴趣、觉得活着没意思，诉起病前曾有近1年情绪不稳定，有时有较为高兴的时候，很爱说话，很想加入同学的聊天；睡眠、饮食、二便大致正常，部分自知力。：意识清，定向力完整；接触、对答可，目前未引出持续感知觉障碍，未引出思维形式及内容障碍；诉自起病起情绪持续不稳定，起初为持续心情差，没兴趣，高兴不起来，近半年则以情绪不稳定突出，一天之内情绪像做过山车，好的时候充满干劲，认真听课、认真做作业，没有原因的就想笑，不好的时候悲伤，觉得自己做什么都不行，对既往2次自杀行为，诉第一次跳楼是跟母亲吵架后冲动下出现的，第二次烧炭自杀是计划了几天，就算不跟母亲吵架，自己也会落实，但对2次行为均不感到后悔；最近2周仍是情绪低落为主，没办法继续上课，觉得活着没意思，在病房内多独处，卧床睡觉，一般情况食大致正常，部分自知力。\n入院日期：2022年02月15日\n"
    # 双相情感障碍,目前为不伴有精神病性症状的重度抑郁发作

    test_case3 = "阅读下面的患者病历信息，诊断患者可能的精神疾病，并分析诊断依据，同时完成鉴别诊断。患者病历信息：患者男性，19岁。\n现病史:（部分病史家属回忆不清）2020年10月患者无明显诱因下出现情绪差，没兴趣，心烦，后逐渐出现凭空闻声，看不见人的时候可以听见声音，听到路上有人说“你活着有什么用”等；敏感多疑，认为有人在监视自己，周围人在嘲笑自己、议论自己，自己不管做什么都会被别人知道，情绪差加重，不断哭泣，自残，遂休学，于当地治疗，诊断不详，予舍曲林、奥氮平、利培酮等药物治疗（具体剂量不详），服药后2月中体重增长15kg，病情无好转。同年底，患者跟母亲争吵后跳楼，导致左锁骨、右腓骨远端、跟骨骨折，L4压缩性骨折，外固定后好转，未遗留肢体活动不利。2021-01-27患者骨折等好转后，因仍情绪差及凭空闻声，于重庆医科大学附属医院治疗，诊断“抑郁发作”，予草酸艾司西酞普兰20mg qd+阿立哌唑 10mg qd治疗，情绪较前好转，凭空闻声频率减少，好转出院。出院后患者规律服药，生活基本正常，可自理，在家休养数月后，同年8月复学，复学后可考年级前10。10月起患者再次出现情绪差，看到老师、同学都紧张害怕、心慌，在学校多次自残自伤，凭空闻声，敏感多疑，在老师建议下遂再次休学，休学在家玩手机，与家人交流少，反复向母亲要钱，不给钱就以死相逼，遂于北医六院、当地医院等治疗，诊断不详，调整治疗为布南色林8mg bid、伏硫西汀10mg qd、文拉法辛75mg qd、喹硫平200mg qd，规律用药，病情无明显改善。 2022年1月，患者与母亲争吵后，在家烧炭自杀，未昏迷，患者家属报警，警察来后，患者又企图跳楼，被阻止，未诊治。2月患者母亲给患者口服中药及朱砂（每日约3g）治疗，02-11首次就诊于我院，完善CT提示，胃内多发不规则状高密度影，遂就诊于北大第一医院，行腹部CT检查，阑尾、结直肠内散在点状高密度，结合临床，家属表示医生考虑进一步观察即可，遂02-15于在家属陪同下来我院就诊，急诊以“抑郁状态”第1次收入院。本次为非自愿入院。入院前2周，患者饮食、睡眠、二便无殊、体重无明显变化。2022-02-08患者着凉后发热，Tmax38.8℃，服用连花清瘟胶囊后退热，未再发热，否认咳嗽、腹泻表现。患者为非自愿，家属已签署非自愿相关协议书。\n既往史：左锁骨、右腓骨远端、跟骨骨折，L4压缩性骨折后；\n家族史：无殊\n查体，辅助检查及精神检查：无殊；\n  8.精神检查：意识清，定向力完整；主被动接触可，对答切题，语速语量适中，承认既往凭空闻声，内容多为嘲笑自己，现在已几乎消失；承认既往关系妄想，认为周围人都在议论自己等，目前已消失；诉持续的情绪差、没兴趣、觉得活着没意思；睡眠、饮食、二便大致正常，部分自知力。：意识清，定向力完整；接触、对答可，目前未引出持续感知觉障碍，未引出思维形式及内容障碍；觉得自己做什么都不行，有时心情不好的时候会出现幻听，大约1周会出现1次，内容为声音说自己不行；对既往2次自杀行为，诉第一次跳楼是跟母亲吵架后冲动下出现的，第二次烧炭自杀是计划了几天，就算不跟母亲吵架，自己也会落实，但对2次行为均不感到后悔；最近2周仍是情绪低落为主，没办法继续上课，觉得活着没意思，在病房内多独处，卧床睡觉，一般情况食大致正常，部分自知力。\n入院日期：2022年02月15日\n"
    #复发性抑郁障碍,目前为伴有精神病性症状的重度发作。

    test_case4 = "阅读下面的患者病历信息，诊断患者可能的精神疾病，并分析诊断依据，同时完成鉴别诊断。患者病历信息：患者男性，19岁。\n现病史:（部分病史家属回忆不清）2022年1月，患者无明显诱因下出现情绪差，没兴趣，心烦，后逐渐出现凭空闻声，看不见人的时候可以听见声音，听到路上有人说“你活着有什么用”等；敏感多疑，认为有人在监视自己，周围人在嘲笑自己、议论自己，自己不管做什么都会被别人知道，情绪差加重，不断哭泣，自残，遂休学，于当地治疗，诊断不详，予舍曲林、奥氮平、利培酮等药物治疗（具体剂量不详），服药后2月中体重增长15kg，病情无好转。某次患者与母亲争吵后，在家烧炭自杀，未昏迷，患者家属报警，警察来后，患者又企图跳楼，被阻止，未诊治。2月患者母亲给患者口服中药及朱砂（每日约3g）治疗，02-11首次就诊于我院，完善CT提示，胃内多发不规则状高密度影，遂就诊于北大第一医院，行腹部CT检查，阑尾、结直肠内散在点状高密度，结合临床，家属表示医生考虑进一步观察即可，遂02-15于在家属陪同下来我院就诊，急诊以“抑郁状态”第1次收入院。本次为非自愿入院。入院前2周，患者饮食、睡眠、二便无殊、体重无明显变化。2022-02-08患者着凉后发热，Tmax38.8℃，服用连花清瘟胶囊后退热，未再发热，否认咳嗽、腹泻表现。患者为非自愿，家属已签署非自愿相关协议书。\n既往史：左锁骨、右腓骨远端、跟骨骨折，L4压缩性骨折后；\n家族史：无殊\n查体，辅助检查及精神检查：无殊；\n  8.精神检查：意识清，定向力完整；主被动接触可，对答切题，语速语量适中，承认既往凭空闻声，内容多为嘲笑自己，现在已几乎消失；承认既往关系妄想，认为周围人都在议论自己等，目前已消失；诉持续的情绪差、没兴趣、觉得活着没意思；睡眠、饮食、二便大致正常，部分自知力。：意识清，定向力完整；接触、对答可，目前未引出持续感知觉障碍，未引出思维形式及内容障碍；觉得自己做什么都不行，有时心情不好的时候会出现幻听，大约1周会出现1次，内容为声音说自己不行；对既往2次自杀行为，诉第一次跳楼是跟母亲吵架后冲动下出现的，第二次烧炭自杀是计划了几天，就算不跟母亲吵架，自己也会落实，但对2次行为均不感到后悔；最近2周仍是情绪低落为主，没办法继续上课，觉得活着没意思，在病房内多独处，卧床睡觉，一般情况食大致正常，部分自知力。\n入院日期：2022年02月15日\n"
    #伴有精神病性症状的重度抑郁发作。

    
    instruction = f"阅读下面的患者病历信息，诊断患者可能的精神疾病，并分析诊断依据，同时完成鉴别诊断。患者病历信息：{test_case}"

    psychgpt_r1_ans = psychgpt_r1_api.chat(instruction, stream=False)
    psychgpt_r1_diag = re.search(r"<answer>(.*?)</answer>", psychgpt_r1_ans, re.DOTALL).group(1).strip()

    conversation = []
    conversation.append({'from': 'human', 'value': input_info})
    conversation.append({'from': 'gpt', 'value': psychgpt_r1_ans})
    conversations = {"conversations": conversation}
    
    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/ablation/psychgpt-r1_vs_psychgpt_task2_1.jsonl', 'a') as f2:
        json_str = json.dumps(conversations, ensure_ascii=False)
        f2.write(json_str + '\n')
        f2.flush()

# ablation_reaction()

def ablation_erroranalysis():
    '''
    对模型在诊断任务上的错误进行error analysis
    '''

    path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/ablation/psychgpt-r1_vs_psychgpt_task2.jsonl'
    data1 = []
    with open(path1, 'r', encoding='utf-8') as f1:
        for line in f1:
            newline = eval(line)
            data1.append(newline)
    f1.close()

    path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_refine_0314/task2.jsonl'

    data = []
    with open(path, 'r', encoding='utf-8') as f1:
        for line in f1:
            newline = eval(line)
            data.append(newline)
    f1.close()

    print(len(data))
    psychgpt_r1_cnt = []
    psychgpt_cnt = []
    cnt = 0
    error_cases = []
    wrong_cases = []

    for item in tqdm(data):
        if data1[cnt]['conversations'][0]['value'] != item['conversations'][0]['value']:
            
            psychgpt_r1_ans = psychgpt_r1_api.chat(item['conversations'][0]['value'], stream=False)
            conversation = []
            conversation.append({'from': 'human', 'value': item['conversations'][0]['value']})
            conversation.append({'from': 'gpt', 'value': item['conversations'][1]['value']})
            psychgpt_r1 = {"from": "gpt", "value": psychgpt_r1_ans}

            conversations = {"conversations": conversation, "psychgpt_r1": psychgpt_r1}
            
            with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/ablation/error_analysis_diagnosis.jsonl', 'a') as f2:
                json_str = json.dumps(conversations, ensure_ascii=False)
                f2.write(json_str + '\n')
                f2.flush()
            
            continue

        try:
            input_info = item['conversations'][0]['value']
            output_ori = item['conversations'][1]['value']
            gt = re.search(r"<answer>(.*?)</answer>", output_ori, re.DOTALL).group(1).strip()
            try:
                gt = re.search(r"主要诊断：(.*?)\n", gt, re.DOTALL).group(1).strip()
            except:
                gt = re.search(r"主要诊断：(.*)", gt, re.DOTALL).group(1).strip()

            try:
                gt = code_convert[gt.split(' ')[1]]
            except:
                pass

            psychgpt_r1_ans = data1[cnt]['psychgpt_r1']['value']
            psychgpt_r1_diag = re.search(r"<answer>(.*?)</answer>", psychgpt_r1_ans, re.DOTALL).group(1).strip()

            try:
                psychgpt_r1_diag = re.search(r"主要诊断：(.*?)\n", psychgpt_r1_diag, re.DOTALL).group(1).strip()
            except:
                psychgpt_r1_diag = re.search(r"主要诊断：(.*)", psychgpt_r1_diag, re.DOTALL).group(1).strip()

            try:
                psychgpt_r1_diag = code_convert[psychgpt_r1_diag.split(' ')[1]]
            except:
                pass

            if gt != psychgpt_r1_diag:
                conversation = []
                conversation.append({'from': 'human', 'value': input_info})
                conversation.append({'from': 'gpt', 'value': gt})
                psychgpt_r1 = {"from": "gpt", "value": psychgpt_r1_ans}

                conversations = {"conversations": conversation, "psychgpt_r1": psychgpt_r1}
                
                with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/ablation/error_analysis_diagnosis.jsonl', 'a') as f2:
                    json_str = json.dumps(conversations, ensure_ascii=False)
                    f2.write(json_str + '\n')
                    f2.flush()
            cnt +=1
        except:
            cnt +=1
            print(f"Error on: {cnt}")

# ablation_erroranalysis()

def test_psychbench_task3():
    path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/psychbench-0219/0shot/task4_deepseek-r1.json'

    with open(path, 'r') as file:
        data = json.load(file)
    file.close()

    print(len(data))
    psychgpt_r1_cnt = []
    psychgpt_cnt = []
    cnt = -1
    error_cases = []
    wrong_cases = []

    for item in tqdm(data):
        cnt +=1

        input_info = item['query']
        output_ori = item['conversations'][1]['value']
        gt = output_ori

        psychgpt_r1_ans = psychgpt_r1_api.chat(input_info, stream=False)

        try:
            psychgpt_r1_diff = re.search(r"<answer>(.*?)</answer>", psychgpt_r1_ans, re.DOTALL).group(1).strip()
        except:
            psychgpt_r1_diff = psychgpt_r1_ans

        print(gt)
        print(psychgpt_r1_diff)

        conversation = []
        conversation.append({'from': 'human', 'value': input_info})
        conversation.append({'from': 'gpt', 'value': gt})
        psychgpt_r1 = {"from": "gpt", "value": psychgpt_r1_ans}
        conversations = {"conversations": conversation, "psychgpt_r1": psychgpt_r1}
        
        with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/ablation/psychgpt-r1_psychbench_task4.jsonl', 'a') as f2:
            json_str = json.dumps(conversations, ensure_ascii=False)
            f2.write(json_str + '\n')
            f2.flush()
    

test_psychbench_task3()