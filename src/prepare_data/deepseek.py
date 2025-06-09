import ollama
import re
import json
import os
from tqdm import tqdm
import requests
from openai import OpenAI

class psychgpt_api():
    def __init__(self):
        self.api_base_url = "http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000))
        self.client = OpenAI(
            api_key="{}".format(os.environ.get("API_KEY", "0")),
            base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
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


    def knowledge_chat(self, query, knowledge_base, threshold, top_k, history=None, stream=True, api="/knowledge_chat/completions"):
        # message = []
        # if history:
        #     message.extend(history)
        # else:
        #     message.append(
        #         {
        #             "role": "user",
        #             "content": query
        #         }   
        #     ) 
        # completion = self.client.knowledge_chat.completions.create(
        #     model='psychgpt',
        #     messages=message,
        #     stream=stream,
        # )
        
        # if stream:
        #     return completion
        # else:
        #     print(completion.choices[0].message.content)
        #     return completion.choices[0].message.content
        knowledge_chat_data = {
            "query": query,
            "knowledge_base_name": knowledge_base,
            "history": [
                # {
                #     "role": "user",
                #     "content": "你好"
                # },
                # {
                #     "role": "assistant",
                #     "content": "你好，我是 ChatGLM"
                # }
            ],
            "stream": True,
            "temperature": 0.1,
            "max_tokens": 1024,
            "model_name": 'AndingGPT-1.0',
            "score_threshold": threshold,
            "top_k": top_k,
            
        }
        url = f"{self.api_base_url}{api}"
        response = requests.post(url, headers=self.header, json=knowledge_chat_data, stream=True)

        content = response.json()
        print(content['Answer'])
        print(content['Reference'])
        return content

def deepseek_api(query):

    api_key = 'sk-3d8155d300fd4f938ac1f6de46c67432'
    url = 'http://192.168.10.180:3000/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

    payload = {
        "model": 'deepseek-r1',
        "messages": [
            {"role": "user", "content": query}
        ],
        "temperature": 0.7,
        "max_new_tokens": 1024,
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        result_content = result["choices"][0]["message"]["content"]
        print(result_content)
        return result_content
    else:
        print("ERROR:", response.status_code, response.text)
        return 0

api = psychgpt_api()
file_path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0307/firstcourse_missing_300.jsonl'

data = []
with open(file_path, 'r', encoding='utf-8') as f1:
    for line in f1:
        newline = eval(line)
        data.append(newline)
f1.close()

for item in tqdm(data):
    query = item['conversations'][0]['value']
    output = item['chosen']['value']

    psychgpt_ans = api.chat(query=query, stream=False)
    print(psychgpt_ans)

    conversation = []
    conversation.append({'from': 'human', 'value': query})
    chosen = {"from": "gpt", "value": output}
    rejected = {"from": "gpt", "value": psychgpt_ans}
    conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
    
    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0307/firstcourse_missing_300_refine.jsonl', 'a') as f2:
        json_str = json.dumps(conversations, ensure_ascii=False)
        f2.write(json_str + '\n')
        f2.flush()

# api = psychgpt_api()
# path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0307/firstcourse_missing_300.jsonl'
# file_list = os.listdir(path)
# for file in tqdm(file_list):
#     file_path = os.path.join(path, file)

#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f1:
#         for line in f1:
#             newline = eval(line)
#             data.append(newline)
#     f1.close()

#     for item in tqdm(data):
#         query = item['conversations'][0]['value']
#         output = item['conversations'][1]['value']

#         psychgpt_ans = api.chat(query=query, stream=False)
#         print(psychgpt_ans)

#         conversation = []
#         conversation.append({'from': 'human', 'value': query})
#         chosen = {"from": "gpt", "value": psychgpt_ans}
#         rejected = {"from": "gpt", "value": output}
#         conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
        
#         with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_A3/distillation_dpo_A3.jsonl', 'a') as f2:
#             json_str = json.dumps(conversations, ensure_ascii=False)
#             f2.write(json_str + '\n')
#             f2.flush()

# api = psychgpt_api()
# path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/A3_SFT'
# data = []
# with open(path, 'r', encoding='utf-8') as f1:
#     for line in f1:
#         newline = eval(line)
#         data.append(newline)
# f1.close()

# for item in tqdm(data):
#     query = item['conversations'][0]['value']
#     output_ori = item['rejected']
#     if len(output_ori['value']) == 2:
#         # deepseek_ans = item['chosen']
#         # deepseek_think = re.search(r'(<think>.*?</think>)', deepseek_ans['value'], re.DOTALL).group(1)
#         # psychgpt_ans = api.chat(query, stream=False)

#         conversation = []
#         conversation.append({'from': 'human', 'value': query})
#         chosen = {"from": "gpt", "value": item['chosen']['value']}
#         rejected = {"from": "gpt", "value": item['rejected']['value']['value']}
#         conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
        
#         with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0305/distillation_sft_0305.jsonl', 'a') as f2:
#             json_str = json.dumps(conversations, ensure_ascii=False)
#             f2.write(json_str + '\n')
#             f2.flush()


#     else:
#          with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0305/distillation_sft_0305.jsonl', 'a') as f2:
#             json_str = json.dumps(item, ensure_ascii=False)
#             f2.write(json_str + '\n')
#             f2.flush()



# # path1 = "/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_knowledge_1224/沈渔邨1.jsonl"
# # path2 = "/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_knowledge_1224/沈渔邨2.jsonl"
# # path3 = "/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_knowledge_1224/沈渔邨3.jsonl"
# # path4 = "/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_knowledge_1224/沈渔邨4.jsonl"
# # path5 = "/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_knowledge_1224/沈渔邨5.jsonl"

# ICD_LIST = [
#     "F00.0 阿尔茨海默病性痴呆，早发型",
#     "F00.1 阿尔茨海默病性痴呆，晚发型",
#     "F00.2 阿尔茨海默病性痴呆，非典型或混合型",
#     "F00.9 阿尔茨海默病性痴呆，未特指",
#     "F01.0 血管性痴呆，急性发作",
#     "F01.1 多发脑梗死性痴呆",
#     "F01.2 皮层下血管性痴呆",
#     "F01.3 混合型皮层和皮层下血管性痴呆",
#     "F01.8 其他血管性痴呆",
#     "F01.9 血管性痴呆，未特指",
#     "F02.0 匹克病性痴呆",
#     "F02.1 克雅病性痴呆",
#     "F02.2 亨廷顿病性痴呆",
#     "F02.3 帕金森病性痴呆",
#     "F02.4 人类免疫缺陷病毒[HIV]病性痴呆",
#     "F02.8 其他疾病分类中分类的其他疾病引起的痴呆",
#     "F03 未特指的痴呆",
#     "F04 器质性遗忘综合征，非由酒精和其他精神活性物质引起",
#     "F05.0 谵妄，非叠加于痴呆",
#     "F05.1 谵妄，叠加于痴呆",
#     "F05.8 其他谵妄",
#     "F05.9 谵妄，未特指",
#     "F06.0 器质性幻觉症",
#     "F06.1 器质性紧张性障碍",
#     "F06.2 器质性妄想性[精神分裂样]障碍",
#     "F06.3 器质性心境[情感]障碍",
#     "F06.4 器质性焦虑障碍",
#     "F06.5 器质性分离性障碍",
#     "F06.6 器质性情绪不稳定[衰弱]障碍",
#     "F06.7 轻度认知障碍",
#     "F06.8 由脑损害和功能紊乱及躯体疾病引起的其他精神障碍",
#     "F06.9 由脑损害和功能紊乱及躯体疾病引起的未特指的精神障碍",
#     "F07.0 器质性人格障碍",
#     "F07.1 脑炎后综合征",
#     "F07.2 脑震荡后综合征",
#     "F07.8 由脑疾病、脑损害和脑功能紊乱引起的其他人格和行为障碍",
#     "F07.9 由脑疾病、脑损害和脑功能紊乱引起的未特指的人格和行为障碍",
#     "F09 未特指的器质性或症状性精神障碍",
    
#     "F10.0 急性酒精中毒",
#     "F10.1 有害性酒精使用",
#     "F10.2 酒精依赖综合征",
#     "F10.3 酒精戒断状态",
#     "F10.4 酒精戒断状态伴谵妄",
#     "F10.5 酒精性精神病性障碍",
#     "F10.6 酒精性遗忘综合征[科尔萨科夫综合征]",
#     "F10.7 酒精性残留性和迟发性精神病性障碍",
#     "F10.8 其他酒精引起的精神和行为障碍",
#     "F10.9 酒精引起的精神和行为障碍，未特指",
#     "F11-F19 其他精神活性物质引起的精神和行为障碍（结构与F10类似，具体物质不同）",
    
#     "F20.0 偏执型精神分裂症",
#     "F20.1 青春型精神分裂症",
#     "F20.2 紧张型精神分裂症",
#     "F20.3 未分化型精神分裂症",
#     "F20.4 精神分裂症后抑郁",
#     "F20.5 残留型精神分裂症",
#     "F20.6 单纯型精神分裂症",
#     "F20.8 其他精神分裂症",
#     "F20.9 精神分裂症，未特指",
#     "F21 分裂型障碍",
#     "F22.0 妄想性障碍",
#     "F22.8 其他持续性妄想性障碍",
#     "F22.9 持续性妄想性障碍，未特指",
#     "F23.0 急性多形性精神病性障碍，不伴精神分裂症症状",
#     "F23.1 急性多形性精神病性障碍，伴精神分裂症症状",
#     "F23.2 急性精神分裂症样精神病性障碍",
#     "F23.3 其他急性以妄想为主的精神病性障碍",
#     "F23.8 其他急性短暂性精神病性障碍",
#     "F23.9 急性短暂性精神病性障碍，未特指",
#     "F24 感应性妄想性障碍",
#     "F25.0 分裂情感性障碍，躁狂型",
#     "F25.1 分裂情感性障碍，抑郁型",
#     "F25.2 分裂情感性障碍，混合型",
#     "F25.8 其他分裂情感性障碍",
#     "F25.9 分裂情感性障碍，未特指",
#     "F28 其他非器质性精神病性障碍",
#     "F29 未特指的非器质性精神病",
    
#     "F30.0 轻躁狂",
#     "F30.1 躁狂，不伴精神病性症状",
#     "F30.2 躁狂，伴精神病性症状",
#     "F30.8 其他躁狂发作",
#     "F30.9 躁狂发作，未特指",
#     "F31.0 双相情感障碍，当前为轻躁狂发作",
#     "F31.1 双相情感障碍，当前为不伴精神病性症状的躁狂发作",
#     "F31.2 双相情感障碍，当前为伴精神病性症状的躁狂发作",
#     "F31.3 双相情感障碍，当前为轻度或中度抑郁发作",
#     "F31.4 双相情感障碍，当前为不伴精神病性症状的重度抑郁发作",
#     "F31.5 双相情感障碍，当前为伴精神病性症状的重度抑郁发作",
#     "F31.6 双相情感障碍，当前为混合发作",
#     "F31.7 双相情感障碍，当前为缓解状态",
#     "F31.8 其他双相情感障碍",
#     "F31.9 双相情感障碍，未特指",
#     "F32.0 轻度抑郁发作",
#     "F32.1 中度抑郁发作",
#     "F32.2 重度抑郁发作，不伴精神病性症状",
#     "F32.3 重度抑郁发作，伴精神病性症状",
#     "F32.8 其他抑郁发作",
#     "F32.9 抑郁发作，未特指",
#     "F33.0 复发性抑郁障碍，当前为轻度发作",
#     "F33.1 复发性抑郁障碍，当前为中度发作",
#     "F33.2 复发性抑郁障碍，当前为不伴精神病性症状的重度发作",
#     "F33.3 复发性抑郁障碍，当前为伴精神病性症状的重度发作",
#     "F33.4 复发性抑郁障碍，当前为缓解状态",
#     "F33.8 其他复发性抑郁障碍",
#     "F33.9 复发性抑郁障碍，未特指",
#     "F34.0 环性心境",
#     "F34.1 恶劣心境",
#     "F34.8 其他持续性心境[情感]障碍",
#     "F34.9 持续性心境[情感]障碍，未特指",
#     "F38.0 其他单次发作的心境[情感]障碍",
#     "F38.1 其他复发性心境[情感]障碍",
#     "F38.8 其他特指的心境[情感]障碍",
#     "F39 未特指的心境[情感]障碍",
    
#     "F40.0 广场恐怖症",
#     "F40.1 社交恐怖症",
#     "F40.2 特定（孤立）恐怖症",
#     "F40.8 其他恐怖性焦虑障碍",
#     "F40.9 恐怖性焦虑障碍，未特指",
#     "F41.0 惊恐障碍[间歇性发作性焦虑]",
#     "F41.1 广泛性焦虑障碍",
#     "F41.2 混合性焦虑和抑郁障碍",
#     "F41.3 其他混合性焦虑障碍",
#     "F41.8 其他特指的焦虑障碍",
#     "F41.9 焦虑障碍，未特指",
#     "F42.0 以强迫思维或穷思竭虑为主",
#     "F42.1 以强迫动作[强迫仪式]为主",
#     "F42.2 混合性强迫思维和动作",
#     "F42.8 其他强迫性障碍",
#     "F42.9 强迫性障碍，未特指",
#     "F43.0 急性应激反应",
#     "F43.1 创伤后应激障碍",
#     "F43.2 适应障碍",
#     "F43.8 其他对严重应激的反应",
#     "F43.9 对严重应激的反应，未特指",
#     "F44.0 分离性遗忘",
#     "F44.1 分离性漫游",
#     "F44.2 分离性木僵",
#     "F44.3 出神与附体障碍",
#     "F44.4 分离性运动障碍",
#     "F44.5 分离性抽搐",
#     "F44.6 分离性感觉麻木和感觉丧失",
#     "F44.7 混合性分离[转换]性障碍",
#     "F44.8 其他分离[转换]性障碍",
#     "F44.9 分离[转换]性障碍，未特指",
#     "F45.0 躯体化障碍",
#     "F45.1 未分化性躯体形式障碍",
#     "F45.2 疑病障碍",
#     "F45.3 躯体形式植物神经功能紊乱",
#     "F45.4 持续性躯体形式疼痛障碍",
#     "F45.8 其他躯体形式障碍",
#     "F45.9 躯体形式障碍，未特指",
#     "F48.0 神经衰弱",
#     "F48.1 人格解体-现实解体综合征",
#     "F48.8 其他特指的神经症性障碍",
#     "F48.9 神经症性障碍，未特指",
    
#     "F50.0 神经性厌食",
#     "F50.1 非典型神经性厌食",
#     "F50.2 神经性贪食",
#     "F50.3 非典型神经性贪食",
#     "F50.4 与其他心理紊乱相关的暴食",
#     "F50.5 与其他心理紊乱相关的呕吐",
#     "F50.8 其他进食障碍",
#     "F50.9 进食障碍，未特指",
#     "F51.0 非器质性失眠症",
#     "F51.1 非器质性嗜睡症",
#     "F51.2 非器质性睡眠-觉醒节律障碍",
#     "F51.3 睡行症[夜游症]",
#     "F51.4 睡惊症[夜惊症]",
#     "F51.5 梦魇",
#     "F51.8 其他非器质性睡眠障碍",
#     "F51.9 非器质性睡眠障碍，未特指",
#     "F52.0 性欲缺乏或丧失",
#     "F52.1 性厌恶和性乐缺乏",
#     "F52.2 生殖器反应丧失",
#     "F52.3 性高潮功能障碍",
#     "F52.4 早泄",
#     "F52.5 非器质性阴道痉挛",
#     "F52.6 非器质性性交疼痛",
#     "F52.7 性欲亢进",
#     "F52.8 其他性功能障碍，非由器质性障碍或疾病引起",
#     "F52.9 未特指的性功能障碍，非由器质性障碍或疾病引起",
#     "F53.0 与产褥期有关的轻度精神及行为障碍，不可归类在他处者",
#     "F53.1 与产褥期有关的重度精神及行为障碍，不可归类在他处者",
#     "F53.8 其他与产褥期有关的精神及行为障碍，不可归类在他处者",
#     "F53.9 与产褥期有关的精神及行为障碍，未特指",
#     "F54 在它处分类的障碍及疾病伴有的心理及行为因素",
#     "F55 非依赖性物质滥用",
#     "F59 伴有生理紊乱和躯体因素的未特指的行为综合征",
    
#     "F60.0 偏执型人格障碍",
#     "F60.1 分裂样人格障碍",
#     "F60.2 社交紊乱型人格障碍",
#     "F60.3 情绪不稳型人格障碍",
#     "F60.4 表演型人格障碍",
#     "F60.5 强迫型人格障碍",
#     "F60.6 焦虑（回避）型人格障碍",
#     "F60.7 依赖型人格障碍",
#     "F60.8 其他人格障碍",
#     "F60.9 人格障碍，未特指",
#     "F61 混合型和其他人格障碍",
#     "F62.0 持久的人格改变，灾难性经历后",
#     "F62.1 持久的人格改变，精神科疾病后",
#     "F62.8 其他持久的人格改变",
#     "F62.9 持久的人格改变，未特指",
#     "F63.0 病理性赌博",
#     "F63.1 病理性纵火[纵火癖]",
#     "F63.2 病理性偷窃[偷窃癖]",
#     "F63.3 拔毛癖",
#     "F63.8 其他习惯和冲动障碍",
#     "F63.9 习惯和冲动障碍，未特指",
#     "F64.0 易性症",
#     "F64.1 双重异装症",
#     "F64.2 童年性身份障碍",
#     "F64.8 其他性身份障碍",
#     "F64.9 性身份障碍，未特指",
#     "F65.0 恋物症",
#     "F65.1 恋物性异装症",
#     "F65.2 露阴症",
#     "F65.3 窥阴症",
#     "F65.4 恋童症",
#     "F65.5 施虐受虐症",
#     "F65.6 多种性偏好障碍",
#     "F65.8 其他性偏好障碍",
#     "F65.9 性偏好障碍，未特指",
#     "F66.0 性成熟障碍",
#     "F66.1 自我不和谐的性取向",
#     "F66.2 性关系障碍",
#     "F66.8 其他与性发育和性取向有关的心理及行为障碍",
#     "F66.9 与性发育和性取向有关的心理及行为障碍，未特指",
#     "F68.0 出于心理原因夸大躯体症状",
#     "F68.1 有意制造或伪装症状或残疾[人为障碍]",
#     "F68.8 其他特指的人格和行为障碍",
#     "F69 未特指的成人人格和行为障碍",
    
# ]



# for item in tqdm(ICD_LIST):
#     prompt = f"按照ICD-10，'{item}'的诊断标准是什么？"

#     response = ollama.chat(model='deepseek-r1:14b', messages=[
#         {
#             'role': 'user',
#             'content': prompt,
#         },
#     ],
#     )
#     print(response['message']['content'])

#     deepseek_ans = response['message']['content']
#     deepseek_ans_think = re.search(r"(<think>.*?</think>)", deepseek_ans, re.DOTALL).group(1)
#     deepseek_ans_wo_think = deepseek_ans.replace(deepseek_ans_think, "")

#     conversation = []
#     conversation.append({'from': 'human', 'value': prompt})
#     chosen = {"from": "gpt", "value": deepseek_ans}
#     rejected = {"from": "gpt", "value": deepseek_ans_wo_think}
#     conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}

#     with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0213/diagnosis_criteria_deepseek.jsonl', 'a') as f2:
#         json_str = json.dumps(conversations, ensure_ascii=False)
#         f2.write(json_str + '\n')
#         f2.flush()





# # data = []
# # for path in [path2, path3, path4, path5, path1]:
# #     with open(path, 'r', encoding='utf-8') as f1:
# #         for line in f1:
# #             newline = eval(line)
# #             data.append(newline)
# #     f1.close()

# # for item in tqdm(data[::5]):
# #     question = item['conversations'][0]['value']
# #     ori_ans = item['conversations'][1]['value']

# #     response = ollama.chat(model='deepseek-r1:14b', messages=[
# #         {
# #             'role': 'user',
# #             'content': question,
# #         },
# #     ],
# #     )
# #     print(response['message']['content'])
  
# #     deepseek_ans = response['message']['content']

# #     conversation = []
# #     conversation.append({'from': 'human', 'value': question})
# #     chosen = {"from": "gpt", "value": deepseek_ans}
# #     rejected = {"from": "gpt", "value": ori_ans}
# #     conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}

# #     with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0213/QA_deepseek_vs_qwen.jsonl', 'a') as f2:
# #         json_str = json.dumps(conversations, ensure_ascii=False)
# #         f2.write(json_str + '\n')
# #         f2.flush()


# # 使用deepseek-r1 构建一批任务相关的理论基础知识数据，用于对模型进行相关知识注入


# # response = ollama.chat(model='deepseek-r1:14b', messages=[
# #     {
# #         'role': 'user',
# #         'content': "请列出ICD-10中，F谱系所包含的所有疾病，按顺序列出。"
# #     },
# # ],
# # )
# # print(response['message']['content'])
