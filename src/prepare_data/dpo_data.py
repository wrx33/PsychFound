import json
import re
import pandas as pd
import os
from tqdm import tqdm
from openai import OpenAI
import random
from copy import deepcopy

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

api_qwen = qwen2_api()

# 使用DPO对SFT后效果仍需提升的任务进行优化
# 1. 优化生成的首程中的简要病史部分过长，基本没有缩写的问题
# 2. 优化生成的首程中的入院印象包含不存在的内容的问题

# path = '/home/sjtu/anding_data/F30-F33/患者信息.xlsx'
# df = pd.read_excel(path)
# disease_list = df['出院主要诊断']
# disease_list = list(set(disease_list.to_list()))
# disease_list.extend(['妄想状态', '强迫状态', '缄默状态'])


# path2 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task6-write_firstcourse_2x.jsonl'
# path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task6-write_firstcourse_3x.jsonl'

# data = []
# with open(path1, 'r') as f1:
#     for line in f1:
#         newline = eval(line)
#         data.append(newline)
# f1.close()

# cnt = 0
# for item in tqdm(data):
#     input_info = item['conversations'][0]['value']
#     output = item['conversations'][1]['value']

#     preliminary_diganosis = re.search(r"5.入院印象：(.*?)鉴别诊断", output, re.DOTALL).group(1)
#     preliminary_diganosis_list = re.split(':|：|，|,| ', preliminary_diganosis)
#     preliminary_diganosis_mental_list = [pdm for pdm in preliminary_diganosis_list if pdm.strip('\n,，.。') in disease_list]


#     if len(preliminary_diganosis_mental_list) == 0 or len(preliminary_diganosis_list) == len(preliminary_diganosis_mental_list):
#       continue
    
#     preliminary_diganosis_mental = '，'.join(preliminary_diganosis_mental_list)+'\n'

#     briefhis = re.search(r"4.临床表现(.*?)5.既往史", output, re.DOTALL).group(1)[1:]
#     currenthis = re.split('\n', input_info)[2].replace("现病史:", "")

#     if len(briefhis) / len(currenthis) > 0.4:
#         continue

#     conversation = []
#     conversation.append({'from': 'human', 'value': input_info})

#     # chosen = {"from": "gpt", "value": output}
#     # rejected = {"from": "gpt", "value": output.replace(briefhis, currenthis)}

#     # conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}

#     # conversation = []
#     # conversation.append({'from': 'human', 'value': input_info})

#     chosen = {"from": "gpt", "value": output.replace(preliminary_diganosis, preliminary_diganosis_mental)}
#     rejected = {"from": "gpt", "value": output.replace(briefhis, currenthis)}

#     conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
    
#     if cnt >= 2000:
#       break
#     cnt += 1
#     with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0213/firstcourse_preliminary_brief_2000.jsonl', 'a') as f2:
#         json_str = json.dumps(conversations, ensure_ascii=False)
#         f2.write(json_str + '\n')
#         f2.flush()

# 3. 优化生成的诊断和鉴别诊断分析中鉴别诊断项目较少的问题

# prompt_depression_differential_diagnosis = f"""## 鉴别诊断：
# 1. 双相障碍：双相障碍存在躁狂/轻躁狂发作史，其核心特征为“不稳定性”。双相障碍患者多以抑郁发作起病，可能多次抑郁发作后才出现躁狂/轻躁狂发作，因此早期识别提示可能为双相障碍的线索非常重要，如青少年起病、情感旺盛人格、抑郁发作频繁且好转速度快、伴精神病性特征、不典型特征或混合特征、难治性抑郁、产后抑郁、季节性抑郁、共病物质滥用或边缘性人格障碍、双相障碍家族史等。

# 2. 焦虑障碍：抑郁和焦虑常同时出现，抑郁障碍的核心症状为“心境低落”，焦虑障碍则多表现为过度的“紧张、恐惧、担忧”等，常伴有明显的躯体焦虑症状。

# 3. 创伤后应激障碍：发生于极其严重创伤性事件后的 6 个月内，其典型症状为反复出现的“闪回”、回避创伤相关情境、情感疏远、麻木感等，情感改变多为焦虑、痛苦、易激惹，波动性大。

# 4. 精神分裂症：鉴别点主要包括原发症状多为思维障碍或感知觉障碍，病程多迁延而非间歇性，精神活动缺乏协调性。出现的抑郁症状为继发，且短于原发症状。

# 5. 躯体疾病相关抑郁：不少躯体疾病可伴发或导致抑郁障碍（如心血管系统疾病，呼吸系统疾病，肾脏疾病，消化系统疾病，内分泌系统疾病，血液系统疾病，风湿免疫类系统疾病等），此时抑郁与躯体状况之间的关系可以是：躯体疾病是抑郁障碍的直接原因（生理性）；躯体疾病时抑郁障碍发生的诱因（心理性）；躯体疾病与抑郁障碍伴发，没有直接的因果关系，但两者相互促进；抑郁障碍是躯体疾病的直接原因。

# 6. 神经系统疾病：帕金森，痴呆性疾病，癫痫，脑血管病和肿瘤，这些神经系统疾病易导致抑郁。

# 7. 药物所致抑郁障碍：临床上有时很难区分抑郁症状究竟是原发还是继发于躯体疾病或是由于治疗躯体疾病的药物所致。此时需要详细询问病史和用药史，进行必要的停药观察。

# """

# prompt_mania_differential_diagnosis = f"""## 鉴别诊断
# 1. 抑郁障碍（单相抑郁障碍）：抑郁障碍指只有抑郁发作、而无确切躁狂或轻躁狂发作史的心境障碍。大部分双相障碍患者首次心境发作通常是抑郁，在未发现躁狂或轻躁狂发作史时，将抑郁发作患者诊断为抑郁障碍符合诊断原则，虽然部分患者在之后改诊为双相障碍。目前诊断标准未区分抑郁障碍与双相障碍的抑郁发作，但二者的临床特征存在差异：双相障碍患者抑郁往往发作频繁、急性起病或快速缓解、首发年龄小（通常小于20 岁），具有情感波动性、伴精神病性症状、非典型症状、激越、自伤、共病、双相障碍家族史等。

# 2. 器质性精神障碍：某些躯体或脑部疾病（如甲状腺功能异常、脑外伤或肿瘤、癫痫等）及药物（如皮质醇、抗结核药及抗肿瘤药等）可导致患者出现情感症状。

# 3. 精神活性物质所致精神障碍：精神活性物质可诱发抑郁、轻躁狂甚至躁狂症状，该病与双相障碍关系复杂，二者有很高的共病率。鉴别主要依据病史、精神活性物质定性及体格检查（可有阳性体征）。使用精神活性物质的患者出现心境发作需待戒除精神活性物质后再次评估其心境，若仍存在症状则可诊断双相障碍；相反，则考虑为精神活性物质所致。

# 4. 精神分裂症：双相障碍可伴有精神病性症状，常存在于心境发作期间，若心境发作缓解后精神病性症状随之消失，则诊断为双相障碍伴精神病性症状；相反，应考虑为精神分裂症或分裂情感性精神病。此外，精神分裂症患者也可出现情感症状、甚或心境发作，但若心境发作不满足抑郁发作、躁狂发作或混合发作的诊断要求，则仍诊断为精神分裂症。

# 5. 人格障碍：情感不稳定性人格障碍容易与双相障碍相混淆，两者常共病。人格障碍常起病于儿童期或青春期早期，持续进展，而双相障碍多起病于青春期后期或成年初期，临床表现呈间歇性，心境稳定剂治疗有效，缓解期可基本恢复正常。若考虑人格障碍，采集病史时应仔细评估其成长及人际关系史等以资鉴别。

# 6. 焦虑障碍：有时，焦虑性反复思考会被人为是思维奔逸，需鉴别患者是否共病焦虑障碍，还是伴发焦虑症状。

# 7. 注意缺陷/多动障碍（ADHD）：ADHD一般起病于儿童期，而双相情感障碍起兵多在青少年期或青春期后。ADHD以注意力缺陷为主要特点，而双相情感障碍以情绪不稳定性为主要特点。ADHD发病无季节性，双相情感障碍有季节性波动的特点。

# """

# path2 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task2-diagnosis-f2x.jsonl'
# path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task2-diagnosis-f3x.jsonl'

# data = []
# with open(path1, 'r') as f1:
#     for line in f1:
#         newline = eval(line)
#         data.append(newline)
# f1.close()

# cnt = 0
# for item in tqdm(data):
#     input_info = item['conversations'][0]['value']
#     output = item['conversations'][1]['value']

#     continue_analysis = ""

#     if "抑郁" in output and "双相" not in output:
#         continue_analysis = api_qwen.chat(query=f"患者病历信息：{input_info}\n 医生给出的诊断分析：{output}。\n 请结合上述患者病历信息和医生做出的诊断分析，对下面的鉴别诊断项目中可能的疾病进行个体化分析，明显不可能的疾病可以不进行分析。若有其他的需鉴别的疾病也请做出分析。并以可能性大小进行排序。只输出鉴别分析相关内容即可，不要输出其他内容。 {prompt_depression_differential_diagnosis}", stream=False)
#     elif "双相" in output:
#         continue_analysis = api_qwen.chat(query=f"患者病历信息：{input_info}\n 医生给出的诊断分析：{output}。\n 请结合上述患者病历信息和医生做出的诊断分析，对下面的鉴别诊断项目中可能的疾病进行个体化分析，明显不可能的疾病可以不进行分析。若有其他的需鉴别的疾病也请做出分析。并以可能性大小进行排序。只输出鉴别分析相关内容即可，不要输出其他内容。  {prompt_mania_differential_diagnosis}", stream=False)
#     else:
#         continue_analysis = api_qwen.chat(query=f"患者病历信息：{input_info}\n 医生给出的诊断分析：{output}。\n 请结合上述患者病历信息和医生做出的诊断分析，进一步补充完善鉴别诊断分析，对该患者所有可能的鉴别诊断项进行分析（严格参考ICD-10临床诊断标准）。只输出鉴别分析相关内容即可，不要输出其他内容。", stream=False)

    

#     conversation = []
#     conversation.append({'from': 'human', 'value': input_info})

#     # chosen = {"from": "gpt", "value": output}
#     # rejected = {"from": "gpt", "value": output.replace(briefhis, currenthis)}

#     # conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}

#     # conversation = []
#     # conversation.append({'from': 'human', 'value': input_info})

#     chosen = {"from": "gpt", "value": output + '\n\n' + continue_analysis}
#     rejected = {"from": "gpt", "value": output}

#     conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
    
#     if cnt >= 1000:
#       break
#     cnt += 1
#     with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0213/diagnosis_1000.jsonl', 'a') as f2:
#         json_str = json.dumps(conversations, ensure_ascii=False)
#         f2.write(json_str + '\n')
#         f2.flush()


# 4. 优化输入信息中缺失检验检查结果等信息，但生成的首程因为幻觉虚拟出相关信息的问题
path = '/home/sjtu/anding_data/F30-F33/患者信息.xlsx'
df = pd.read_excel(path)
disease_list = df['出院主要诊断']
disease_list = list(set(disease_list.to_list()))
disease_list.extend(['妄想状态', '强迫状态', '缄默状态'])


path2 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task6-write_firstcourse_2x.jsonl'
path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task6-write_firstcourse_3x.jsonl'

data = []
with open(path1, 'r') as f1:
    for line in f1:
        newline = eval(line)
        data.append(newline)
f1.close()

cnt = 0
for item in tqdm(data):
    input_info = item['conversations'][0]['value']
    output = item['conversations'][1]['value']

    preliminary_diganosis = re.search(r"5.入院印象：(.*?)鉴别诊断", output, re.DOTALL).group(1)
    preliminary_diganosis_list = re.split(':|：|，|,| ', preliminary_diganosis)
    preliminary_diganosis_mental_list = [pdm for pdm in preliminary_diganosis_list if pdm.strip('\n,，.。') in disease_list]

    preliminary_diganosis_mental = '，'.join(preliminary_diganosis_mental_list)+'\n'

    briefhis = re.search(r"4.临床表现(.*?)5.既往史", output, re.DOTALL).group(1)[1:]
    currenthis = re.split('\n', input_info)[2].replace("现病史:", "")

    if len(preliminary_diganosis_mental_list) == 0 or len(preliminary_diganosis_list) == len(preliminary_diganosis_mental_list):
      continue

    if len(briefhis) / len(currenthis) > 0.4:
        continue

    pasthis_in = re.search(r"(既往史：.*?)家族史", input_info, re.DOTALL).group(1)
    familyhis_in = re.search(r"(家族史：.*?)查体，辅助检查及精神检查：", input_info, re.DOTALL).group(1)
    exam_in = re.search(r"(查体，辅助检查及精神检查：.*?)入院日期", input_info, re.DOTALL).group(1)

    pasthis_out = re.search(r"(5.既往史：.*?)6.家族史", output, re.DOTALL).group(1)
    familyhis_out = re.search(r"(6.家族史：.*?)7.查体及辅助检查：", output, re.DOTALL).group(1)
    exam_out = re.search(r"(7.查体及辅助检查：.*?)拟诊讨论", output, re.DOTALL).group(1)


    # chosen: in有，out有 / in无，out无

    # rejected: in有，out无 / in无，out有

    output_made = output.replace(preliminary_diganosis, preliminary_diganosis_mental)
    input_made = input_info
    if random.random() > 0.4:
      input_made = input_made.replace(exam_in, "")
      output_made = output_made.replace(exam_out, "7.查体及辅助检查：未提供。\n")
    
    if random.random() > 0.5:
      input_made = input_made.replace(familyhis_in, "")
      output_made = output_made.replace(familyhis_out, "6.家族史：未提供。\n")

    if random.random() > 0.5:
      input_made = input_made.replace(pasthis_in, "")
      output_made = output_made.replace(pasthis_out, "5.既往史：未提供。\n")

    conversation = []
    conversation.append({'from': 'human', 'value': input_made})
    chosen = {"from": "gpt", "value": output_made}
    rejected = {"from": "gpt", "value": output}
    conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
    
    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0213/firstcourse_missing_2000.jsonl', 'a') as f2:
        json_str = json.dumps(conversations, ensure_ascii=False)
        f2.write(json_str + '\n')
        f2.flush()
    
    # conversation = []
    # conversation.append({'from': 'human', 'value': input_info.replace(familyhis_in, "")})
    # chosen = {"from": "gpt", "value": output.replace(preliminary_diganosis, preliminary_diganosis_mental).replace(familyhis_out, "6.家族史：未提供。\n")}
    # rejected = {"from": "gpt", "value": output}
    # conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
    
    # with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0213/firstcourse_missing_2000.jsonl', 'a') as f2:
    #     json_str = json.dumps(conversations, ensure_ascii=False)
    #     f2.write(json_str + '\n')
    #     f2.flush()
    
    # conversation = []
    # conversation.append({'from': 'human', 'value': input_info.replace(pasthis_in, "")})
    # chosen = {"from": "gpt", "value": output.replace(preliminary_diganosis, preliminary_diganosis_mental).replace(pasthis_out, "5.既往史：未提供。\n")}
    # rejected = {"from": "gpt", "value": output}
    # conversations = {"conversations": conversation, "chosen": chosen, "rejected": rejected}
    
    
    # with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/dpo_0213/firstcourse_missing_2000.jsonl', 'a') as f2:
    #     json_str = json.dumps(conversations, ensure_ascii=False)
    #     f2.write(json_str + '\n')
    #     f2.flush()
    
    if cnt >= 2000:
      break
    cnt += 1