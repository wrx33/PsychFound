import os
import sys
import json
import re
import datetime
import ollama
from tqdm import tqdm
import openai

import pandas as pd
import numpy as np
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

dzz_basic_df = pd.read_excel("/home/sjtu/anding_data/dzz/20250522/dzz基本信息.xlsx")
syz_basic_df = pd.read_excel("/home/sjtu/anding_data/dzz/20250522/syz基本信息.xlsx")
# 2 问诊全面性 （专家打分）

print("="*25+"问诊全面性"+"="*25)

interview_content_dzz = dzz_basic_df['现病史'].tolist()
interview_score_dzz = []
for content in tqdm(interview_content_dzz):
    query = f"""请根据以下评分标准，对低年资医生通过向患者问诊后撰写的现病史内容的全面性进行评分。只输出最终评分，不要输出任何其他内容。评分标准如下：
    问诊内容应包括：
    1. 起病情况：年龄、诱因（社会心理因素/躯体因素/无诱因）、发病形式（急性/亚急性/慢性）。关键点：是否明确诱因性质（如具体生活事件）、发病时间节点（如精确到周/月）。
    2. 病程特点：总病程时长、波动性（发作性/持续性）、加重或缓解因素（如季节、应激事件）。关键点：是否描述症状演变趋势（如进行性加重或周期性缓解）。
    3. 临床表现：早期症状（首发症状、前驱症状）、核心症状（如幻觉、妄想、情感症状）。关键点：是否区分症状的严重程度（如对功能的影响）及症状群关联性。
    4. 鉴别诊断信息：阳性症状（支持主诊断的特征）、阴性症状（排除其他疾病的依据，如器质性疾病证据、物质滥用史）。关键点：是否主动询问躯体疾病史、家族史、物质使用史。
    5. 既往治疗史：既往诊断（具体诊断名称）、用药史（药物名称、剂量、疗程、疗效、副作用）。关键点：是否记录治疗依从性及患者对治疗的主观反馈。
    6. 社会功能评估：病后职业/学业、人际交往、日常生活能力变化（需量化，如缺勤天数、社交回避频率）。关键点：是否使用标准化工具（如GAF量表）或具体行为描述。
    7. 伴随症状：躯体症状（睡眠、食欲、躯体疼痛）、认知功能（注意力、记忆力）、自杀/自伤风险。关键点：是否评估生物学症状（如早醒）及风险行为细节。
    
    评分标准（5分制）：
    5分：7项均完整且包含细节（如诱因具体化、病程时间轴、症状量化描述、鉴别依据明确）。
    4分：7项完整，但≤2项缺乏细节（如仅记录“有诱因”未说明内容）。
    3分：1项关键信息缺失，或≥3项信息笼统。
    2分：2项关键信息缺失。
    1分：≥3项关键信息缺失，或遗漏核心模块（如未问诊自杀风险）。
    
    评分说明
    “关键信息缺失”定义：任一核心模块（1-7项）完全未涉及。
    “信息详细”标准：需包含具体描述而非笼统回答（如“因失业发病”优于“有诱因”）。
    鉴别诊断加分项：主动询问非精神科问题（如甲状腺功能、脑损伤史）额外加权。
    
    问诊后撰写的现病史：
    {content}
    
    """
    response = qwen_api.chat(query, stream=False)
    print(response)
    
    interview_score_dzz.append(response)
print(interview_score_dzz)

print("="*50)



interview_content_syz = syz_basic_df['现病史'].tolist()
interview_score_syz = []
for content in tqdm(interview_content_syz):
    query = f"""请根据以下评分标准，对低年资医生通过向患者问诊后撰写的现病史内容的全面性进行评分。只输出最终评分，不要输出任何其他内容。评分标准如下：
    问诊内容应包括：
    1. 起病情况：年龄、诱因（社会心理因素/躯体因素/无诱因）、发病形式（急性/亚急性/慢性）。关键点：是否明确诱因性质（如具体生活事件）、发病时间节点（如精确到周/月）。
    2. 病程特点：总病程时长、波动性（发作性/持续性）、加重或缓解因素（如季节、应激事件）。关键点：是否描述症状演变趋势（如进行性加重或周期性缓解）。
    3. 临床表现：早期症状（首发症状、前驱症状）、核心症状（如幻觉、妄想、情感症状）。关键点：是否区分症状的严重程度（如对功能的影响）及症状群关联性。
    4. 鉴别诊断信息：阳性症状（支持主诊断的特征）、阴性症状（排除其他疾病的依据，如器质性疾病证据、物质滥用史）。关键点：是否主动询问躯体疾病史、家族史、物质使用史。
    5. 既往治疗史：既往诊断（具体诊断名称）、用药史（药物名称、剂量、疗程、疗效、副作用）。关键点：是否记录治疗依从性及患者对治疗的主观反馈。
    6. 社会功能评估：病后职业/学业、人际交往、日常生活能力变化（需量化，如缺勤天数、社交回避频率）。关键点：是否使用标准化工具（如GAF量表）或具体行为描述。
    7. 伴随症状：躯体症状（睡眠、食欲、躯体疼痛）、认知功能（注意力、记忆力）、自杀/自伤风险。关键点：是否评估生物学症状（如早醒）及风险行为细节。
    
    评分标准（5分制）：
    5分：7项均完整且包含细节（如诱因具体化、病程时间轴、症状量化描述、鉴别依据明确）。
    4分：7项完整，但≤2项缺乏细节（如仅记录“有诱因”未说明内容）。
    3分：1项关键信息缺失，或≥3项信息笼统。
    2分：2项关键信息缺失。
    1分：≥3项关键信息缺失，或遗漏核心模块（如未问诊自杀风险）。
    
    评分说明
    “关键信息缺失”定义：任一核心模块（1-7项）完全未涉及。
    “信息详细”标准：需包含具体描述而非笼统回答（如“因失业发病”优于“有诱因”）。
    鉴别诊断加分项：主动询问非精神科问题（如甲状腺功能、脑损伤史）额外加权。
    
    问诊后撰写的现病史：
    {content}
    
    """
    response = qwen_api.chat(query, stream=False)
    print(response)
    
    interview_score_syz.append(response)
print(interview_score_syz)

print("="*50)