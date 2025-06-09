import pandas as pd
import json
import re
import random
import os
import numpy as np
import ollama
from tqdm import tqdm

path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/sft_0113/task5-management-f3x-instruction.jsonl'

data = []
with open(path1, 'r') as f1:
    for line in f1:
        newline = eval(line)
        data.append(newline)
f1.close()

path_f3x_bc = "/home/sjtu/wrx/data/F30-F33病程_匿名化.xlsx"

path_f2x_bc = "/home/sjtu/wrx/data/F20-F29病程_匿名化.xlsx"





# F30-F33
df_bc_f3x = pd.read_excel(path_f3x_bc)

conversations = []
cnt_data = 0
for ipid in tqdm(df_bc_f3x['IPID'].unique()):
    try:
        if random.random() > 0.5:
            continue
        df_ipid = df_bc_f3x[df_bc_f3x['IPID'] == ipid]
        if df_ipid.shape[0] == 0:
            continue
        
        records = df_ipid[df_ipid['BL_TYPE'].isin(['主治医师查房记录', '主任医师查房记录', '副主任医师查房记录'])]
        records = records.sort_values(by='YZTL')
        records = records.drop_duplicates(subset=['YZTL'], keep='last')
        
        conversation = []
        cnt = 0
        for idx, record in records.iterrows():
            if cnt > 5:
                break
            
            record_content = record['YZTL']
            
            instruction = f"""
            请将以下精神疾病患者的住院病程记录按照以下两部分进行整理和提取：

            患者的病情进展：包括患者的不良反应、体格检查、精神检查、辅助检查（如化验检查、影像学检查）等内容。整理时保持记录的原始连续文本，不进行总结或概括。
            医生的临床分析和决策：包括医生对患者病情和检查结果的分析、下一步诊疗计划等内容。整理时保持记录的原始连续文本，不进行总结或概括。
            以下是具体的示例：

            示例 1
            输入：
            精神检查：神清，定向力完整，表情烦躁，愁苦，患者今晨突然进入兴奋室，躺于床上，表情痛苦，询问其原因，患者手扶胸口，不耐烦地大喊：“我难受着呢。”情绪激动，测生命体征及血糖均正常。要求医生给其打针，劝说安抚不能听从，自知力部分，情感反应协调。
            化验结果：2021-07-13 帕罗西汀血药浓度(样本:血清):脱烷基喹硫平浓度<10ng/ml,帕罗西汀浓度24.24ng/ml,喹硫平浓度80.95ng/ml↓2021-07-12 尿有形成分分析+尿常规(样本:尿):大致正常。2021-07-12 全血细胞分析（静脉血）(样本:全血):白细胞计数3.510^9/L↓,中性粒细胞数1.76*10^9/L↓,患者白细胞数及中性粒细胞数有所减少，已服用升白药物治疗，定期复查。2021-07-12 生化26项(样本:血清):*白蛋白39.3g/L↓,*高密度脂蛋白1.15mmol/L↓,嘱患者高蛋白低脂饮食，适当运动，定期监测。
            下一步诊疗计划：不排除患者焦虑与减帕罗西汀有关，今日将帕罗西汀加至30mg/d，继续观察，结合心理治疗。

            输出：
            患者的病情进展：
            精神检查：神清，定向力完整，表情烦躁、愁苦。患者今晨突然进入兴奋室，躺于床上，表情痛苦。询问其原因，患者手扶胸口，不耐烦地大喊：“我难受着呢。”情绪激动。测量生命体征及血糖均正常。劝说安抚无效，自知力部分存在，情感反应协调。
            化验结果：2021-07-13 帕罗西汀血药浓度：脱烷基喹硫平浓度<10ng/ml，帕罗西汀浓度24.24ng/ml，喹硫平浓度80.95ng/ml（↓）。2021-07-12 尿有形成分分析+尿常规：大致正常。2021-07-12 全血细胞分析（静脉血）：白细胞计数3.5×10^9/L（↓），中性粒细胞数1.76×10^9/L（↓）。患者白细胞数及中性粒细胞数有所减少，已服用升白药物治疗，定期复查。2021-07-12 生化26项：白蛋白39.3g/L（↓），高密度脂蛋白1.15mmol/L（↓）。

            医生的临床分析和决策：
            临床分析：不排除患者焦虑与减帕罗西汀有关。患者出现烦躁、痛苦表现，需结合病情及化验结果综合分析。白细胞数及中性粒细胞数减少，已采取升白药物治疗措施，饮食建议为高蛋白低脂，建议适当运动并定期监测。
            下一步诊疗计划：今日将帕罗西汀剂量加至30mg/d，继续观察患者状态并结合心理治疗。

            示例 2
            输入：
            精神检查：神清，定向力完整，接触可，自诉将药物加量后情绪稍有改善，中午仍心烦，每日要求服用一片奥沙西泮，下午状态较好，能打牌，聊天，询问医生：“我什么时候能好，能出院，你们多好，干什么都正常，我还是懒得动。”自知力部分，情感反应协调。
            化验结果：2021-05-25 生化26项(样本:血清):*白蛋白37.4g/L↓,*高密度脂蛋白1.02mmol/L↓,氯110.1mmol/L↑,继续观察，定期复查。2021-05-25 全血细胞分析（静脉血）(样本:全血):白细胞计数3.410^9/L↓,中性粒细胞数1.8710^9/L↓,患者白细胞及中性粒细胞数减少。目前服用磷酸腺嘌呤对症处理，继续观察，择日复查。2021-05-25 尿有形成分分析+尿常规(样本:尿):PH值5.0↓,白细胞40/ul↑,患者无尿路刺激征，考虑标本污染，择日复查。
            下一步诊疗计划：根据患者监测血压，晨起血压正常上限，最高142/82mmHg，考虑今日将早晨酒石酸美托洛尔片加量至25mg/d,维持目前治疗方案。

            输出：
            患者的病情进展：
            精神检查：神清，定向力完整，接触可。患者自述加量药物后情绪稍有改善，但中午仍感到心烦，每日要求服用一片奥沙西泮。下午状态较好，能够打牌、聊天，并询问医生：“我什么时候能好，能出院，你们多好，干什么都正常，我还是懒得动。”自知力部分存在，情感反应协调。
            化验结果：2021-05-25 生化26项：白蛋白37.4g/L（↓），高密度脂蛋白1.02mmol/L（↓），氯110.1mmol/L（↑），嘱患者继续观察并定期复查。2021-05-25 全血细胞分析（静脉血）：白细胞计数3.4×10^9/L（↓），中性粒细胞数1.87×10^9/L（↓）。患者白细胞及中性粒细胞数减少，正在服用磷酸腺嘌呤对症处理，继续观察并择日复查。2021-05-25 尿有形成分分析+尿常规：PH值5.0（↓），白细胞40/μl（↑）。患者无尿路刺激征，考虑标本污染，择日复查。

            医生的临床分析和决策：
            临床分析：患者自述加药后情绪有所改善，但中午仍感心烦，提示焦虑情绪可能未完全缓解。化验检查显示白细胞及中性粒细胞数减少，尿液检查标本可能存在污染，但需择日复查以确认异常情况。血压监测晨起值接近正常上限。
            下一步诊疗计划：今日将早晨酒石酸美托洛尔片加量至25mg/d，以控制血压，维持目前的治疗方案，继续观察患者的情绪与生理指标变化。

            请按照以上格式完成任务。
            输入：
            {record_content}

            输出：

            """
            
            item_data = data[cnt_data]['conversations'][2*cnt]['value'] + '\n' + data[cnt_data]['conversations'][2*cnt+1]['value']
            
            print("=="*50)
            print(record_content)
            print("=="*50)
            print(item_data)
            print("=="*50)
            
            response = ollama.chat(model='qwen2.5:ctx32k', messages=[
                {
                    'role': 'user',
                    'content': instruction
                },
            ])
            print(response['message']['content'])
            
            split = re.search(r"患者的病情进展：(.*?)医生的临床分析和决策：(.*)", response['message']['content'], re.DOTALL)
            conv_input = split.group(1).strip()
            conv_output = split.group(2).strip()
            
            conversation.append({'from':'human','value': "患者的病情进展：{}".format(conv_input)})
            conversation.append({'from':'gpt','value': "医生的临床分析和决策：{}".format(conv_output)})

            cnt += 1
        cnt_data += 1
        conversations.append({'conversations': conversation})
        

        item = {'conversations': conversation}
        
        
        # with open('/root/autodl-tmp/nano-graphrag-main/data/sft/task4-management-f3x.jsonl', 'a') as file:
        #     # for item in conversations:
        #     # 将每个字典转换为JSON字符串并写入文件
        #     json_str = json.dumps(item, ensure_ascii=False)
        #     file.write(json_str + '\n')
        #     file.flush()
        # # file.close()
    except:
        pass



# F20-F29
df_bc_f2x = pd.read_excel(path_f2x_bc)

conversations = []

for ipid in tqdm(df_bc_f2x['IPID'].unique()):
    try:
        if random.random() > 0.5:
            continue
        df_ipid = df_bc_f2x[df_bc_f2x['IPID'] == ipid]
        if df_ipid.shape[0] == 0:
            continue
        
        records = df_ipid[df_ipid['BL_TYPE'].isin(['主治医师查房记录', '主任医师查房记录', '副主任医师查房记录'])]
        records = records.sort_values(by='YZTL')
        records = records.drop_duplicates(subset=['YZTL'], keep='last')
        
        conversation = []
        cnt =0
        for idx, record in records.iterrows():
            if cnt > 5:
                break
            cnt += 1
            
            record_content = record['YZTL']
            
            instruction = f"""
            请将以下精神疾病患者的住院病程记录按照以下两部分进行整理和提取：

            患者的病情进展：包括患者的当前用药不良反应、体格检查、精神检查、辅助检查（如化验检查、影像学检查）等内容。整理时保持记录的原始连续文本，不进行总结或概括。
            医生的临床分析和决策：包括医生对患者病情和检查结果的分析、下一步诊疗计划等内容。整理时保持记录的原始连续文本，不进行总结或概括。
            以下是具体的示例：

            示例 1
            输入：
            精神检查：神清，定向力完整，表情烦躁，愁苦，患者今晨突然进入兴奋室，躺于床上，表情痛苦，询问其原因，患者手扶胸口，不耐烦地大喊：“我难受着呢。”情绪激动，测生命体征及血糖均正常。要求医生给其打针，劝说安抚不能听从，自知力部分，情感反应协调。
            化验结果：2021-07-13 帕罗西汀血药浓度(样本:血清):脱烷基喹硫平浓度<10ng/ml,帕罗西汀浓度24.24ng/ml,喹硫平浓度80.95ng/ml↓2021-07-12 尿有形成分分析+尿常规(样本:尿):大致正常。2021-07-12 全血细胞分析（静脉血）(样本:全血):白细胞计数3.510^9/L↓,中性粒细胞数1.76*10^9/L↓,患者白细胞数及中性粒细胞数有所减少，已服用升白药物治疗，定期复查。2021-07-12 生化26项(样本:血清):*白蛋白39.3g/L↓,*高密度脂蛋白1.15mmol/L↓,嘱患者高蛋白低脂饮食，适当运动，定期监测。
            下一步诊疗计划：不排除患者焦虑与减帕罗西汀有关，今日将帕罗西汀加至30mg/d，继续观察，结合心理治疗。

            输出：
            患者的病情进展：
            精神检查：神清，定向力完整，表情烦躁、愁苦。患者今晨突然进入兴奋室，躺于床上，表情痛苦。询问其原因，患者手扶胸口，不耐烦地大喊：“我难受着呢。”情绪激动。测量生命体征及血糖均正常。劝说安抚无效，自知力部分存在，情感反应协调。
            化验结果：2021-07-13 帕罗西汀血药浓度：脱烷基喹硫平浓度<10ng/ml，帕罗西汀浓度24.24ng/ml，喹硫平浓度80.95ng/ml（↓）。2021-07-12 尿有形成分分析+尿常规：大致正常。2021-07-12 全血细胞分析（静脉血）：白细胞计数3.5×10^9/L（↓），中性粒细胞数1.76×10^9/L（↓）。患者白细胞数及中性粒细胞数有所减少，已服用升白药物治疗，定期复查。2021-07-12 生化26项：白蛋白39.3g/L（↓），高密度脂蛋白1.15mmol/L（↓）。

            医生的临床分析和决策：
            临床分析：不排除患者焦虑与减帕罗西汀有关。患者出现烦躁、痛苦表现，需结合病情及化验结果综合分析。白细胞数及中性粒细胞数减少，已采取升白药物治疗措施，饮食建议为高蛋白低脂，建议适当运动并定期监测。
            下一步诊疗计划：今日将帕罗西汀剂量加至30mg/d，继续观察患者状态并结合心理治疗。

            示例 2
            输入：
            精神检查：神清，定向力完整，接触可，自诉将药物加量后情绪稍有改善，中午仍心烦，每日要求服用一片奥沙西泮，下午状态较好，能打牌，聊天，询问医生：“我什么时候能好，能出院，你们多好，干什么都正常，我还是懒得动。”自知力部分，情感反应协调。
            化验结果：2021-05-25 生化26项(样本:血清):*白蛋白37.4g/L↓,*高密度脂蛋白1.02mmol/L↓,氯110.1mmol/L↑,继续观察，定期复查。2021-05-25 全血细胞分析（静脉血）(样本:全血):白细胞计数3.410^9/L↓,中性粒细胞数1.8710^9/L↓,患者白细胞及中性粒细胞数减少。目前服用磷酸腺嘌呤对症处理，继续观察，择日复查。2021-05-25 尿有形成分分析+尿常规(样本:尿):PH值5.0↓,白细胞40/ul↑,患者无尿路刺激征，考虑标本污染，择日复查。
            下一步诊疗计划：根据患者监测血压，晨起血压正常上限，最高142/82mmHg，考虑今日将早晨酒石酸美托洛尔片加量至25mg/d,维持目前治疗方案。

            输出：
            患者的病情进展：
            精神检查：神清，定向力完整，接触可。患者自述加量药物后情绪稍有改善，但中午仍感到心烦，每日要求服用一片奥沙西泮。下午状态较好，能够打牌、聊天，并询问医生：“我什么时候能好，能出院，你们多好，干什么都正常，我还是懒得动。”自知力部分存在，情感反应协调。
            化验结果：2021-05-25 生化26项：白蛋白37.4g/L（↓），高密度脂蛋白1.02mmol/L（↓），氯110.1mmol/L（↑），嘱患者继续观察并定期复查。2021-05-25 全血细胞分析（静脉血）：白细胞计数3.4×10^9/L（↓），中性粒细胞数1.87×10^9/L（↓）。患者白细胞及中性粒细胞数减少，正在服用磷酸腺嘌呤对症处理，继续观察并择日复查。2021-05-25 尿有形成分分析+尿常规：PH值5.0（↓），白细胞40/μl（↑）。患者无尿路刺激征，考虑标本污染，择日复查。

            医生的临床分析和决策：
            临床分析：患者自述加药后情绪有所改善，但中午仍感心烦，提示焦虑情绪可能未完全缓解。化验检查显示白细胞及中性粒细胞数减少，尿液检查标本可能存在污染，但需择日复查以确认异常情况。血压监测晨起值接近正常上限。
            下一步诊疗计划：今日将早晨酒石酸美托洛尔片加量至25mg/d，以控制血压，维持目前的治疗方案，继续观察患者的情绪与生理指标变化。

            请按照以上格式完成任务。
            输入：
            {record_content}

            输出：

            """
            
            response = ollama.chat(model='qwen2.5:ctx32k', messages=[
                {
                    'role': 'user',
                    'content': instruction
                },
            ])
            print(response['message']['content'])
            
            split = re.search(r"患者的病情进展：(.*?)医生的临床分析和决策：(.*)", response['message']['content'], re.DOTALL)
            conv_input = split.group(1).strip()
            conv_output = split.group(2).strip()
            
            conversation.append({'from':'human','value': "患者的病情进展：{}".format(conv_input)})
            conversation.append({'from':'gpt','value': "医生的临床分析和决策：{}".format(conv_output)})
        
        conversations.append({'conversations': conversation})
        
        item = {'conversations': conversation}
        
        
        with open('/root/autodl-tmp/nano-graphrag-main/data/sft/task4-management-f2x.jsonl', 'a') as file:
            # for item in conversations:
            # 将每个字典转换为JSON字符串并写入文件
            json_str = json.dumps(item, ensure_ascii=False)
            file.write(json_str + '\n')
            file.flush()
        # file.close()
    except:
        pass

