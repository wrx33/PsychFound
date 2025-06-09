import pandas as pd
import numpy as np
import re
import os
import json
import tqdm
import random
import ollama

# 心境障碍
patient_info = pd.read_excel("/home/sjtu/anding_data/F30-F33/患者信息.xlsx")
first_record = pd.read_excel("/home/sjtu/anding_data/F30-F33/首程.xlsx")
case_record = pd.read_excel("/home/sjtu/anding_data/F30-F33/病程(2022-2024).xlsx")

# 精神分裂症
# patient_info = pd.read_excel("/home/sjtu/anding_data/F20-F29/F20-F29基本信息.xlsx")
# first_record = pd.read_excel("/home/sjtu/anding_data/F20-F29/F20-F29首程.xlsx")
# case_record = pd.read_excel("/home/sjtu/anding_data/F20-F29/F20-F29病程.xlsx")

# 精分
# case_record = pd.read_excel(r'D:\work\code\psychgpt_benchmark\domain\2023主诊断精神分裂症随机100人-病程.xlsx')
# patient_info = pd.read_excel(r'D:\work\code\psychgpt_benchmark\domain\2023主诊断精神分裂症随机100人.xlsx')

drugs_kyy_kjsb = [
    '利培酮',
    '去甲文拉法辛',
    '阿戈美拉汀',
    '阿立哌唑',
    '阿米替林',
    '艾司西酞普兰',
    '氨磺必利',
    '安非他酮',
    '奥氮平',
    '奥沙西泮',
    '丙戊酸',
    '地西泮',
    '度洛西汀',
    '多虑平',
    '多奈哌齐',
    '奋乃静',
    '氟奋乃静',
    '氟伏沙明',
    '氟西汀',
    '氟哌啶醇',
    '伏硫西汀',
    '卡马西平',
    '拉莫三嗪',
    '鲁拉西酮',
    '氯丙嗪',
    '氯氮平',
    '氯米帕明',
    '美金刚',
    '米安色林',
    '米氮平',
    '米那普仑',
    '帕罗西汀',
    '齐拉西酮',
    '曲唑酮',
    '舍曲林',
    '舒必利',
    '喹硫平',
    '文拉法辛',
    '西酞普兰',
    '硝西泮',
    '唑吡坦',
    '地昔帕明',
    '去甲替林',
    '丙米嗪',
    '多塞平',
    '马普替林',
    '帕利哌酮',
    '咪达唑仑',
    '劳拉西泮',
    '碳酸锂',
]

conversations_1 = []
conversations_2 = []
conversations_3 = []
conversations_4 = []

cases_missing_sth = []

save_path_task4 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F2X/task4-management.jsonl'

for ipid in patient_info['IPID'].unique():
    try:
        patient = patient_info[patient_info['IPID']==ipid]
        first = first_record[first_record['SYXH']==ipid]['YZTL'].iloc[0]
        records = case_record[case_record['IPID']==ipid]
    except:
        print('Something missing on {}'.format(ipid))
        continue
    
    
    # 准备输入字段
    senior_record = pd.concat([records[records['BL_TYPE']=='主任医师首次查房记录'], 
                               records[records['BL_TYPE']=='副主任医师首次查房记录']], axis=0, ignore_index=True)
    senior_record_content = senior_record['YZTL'].to_list()
    if len(senior_record_content) == 0:
        junior_record = records[records['BL_TYPE']=='主治医师首次查房记录']
        junior_record_content = junior_record['YZTL'].to_list()
        if len(junior_record_content) == 0:
            continue
        else:
            senior_record_content = junior_record_content
    senior_record_content.sort()
    content = senior_record_content[0]
    try:
        mental_exam = re.search(r'精神检查：(.*?)\n', content, re.S).group(1).strip().split('精神检查')[1]
    except:
        mental_exam = re.search(r'精神检查：(.*?)\n', content, re.S).group(1).strip().split('精神检查')[0]
    try:
        base = '患者{}性，{}岁。\n'.format(patient['性别'].iloc[0], patient['年龄'].iloc[0])
        current_his = patient['现病史'].iloc[0] + '\n'
        brief_his = '现病史：' + re.search(r'4.临床表现：(.*?)5.既往史：', first, re.S).group(1).strip() + '\n' 
        # brief_his = '简要病史：' + re.search(r'4.临床表现：(.*?)5.既往史：', first, re.S).group(1).strip() + '\n'    
        past_his = '既往史：' + re.search(r'5.既往史：(.*?)6.家族史：', first, re.S).group(1).strip() + '\n' 
        family_his = '家族史：' + re.search(r'6.家族史：(.*?)7.查体及辅助检查：', first, re.S).group(1).strip() + '\n' 
        exam = '查体，辅助检查及精神检查：' + re.search(r'7.查体及辅助检查：(.*?)拟诊讨论：', first, re.S).group(1).strip() + mental_exam + '\n' 
        date_in = '入院日期：{}年{}月{}日'.format(str(patient['入院日期'].iloc[0])[:4], str(patient['入院日期'].iloc[0])[4:6], str(patient['入院日期'].iloc[0])[6:8])
    except:
        print('Something missing on {}'.format(ipid))
        continue
    
    # if random.random() > 0.7:
    #     input_info = base + brief_his + past_his + family_his + exam + date_in
    # else:
    #     input_info = base + current_his + past_his + family_his + exam + date_in
    
    input_info = base + current_his + past_his + family_his + exam + date_in
    
    # 准备输出字段-task1
    # try:
    #     zs = '主诉：{}\n'.format(re.search(r'主因“(.*?)”入院', first, re.S).group(1).strip())
    # except:
    #     zs = first_record[first_record['SYXH']==ipid]['ZS'].iloc[0]
    
    # summary_0 = '0.简要病史：' + brief_his + '\n'
    # summary_1 = '1.病程标准：' + re.search(r'1.病程标准：(.*?)2.症状学标准', first, re.S).group(1).strip() + '\n'
    # summary_2 = '2.症状学标准：' + re.search(r'2.症状学标准：(.*?)3.严重程度标准', first, re.S).group(1).strip() + '\n'
    # summary_3 = '3.严重程度标准：' + re.search(r'3.严重程度标准：(.*?)4.排除标准', first, re.S).group(1).strip() + '\n'
    # summary_4 = '4.排除标准：' + re.search(r'4.排除标准：(.*?)5.入院印象', first, re.S).group(1).strip() + '\n'
    # summary = '病例特点：{}'.format(summary_0 + summary_1 + summary_2 + summary_3 + summary_4)

    # output_task1 = zs + summary
    
    # 准备输出字段-task2
    
    # primary_diag = patient['出院主要诊断'].iloc[0]
    # try:
    #     clean_analysis = re.search(r'对诊断及鉴别诊断分析：(.*?)诊疗原则及处理计划', content, re.S).group(1).strip()
    # except:
    #     print(content)
    #     continue
    # differential = f'主要诊断：{primary_diag}。' + '诊断及鉴别诊断分析：{}'.format("，".join(re.split(r'[,:，：]', clean_analysis)[1:]))
    # if '鉴别' not in differential:
    #     differential += '鉴别诊断：' + re.search(r'鉴别诊断：(.*?)诊疗计划', first, re.S).group(1).strip() + '\n' 
    # output_task2 = differential
    
    
    # 准备输出字段-task3
    
    # try:
    #     output_task3 = '诊疗计划：' + re.search(r'诊疗计划：(.*?)医师签名', first, re.S).group(1).strip() + '\n'
    # except:
    #     print(first)
    #     continue
    
    
    # 准备输出字段-task4
    records = records.sort_values(by='YZTL')
    if len(records) > 30:
        continue
    # prompt = "请从以下患者病程记录中，分别提取出客观的患者病情进展(包括患者的临床表现，查体，精神检查，辅助检查等结果)和主观的医生分析及诊疗决策（包括对患者检查结果的分析，对患者的诊断，治疗计划的制定等内容）。输出格式为：患者病情进展：xxx。医生诊疗决策：xxx。不要输出其他内容。"
    
    prompt = """
    你是一个精通大语言模型开发的精神科医生，现在要将精神科患者数据处理成结构化数据来训练大语言模型。
    请从以下患者病程记录中，精确提取并区分两部分内容：

    ###患者病情进展：包括客观记录的患者临床表现（如症状、查体结果、精神检查结果、辅助检查数据及结果等）。
    
    ###医生诊疗决策：包括主观分析和决策内容（如对检查结果的分析、诊断结论、治疗计划的制定等）。
    
    请严格按照以下输出格式组织结果：

    患者病情进展：xxx
    医生诊疗决策：xxx
    
    注意事项：
    仅提取相关信息，不要输出与任务无关的内容或注释。
    确保信息分类准确，避免混淆客观病情与主观诊疗内容。
    """
    conversation4 = []
    conversation4.append({'from':'IPID','value': str(ipid)})
    
    for index, record in records.iterrows():
        record_type = record['BL_TYPE']
        record_content = record['YZTL']
        
        response = ollama.chat(model='qwen2:ctx32k', messages=[
        {
            'role': 'user',
            'content': f"{prompt}患者病程记录：{record_content}",
        },
        ])
        orgnized_record = response['message']['content']
        
        # 提取患者病情进展字段和医生诊疗决策字段
        try:
            patient_progress = re.search(r'患者病情进展：(.*?)医生诊疗决策', orgnized_record, re.S).group(1).strip()
        except AttributeError:
            patient_progress = "未提取到患者病情进展字段"
        
        try:
            doctor_decision = re.search(r'医生诊疗决策：(.*)', orgnized_record, re.S).group(1).strip()
        except AttributeError:
            doctor_decision = "未提取到医生诊疗决策字段"
        
        
        input_task4 = f"患者病情进展：{patient_progress}。"
        output_task4 = f"医生诊疗决策：{doctor_decision}。"

        conversation4.append({'from':'human','value': input_task4})
        conversation4.append({'from':'gpt','value': output_task4})
        
    with open(save_path_task4, 'a', encoding='utf-8') as f:
        json_str = json.dumps(conversation4, ensure_ascii=False)
        f.write(json_str + '\n')
        f.flush()

   
    # 创建数据集 task1
    # conversation1 = []
    # conversation1.append({'from':'IPID','value': str(ipid)})
    # conversation1.append({'from':'human','value': input_info})
    # conversation1.append({'from':'gpt','value': output_task1})
    # conversations_1.append({'conversations': conversation1})
    
    # # 创建数据集 task2
    # conversation2 = []
    # conversation2.append({'from':'IPID','value': str(ipid)})
    # conversation2.append({'from':'human','value': input_info})
    # conversation2.append({'from':'gpt','value': output_task2})
    # conversations_2.append({'conversations': conversation2})
    
    # # 创建数据集 task3
    # conversation3 = []
    # conversation3.append({'from':'IPID','value': str(ipid)})
    # conversation3.append({'from':'human','value': input_info})
    # conversation3.append({'from':'gpt','value': output_task3})
    # conversations_3.append({'conversations': conversation3})
    
    # # 创建数据集 task4
    # conversation4 = []
    # conversation4.append({'from':'IPID','value': str(ipid)})
    # conversation4.append({'from':'human','value': input_info})
    # conversation4.append({'from':'gpt','value': output_task4})
    # conversations_4.append({'conversations': conversation4})

# print(len(conversations_1))
# print(len(conversations_2))
# print(len(conversations_3))
print(len(conversations_4))

print(cases_missing_sth)

# with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F3X/task1-summary.jsonl', 'w') as file1:
#     for item in conversations_1:
#         # 将每个字典转换为JSON字符串并写入文件
#         json_str = json.dumps(item, ensure_ascii=False)
#         file1.write(json_str + '\n')

# with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F3X/task2-diagnosis.jsonl', 'w') as file2:
#     for item in conversations_2:
#         # 将每个字典转换为JSON字符串并写入文件
#         json_str = json.dumps(item, ensure_ascii=False)
#         file2.write(json_str + '\n')

# with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F3X/task3-treatment.jsonl', 'w') as file3:
#     for item in conversations_3:
#         # 将每个字典转换为JSON字符串并写入文件
#         json_str = json.dumps(item, ensure_ascii=False)
#         file3.write(json_str + '\n')

# with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F2X/task4-management.jsonl', 'w') as file4:
#     for item in conversations_4:
#         # 将每个字典转换为JSON字符串并写入文件
#         json_str = json.dumps(item, ensure_ascii=False)
#         file4.write(json_str + '\n')