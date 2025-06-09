import json
import pandas as pd
import numpy as np
import re
import os
import random


path_basic_2x = "/home/sjtu/anding_data/F20-F29/F20-F29基本信息.xlsx"
path_first_2x = "/home/sjtu/anding_data/F20-F29/F20-F29首程.xlsx"
path_course_2x = "/home/sjtu/anding_data/F20-F29/F20-F29病程.xlsx"

path_basic_3x = "/home/sjtu/anding_data/F30-F33/患者信息.xlsx"
path_first_3x = "/home/sjtu/anding_data/F30-F33/首程.xlsx"
path_course_3x_1 = "/home/sjtu/anding_data/F30-F33/病程(2019-2021).xlsx"
path_course_3x_2 = "/home/sjtu/anding_data/F30-F33/病程(2022-2024).xlsx"

# df_basic_2x = pd.read_excel(path_basic_2x)
# df_first_2x = pd.read_excel(path_first_2x)

# df_basic_3x = pd.read_excel(path_basic_3x)
# df_first_3x = pd.read_excel(path_first_3x)

# 写首程
def task_write_first():

    df_first_2x = pd.read_excel(path_first_2x)
    df_first_3x = pd.read_excel(path_first_3x)

    path_2x = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F2X/task3-treatment.jsonl'
    raw_data_2x = []
    with open(path_2x, 'r') as file:
        for line in file:
            newline = eval(line)
            raw_data_2x.append(newline)
    file.close()

    path_3x = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_F3X/task3-treatment.jsonl'
    raw_data_3x = []
    with open(path_3x, 'r') as file:
        for line in file:
            newline = eval(line)
            raw_data_3x.append(newline)
    file.close()

    # new_conversations = []
    # cnt = 0
    # for item in raw_data_2x:
    #     try:
    #         print(cnt)
    #         cnt+=1
    #         ipid = item['conversations'][0]['value']
    #         new_input = "请根据以下患者信息，撰写首程病历。\n" + item['conversations'][1]['value']
    #         new_output = df_first_2x[df_first_2x['SYXH']==int(ipid)]['YZTL'].iloc[0]

    #         patient_name = re.search(r"\n患者(.*?)，", new_output, re.DOTALL).group(1).strip()
    #         new_output = new_output.replace(patient_name, "")

    #         new_output = new_output.split('医师签名')[0]
    #         new_output = "\n".join(new_output.split('\n')[1:])
            
    #         conversation = []
    #         conversation.append({'from':'human','value': new_input})
    #         conversation.append({'from':'gpt','value': new_output})

    #         new_conversations.append({'conversations': conversation})
    #     except:
    #         pass
    

    # with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/augmented/write_firstcourse_2x.jsonl', 'w') as file1:
    #     for item in new_conversations:
    #         # 将每个字典转换为JSON字符串并写入文件
    #         json_str = json.dumps(item, ensure_ascii=False)
    #         file1.write(json_str + '\n')
    # file1.close()


    new_conversations = []
    cnt = 0
    for item in raw_data_3x:
        try:
            print(cnt)
            cnt+=1
            ipid = item['conversations'][0]['value']
            new_input = "请根据以下患者信息，撰写首程病历。\n" + item['conversations'][1]['value']
            new_output = df_first_3x[df_first_3x['SYXH']==int(ipid)]['YZTL'].iloc[0]

            patient_name = re.search(r"\n患者(.*?)，", new_output, re.DOTALL).group(1).strip()
            new_output = new_output.replace(patient_name, "")

            new_output = new_output.split('医师签名')[0]
            new_output = "\n".join(new_output.split('\n')[1:])
            
            conversation = []
            conversation.append({'from':'human','value': new_input})
            conversation.append({'from':'gpt','value': new_output})

            new_conversations.append({'conversations': conversation})
        except:
            pass
    

    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/augmented/write_firstcourse_3x.jsonl', 'w') as file1:
        for item in new_conversations:
            # 将每个字典转换为JSON字符串并写入文件
            json_str = json.dumps(item, ensure_ascii=False)
            file1.write(json_str + '\n')
    file1.close()

# task_write_first()




# 选择题
def task_choice():
    path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/augmented/zeroshot_train_CMB.jsonl'
    
    raw_data = []
    with open(path, 'r') as file:
        for line in file:
            newline = eval(line)
            raw_data.append(newline)
    file.close()

    data = []
    for item in raw_data:
        if random.random() <= 0.7:
            continue
        
        if 'A' not in item['conversations'][0]['value']:
            continue

        if random.random() <= 0.5:
            
            conversation = []
            conversation.append({'from':'human','value': "\n".join(item['conversations'][0]['value'].split('\n')[1:])})
            conversation.append({'from':'gpt','value': item['conversations'][1]['value']})
            data.append({"conversations": conversation})
        else:
            data.append(item)

    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/augmented/choice_questions.jsonl', 'w') as file1:
        for item in data:
            # 将每个字典转换为JSON字符串并写入文件
            json_str = json.dumps(item, ensure_ascii=False)
            file1.write(json_str + '\n')
    file1.close()

task_choice()



