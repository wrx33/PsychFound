import pandas as pd
import json
import os
import re
import numpy as np

def get_anony_names():

    path_bc = "/home/sjtu/anding_data/F20-F29/F20-F29病程.xlsx"
    path_sc = "/home/sjtu/anding_data/F20-F29/F20-F29首程.xlsx"

    data_bc = pd.read_excel(path_bc)
    data_sc = pd.read_excel(path_sc)

    doctor_names = []
    patient_names = []

    for idx, item in data_bc.iterrows():
        # print(item)
        content = item['YZTL']
        try:
            contained_doctor_name = re.search(r'医师签名：(.*?)\n', content, re.DOTALL).group(1).strip()
            doctor_names.extend(re.split(r'[/ ]', contained_doctor_name))
        except:
            continue


    for idx, item in data_sc.iterrows():
        content = item['YZTL']
        try:
            contained_patient_name = re.search(r'患者(.*?)，', content.split('\n')[1], re.DOTALL).group(1).strip()
            patient_names.append(contained_patient_name)
        except:
            continue

    doctor_names = list(set(doctor_names))
    patient_names = list(set(patient_names))

    print(f"医生名单：{doctor_names}\n")
    print(f"患者名单：{patient_names}\n")

    return doctor_names, patient_names

def anony_bc_sft():
    path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/augmented/augmented_bc_F30-F33.jsonl'

    data = []
    with open(path, 'r') as file:
        for line in file:
            newline = eval(line)
            data.append(newline)
    file.close()

    names = []
    for item in data:
        content = item['conversations'][1]['content']
        try:
            content_names = content.split('医师签名：')[1]
            names.extend(re.split(r'[/ ]', content_names))
        except:
            pass
    names = list(set(names))
    print(names)

    annoy_data = []
    for item in data:
        conversation = []
        annoy_input = item['conversations'][0]['content']
        annoy_output = item['conversations'][1]['content']
        annoy_output = annoy_output.split('医师签名')[0]
        for name in names:
            annoy_output = annoy_output.replace(name, "")

        conversation.append({'from':'human','value': annoy_input})
        conversation.append({'from':'gpt','value': annoy_output})

        annoy_data.append({'conversations': conversation})
    
    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/augmented/augmented_bc_F30-F33_annoy.jsonl', 'w') as file1:
        for item in annoy_data:
            # 将每个字典转换为JSON字符串并写入文件
            json_str = json.dumps(item, ensure_ascii=False)
            file1.write(json_str + '\n')
    file1.close()

def anony_sc_sft():
    path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/augmented/augmented_sc_F20-F29.jsonl'

    data = []
    with open(path, 'r') as file:
        for line in file:
            newline = eval(line)
            data.append(newline)
    file.close()

    # names = []
    # for item in data:
    #     content1 = item['conversations'][0]['content']
    #     names.append(re.search(r"患者(.*?)，", content1.split('\n')[1]).group(1).strip())

    #     content2 = item['conversations'][1]['content']
    #     try:
    #         content_names = content.split('医师签名：')[1]
    #         names.extend(re.split(r'[/ ]', content_names))
    #     except:
    #         pass
    # names = list(set(names))
    # print(names)

    annoy_data = []
    for item in data:
        # annoy_data.append(item)

        conversation = []
        # annoy_input = item['conversations'][0]['content']
        # annoy_output = item['conversations'][1]['content']
        # patient_name = re.search(r"\n患者(.*?)，", annoy_input, re.DOTALL).group(1).strip()
        # annoy_input = annoy_input.replace(patient_name, "")

        # annoy_output = annoy_output.split('医师签名')[0]

        conversation.append({'from':'human','value': item['conversations'][0]['content']})
        conversation.append({'from':'gpt','value': item['conversations'][1]['content']})

        annoy_data.append({'conversations': conversation})
    
    with open('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/augmented/augmented_sc_F20-F29_annoy.jsonl', 'w') as file1:
        for item in annoy_data:
            # 将每个字典转换为JSON字符串并写入文件
            json_str = json.dumps(item, ensure_ascii=False)
            file1.write(json_str + '\n')
    file1.close()


anony_sc_sft()