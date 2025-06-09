import json
import re
import os

path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_knowledge_1224'
file_list = os.listdir(path)

for file in file_list:
    file_path = os.path.join(path, file)
    
    raw_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            newline = eval(line)
            raw_data.append(newline)
    f.close()

    conversations = []
    for item in raw_data:
        conversation = []
        conversation.append({'from': 'human', 'value': item[0]['value']})
        conversation.append({'from': 'gpt', 'value': item[1]['value']})
        conversations.append({'conversations': conversation})
    
    with open(file_path, 'w') as file1:
        for item in conversations:
            # 将每个字典转换为JSON字符串并写入文件
            json_str = json.dumps(item, ensure_ascii=False)
            file1.write(json_str + '\n')
    file1.close()
