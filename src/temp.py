import os
import json
import pandas as pd
import numpy as np
import glob


# path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_knowledge'
# file_list = os.listdir(path)
# for file in file_list:
#     file_path = os.path.join(path, file)
#     data = []

#     # 剔除输出不全的条目
#     with open(file_path, 'r') as f:
#         for line in f:
#             newline = eval(line)
#             ans = newline[1]['value']
#             if len(ans) == 0 or ans[-1] == ':' or ans[-1] == '：':
#                 continue
#             data.append(newline)
#     f.close()

#     save_path = file_path.replace('psychgpt_sft_knowledge', 'psychgpt_sft_knowledge_1224')
#     with open(save_path, 'w') as f:
#         for item in data:
#             json_str = json.dumps(item, ensure_ascii=False)
#             f.write(json_str + '\n')
#     f.close()

#     # 剔除输入问题中包含信息不全的条目


root = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/JAMA*'
path_list = glob.glob(root)

num_words = 0
for path in path_list:
    file_list = os.listdir(path)

    for file in file_list:
        file_path = os.path.join(path,file)
        content = ""
        with open(file_path, 'r') as f:
            content = f.read()
        
        num_words += len(content)

        print(f"{file} 字数：{len(content)}")


print(num_words)