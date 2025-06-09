import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from llamafactory.chat import ChatModel
# from llmtuner.extras.misc import torch_gc

def run_task(model, instruction, data, save_path):
    answers = []

    for item in tqdm(data):
        input_content = item['conversations'][1]['value']
        output_label = item['conversations'][2]['value']
        model_input = "{}\n 患者信息：{}".format(instruction, input_content)
        messages = [
            {"role": "user", "content": model_input}
        ]
        answer = generate_answer(model, messages)
        conversation = []
        conversation.append({'input': model_input, 'output': answer, 'label': output_label})
        answers.append({'conversations': conversation})

    
    # os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as file:
        for item in answers:
            # 将每个字典转换为JSON字符串并写入文件
            json_str = json.dumps(item, ensure_ascii=False)
            file.write(json_str + '\n')
    file.close()
    print("{}Mission accomplished{}".format('='*30, '='*30))


def generate_answer(model, messages):
    response = ""
    for new_text in model.stream_chat(messages):
        # print(new_text, end="", flush=True)
        response += new_text

    return response

import argparse
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--save_path", type=str, help="psych exam", default='/home/sjtu/wrx/code/LLaMA-Factory-main/evaluation/psychbench/results/')
    # args = parser.parse_args()
    save_path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/psychbench/results/qwen2-7b-sft-lora-256-all-1227'
    os.makedirs(save_path, exist_ok=True)

    path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/psychbench/task1.jsonl'
    path2 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/psychbench/task2.jsonl'
    path3 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/psychbench/task3.jsonl'
    path4 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/psychbench/task4.jsonl'
    path5 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/evaluation/psychbench/task5.jsonl'

    data1 = []
    with open(path1, 'r') as file1:
        for line in file1:
            newline = eval(line)
            data1.append(newline)
    file1.close()

    data2 = []
    with open(path2, 'r') as file2:
        for line in file2:
            newline = eval(line)
            data2.append(newline)
    file2.close()
        
    data3 = []
    with open(path3, 'r') as file3:
        for line in file3:
            newline = eval(line)
            data3.append(newline)
    file3.close()
            
    data4 = []
    with open(path4, 'r') as file4:
        for line in file4:
            newline = eval(line)
            data4.append(newline)
    file4.close()
    
    data5 = []
    with open(path5, 'r') as file5:
        for line in file5:
            newline = eval(line)
            data5.append(newline)
    file5.close()
    # model = AutoModelForCausalLM.from_pretrained(
    #     '/data/sjtu/wrx/llamafactory/psychgpt_sft-glm3-6b-lora-256-merge-0729/', # "qwen/Qwen2-72B-Instruct"
    #     torch_dtype="auto",
    #     device_map="auto",
    #     trust_remote_code=True
    # )
    # tokenizer = AutoTokenizer.from_pretrained('/data/sjtu/wrx/llamafactory/psychgpt_sft-glm3-6b-lora-256-merge-0729/', trust_remote_code=True)
    
    chat_model = ChatModel()

    instructions_1 = "请根据提供的患者信息提炼主诉，并依据ICD-10标准总结该患者的诊断标准，包括病程、症状学、严重程度和排除标准。"
    save_path1 = os.path.join(save_path, 'task1.jsonl')
    run_task(chat_model, instructions_1, data1, save_path1)
    
    instructions_2 = "根据下述患者信息按照ICD-10诊断标准给出主要诊断以及共病诊断（若有）的疾病名称（精确到亚型）。仅需要给出诊断疾病名称，无需进行分析。"
    save_path2 = os.path.join(save_path, 'task2.jsonl')
    run_task(chat_model, instructions_2, data2, save_path2)
    
    instructions_3 = "根据以下患者信息，进行精神心理疾病之间的临床鉴别诊断分析，给出主要诊断和可能的鉴别诊断。"
    save_path3 = os.path.join(save_path, 'task3.jsonl')
    run_task(chat_model, instructions_3, data3, save_path3)
    
    instructions_4 = "根据患者当前的病情信息，给出最合适的精神科药物治疗用药建议。"
    save_path4 = os.path.join(save_path, 'task4.jsonl')
    run_task(chat_model, instructions_4, data4, save_path4)

    instructions_5 = "请阅读以下患者病程记录，然后回答问题。"
    save_path5 = os.path.join(save_path, 'task5.jsonl')
    run_task(chat_model, instructions_5, data5, save_path5)

if __name__ == '__main__':
    main()