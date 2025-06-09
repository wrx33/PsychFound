import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from llamafactory.chat import ChatModel

def run_task(model, instruction, data, save_path):
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    writer = open(save_path, "a", encoding='utf-8')

    # answers = []
    for item in tqdm(data):
        input_content = item['conversations'][1]['value']
        output_label = item['conversations'][2]['value']
        model_input = "{}\n {}".format(instruction, input_content)
        messages = [
            {"role": "user", "content": model_input}
        ]
        answer = generate_answer(model, messages)
        conversation = []
        conversation.append({'from': 'human', 'value': '请阅读以下患者病程记录，然后回答问题。\n 病程记录：{}\n 问题：{}'.format(input_content, answer)})
        conversation.append({'from': 'gpt', 'value': answer})
        # answers.append({'conversations': conversation})
        writer.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    
    # with open(save_path, 'w', encoding='utf-8') as file:
    #     for item in answers:
    #         # 将每个字典转换为JSON字符串并写入文件
    #         json_str = json.dumps(item, ensure_ascii=False)
    #         file.write(json_str + '\n')
    # file.close()
    print("{}Mission accomplished{}".format('='*30, '='*30))


def generate_answer(model, messages):
    response = ""
    for new_text in model.stream_chat(messages):
        # print(new_text, end="", flush=True)
        response += new_text

    return response

def annotate_sc():
    path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_cpt_0805/首程.txt'
    save_path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_cpt_0805/首程_annotated.txt'

    with open(path1, 'r') as file1:
        data = file1.read()

    file1.close()
    original_texts = data.split('**'*50)
    annotation_texts = []
    
    chat_model = ChatModel()

    writer = open(save_path1, "a", encoding='utf-8')

    for text in tqdm(original_texts):
        query = '你是一位专家级精神科临床医生，请结合精神科专业临床知识阅读以下实习医生书写的患者病历，然后书写一段对于该患者病情和医生诊疗行为的分析性文字。病例内容：{}'.format(text)
        messages = [{"role": "user", "content": query}]
        response = ""
        for new_text in chat_model.stream_chat(messages):
            # print(new_text, end="", flush=True)
            response += new_text
        
        text_w_annotation = '病历：' + text + '\n' + '分析：' + response
        annotation_texts.append('病历：' + text + '\n' + '分析：' + response)
        writer.write(text_w_annotation + "\n")

def annotate_bc():
    chat_model = ChatModel()

    path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_cpt_0805/病程1.txt'
    save_path1 = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_cpt_0805/病程1_annotated.txt'

    with open(path1, 'r') as file1:
        data = file1.read()

    file1.close()
    # original_texts = data.split('**'*50)
    original_texts = data.split('\n\n\n')
    annotation_texts = []

    writer = open(save_path1, "a", encoding='utf-8')

    for text in tqdm(original_texts):
        # if len(text) < 10 or len(text) > 10000 or len(re.findall(r'\n', text)) == 1:
        #     continue
        if (not '目前治疗' in text) or (not '下一步诊疗计划' in text):
            continue
        query = '你是一名精神科临床专家，请结合精神科专业临床知识阅读以下患者住院期间真实病程记录，然后结合患者具体的临床表现（包括体格检查，辅助检查，精神检查等结果），对当前和下一步的治疗方案（包括但不限于如药物和剂量调整，检验检查，其他治疗等）进行详细分析（结合相关疾病，药物等知识，分析治疗方案的意义，若有未考虑到的因素，潜在的风险或不合理的治疗方案也请指出并给出建议）。病例内容：{}'.format(text)
        # query = '请结合精神科专业临床知识阅读以下患者住院期间真实病程记录，然后结合患者具体的临床表现（包括体格检查，辅助检查，精神检查等结果），对当前和下一步的治疗方案进行点评分析（分析的依据要求来源于官方权威指南教材等知识）。病例内容：{}'.format(text)
        messages = [{"role": "user", "content": query}]
        response = ""
        for new_text in chat_model.stream_chat(messages):
            # print(new_text, end="", flush=True)
            response += new_text
        
        text_w_annotation = '病历：' + text + '\n' + '分析：' + response
        # annotation_texts.append('病历：' + text + '\n' + '分析：' + response)
        writer.write(text_w_annotation + "\n")


def generate_sft_from_txt(chat_model, file_path, save_path):
    # chat_model = ChatModel()
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        data = file.read()
    
    

    writer = open(save_path, "a", encoding='utf-8')

    # 全文生成
    query = f"""
    任务描述：
    请根据以下输入的论文内容，生成一组与论文内容相关的问题-答案组合，这些组合将用于对大语言模型进行监督微调训练。每个问题必须包含充分的上下文信息，使其在单独使用时仍然可以被准确回答。

    要求：

    问题设计：

    问题必须包含足够的上下文信息（如背景、定义、实验描述等），使其独立于全文时仍清晰完整。
    问题应涵盖论文的核心内容，包括但不限于研究背景、方法、实验设计、关键数据、结论及创新点。
    问题类型以简答题和详细问答为主，确保全面性和多样性。
    答案生成：

    答案应基于论文内容准确生成，简洁明了，重点突出。
    严格避免脱离论文内容的推测。
    格式要求：
    输出结果以结构化格式呈现：

    问题1：xxx（包含必要上下文）\n\n
    答案1：xxx \n\n
    问题2：xxx（包含必要上下文）\n\n
    答案2：xxx \n\n
    示例格式：

    问题1：在[论文背景或问题描述]中，作者提出的主要研究目标是什么？
    答案1：作者的主要研究目标是[具体目标描述]。
    问题2：根据论文，使用[具体方法]进行实验时，作者获得了什么关键发现？
    答案2：作者发现[关键发现内容]。
    其他注意事项：

    每个问题-答案组合必须自洽、独立，不依赖上下文。
    尽量覆盖论文中的重要信息，确保生成的问题具有多样性和针对性。
    以下是论文内容：

    {data}
    """

    # query = f"请将以下参考内容，转换成一个问答对。输出格式为：###问题：。\n ###答案：。\n 不要输出任何其他内容。参考内容：{text}"
    # query = '请结合精神科专业临床知识阅读以下患者住院期间真实病程记录，然后结合患者具体的临床表现（包括体格检查，辅助检查，精神检查等结果），对当前和下一步的治疗方案进行点评分析（分析的依据要求来源于官方权威指南教材等知识）。病例内容：{}'.format(text)
    messages = [{"role": "user", "content": query}]
    response = ""
    for new_text in chat_model.stream_chat(messages):
        # print(new_text, end="", flush=True)
        response += new_text
    
    # print(response)
    pattern = r"(问题\d+：.*?)(答案\d+：.*?)(?=问题\d+：|$)"
    matches = re.findall(pattern, response, re.DOTALL)
    qa_pairs = [{"问题": match[0].strip(), "答案": match[1].strip()} for match in matches]

    for i, pair in enumerate(qa_pairs, 1):
        conversation = []
        conversation.append({'from': 'human', 'value': pair["问题"]})
        conversation.append({'from': 'gpt', 'value': pair["答案"]})

        writer.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    # 逐段生成
    # original_texts = data.split('\n')
    # for text in tqdm(original_texts):
    #     if len(text) < 50:
    #         continue
        
    #     query = f"""
    #     任务描述：
    #     请根据以下输入的论文内容，生成一组与论文内容相关的问题-答案组合，这些组合将用于对大语言模型进行监督微调训练。每个问题必须包含充分的上下文信息，使其在单独使用时仍然可以被准确回答。

    #     要求：

    #     问题设计：

    #     问题必须包含足够的上下文信息（如背景、定义、实验描述等），使其独立于全文时仍清晰完整。
    #     问题应涵盖论文的核心内容，包括但不限于研究背景、方法、实验设计、关键数据、结论及创新点。
    #     问题类型以简答题和详细问答为主，确保全面性和多样性。
    #     答案生成：

    #     答案应基于论文内容准确生成，简洁明了，重点突出。
    #     严格避免脱离论文内容的推测。
    #     格式要求：
    #     输出结果以结构化格式呈现：

    #     问题1：xxx（包含必要上下文）
    #     答案1：xxx
    #     问题2：xxx（包含必要上下文）
    #     答案2：xxx
    #     示例格式：

    #     问题：在[论文背景或问题描述]中，作者提出的主要研究目标是什么？
    #     答案：作者的主要研究目标是[具体目标描述]。
    #     问题：根据论文，使用[具体方法]进行实验时，作者获得了什么关键发现？
    #     答案：作者发现[关键发现内容]。
    #     其他注意事项：

    #     每个问题-答案组合必须自洽、独立，不依赖上下文。
    #     尽量覆盖论文中的重要信息，确保生成的问题具有多样性和针对性。
    #     以下是论文内容：

    #     {text}
    #     """

    #     # query = f"请将以下参考内容，转换成一个问答对。输出格式为：###问题：。\n ###答案：。\n 不要输出任何其他内容。参考内容：{text}"
    #     # query = '请结合精神科专业临床知识阅读以下患者住院期间真实病程记录，然后结合患者具体的临床表现（包括体格检查，辅助检查，精神检查等结果），对当前和下一步的治疗方案进行点评分析（分析的依据要求来源于官方权威指南教材等知识）。病例内容：{}'.format(text)
    #     messages = [{"role": "user", "content": query}]
    #     response = ""
    #     for new_text in chat_model.stream_chat(messages):
    #         # print(new_text, end="", flush=True)
    #         response += new_text
        
    #     # print(response)
    #     questions = re.findall(r'###问题：(.*?)\n', response)
    #     answers = re.findall(r'###答案：(.*)', response)
    #     # questions = [match[0] for match in qa_matches]
    #     # answers = [match[1] for match in qa_matches]

    #     print(questions)
    #     print(answers)
    #     for i in range(len(questions)):
    #         conversation = []
    #         conversation.append({'from': 'human', 'value': questions[i]})
    #         conversation.append({'from': 'gpt', 'value': answers[i]})

    #         writer.write(json.dumps(conversation, ensure_ascii=False) + "\n")


def main():
    # annotate_sc()
    # annotate_bc()
    chat_model = ChatModel()
    path = '/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_cpt_knowledge_papers'
    file_list = os.listdir(path)
    for file in file_list:
        print(file)
        file_path = os.path.join(path, file)
        save_path = os.path.join('/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_knowledge_papers', file[:-4]+'.jsonl')
        generate_sft_from_txt(chat_model, file_path, save_path)
    



    # while True:
    #     query = '你是一位专家级精神科临床医生，请阅读以下患者病历，然后对该病例进行点评。病例内容：{}'.format(original_texts[0])
    #     messages = [{"role": "user", "content": query}]
    #     response = ""
    #     for new_text in chat_model.stream_chat(messages):
    #         print(new_text, end="", flush=True)
    #         response += new_text
    #     print()



if __name__ == '__main__':
    # --model_name_or_path /data/sjtu/wrx/model_weights/Qwen2-7B-Instruct/ \
    # --template qwen \
    main()