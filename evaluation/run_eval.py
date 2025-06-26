from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from itertools import islice
import time

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, GenerationConfig


class PromptWrapper():
    def __init__(
            self, 
            tokenizer, 
            instruction_template, 
            conv_collater,
            use_cot=False,
            prompt_prefix = '扮演一名专业的精神心理临床医生进行诊疗'
    ):

        self.instruction_template = instruction_template

        self.question_template_option,self.question_template_nonoption = self.get_question_template(use_cot=use_cot)

        if '{fewshot_examples}' in self.instruction_template:
            # use fewshot examples
            # keep the fewshot placeholder, since examples are sample-specific 
            self.input_template_option = self.instruction_template.format(instruction=self.question_template_option, fewshot_examples='{fewshot_examples}') 
            self.input_template_nonoption = self.instruction_template.format(instruction=self.question_template_nonoption, fewshot_examples='{fewshot_examples}') 

        else:
            self.input_template_option = self.instruction_template.format(instruction=self.question_template_option)
            self.input_template_nonoption = self.instruction_template.format(instruction=self.question_template_nonoption)

        self.conv_collater = conv_collater # for multi-turn QA only, implemented for each model
        self.tokenizer = tokenizer
        self.prompt_prefix = prompt_prefix


    def get_system_template(self, t):
        if t.strip() == '':
            return '{instruction}'
        else:
            try:
                t.format(instruction='')
            except:
                raise Exception('there must be a {instruction} placeholder in the system template')
        return t
    
    def get_question_template(self, use_cot):
        if use_cot:
            return ["以下是精神科专业考试的一道{question_type}，请分析每个选项，并最后给出答案。\n{question}\n{option_str}",\
                    "以下是精神科专业考试的一道{question_type}，请根据你的知识给出专业详细的回答。\n{question}"]
        else:
            return ["以下是精神科专业考试的一道{question_type}，不需要做任何分析和解释，直接输出答案选项。\n{question}\n{option_str}",\
                    "以下是精神科专业考试的一道{question_type}，请根据你的知识给出专业详细的回答。\n{question}"]

    def wrap(self, data):
        '''
        data.keys(): ['id', 'exam_type', 'exam_class', 'question_type', 'question', 'option']. These are the raw data.
        We still need 'option_str'.
        '''
        res = []
        lines = []
        for line in data:
            if 'question_type' in line.keys():
                if '选择题' in line['question_type']:
                    line["option_str"] = "\n".join(
                        [f"{k}. {v}" for k, v in line["option"].items() if len(v) > 1]
                    )
                    query = self.input_template_option.format_map(line)
                else:
                    query = self.input_template_nonoption.format_map(line)
            else:
                if not line["conversations"][0]['from'] == 'human':
                    line["conversations"] = line["conversations"][1:]
                line['question'] = line['conversations'][0]['value']
                input_template_nonoption_clinical = self.input_template_nonoption.replace('以下是精神科专业考试的一道{question_type}',self.prompt_prefix)
                query = input_template_nonoption_clinical.format_map(line)
            line['query'] = query

            res.append(query)
            lines.append(line)
        
        return res, lines
    
    def wrap_conv(self, data): # add
        lines = []
        res = []
        for line in data:
            # print(line)
            collated, partial_qa = self.conv_collater(line)
            # collated: ['Q', 'QAQ', 'QAQAQ', ...]
            # partial_qa: [
            #   [{'q': 'q'}], 
            #   [{'q': 'q', 'a': 'a'}, {'q'}], 
            #   [{'q': 'q', 'a': 'a'}, {'q': 'q', 'a': 'a'}, {'q': 'q'}]
            # ]
            res.extend(collated) # 1d list
            lines.extend(partial_qa)           
        return res, lines

    def unwrap(self, outputs, num_return_sequences):        
        batch_return = []
        responses_list = []
        for i in range(len(outputs)):
            # sample_idx = i // num_return_sequences
            output = outputs[i][self.lengths: ] # slicing on token level
            output = self.tokenizer.decode(output, skip_special_tokens=True)

            batch_return.append(output)
            if i % num_return_sequences == num_return_sequences - 1:
                responses_list.append(batch_return)
                batch_return = []
        return responses_list

class MyDataset(Dataset):
    def __init__(self, input_path):
        # data = []
        with open(input_path,encoding='utf-8') as f:
            data = json.load(f)
        print(f"loading {len(data)} data from {input_path}")
        self.data = data


    def __getitem__(self, index):
        item: dict = self.data[index]
        return item

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        # print(batch); exit()
        '''
        [id: '', title: '', description: '', QA_pairs: [
            {question: '', answer: ''},
            {question: '', answer: ''},
        ]]
        '''
        return batch

def get_dataloader_iterator(dataset_path,batch_size=1,start=1):
    dataset = MyDataset(dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )
    
    dataloader_iterator = (tqdm(dataloader, total=len(dataloader)))

    dataloader_iterator = islice(dataloader_iterator, start - 1, None)
    return dataloader_iterator

class Prompt():
    def __init__(self):
        print('\ndefining prompt')
    
    @property
    def system_prompt(self):
        return ""
    @property
    def instruction_template(self):
        return self.system_prompt + '问：{instruction}\n答：'
    @property
    def instruction_template_with_fewshot(self):
        return self.system_prompt + '{fewshot_examples}[Question]\n问：{instruction}\n答：'
        # return self.system_prompt + '{fewshot_examples}[Question]\n问：{instruction}\n用中文回答：'

    @property
    def fewshot_template(self,):
        return "[Example {round}]\n问：{user}\n答：{gpt}\n"





def _chat_stream(model, tokenizer, query, history,generation_config):
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    input_token_len = inputs["input_ids"].shape[-1]
    
    # 直接调用模型的generate函数，不使用流式输出
    outputs = model.generate(**inputs, generation_config=generation_config, num_return_sequences=1)
    
    # 解码生成的输出
    generated_text = tokenizer.decode(outputs[0][input_token_len:], skip_special_tokens=True)
    
    return generated_text    
    



def gen_with_psychgpt(output_pth,dataloader_iterator,prompt_wrapper,max_num=10000,start=0,modelname='psychgpt'):
    # chat_model = ChatModel()

    if modelname == 'psychfound-v1':
        model_path = "./model/checkpoints/psychfound_v1"

    experiment_config: dict = {
        'model_id_or_path': model_path,
        'context_length': 30720, # The context length for each experiment, 
        'output_len': 4096,
        'temperature': 0.1,
        'use_modelscope': False,
    }
    USE_FLASH_ATTN = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    
    attn_impl = 'flash_attention_2' if USE_FLASH_ATTN else 'eager'
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                torch_dtype='auto',
                                                # torch_dtype=torch.bfloat16,
                                                device_map='auto',
                                                attn_implementation=attn_impl
                                                ).eval()

    generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

    # generation_config.min_length = experiment_config['output_len'] + experiment_config['context_length']
    generation_config.max_new_tokens = experiment_config['output_len']
    generation_config.temperature = experiment_config['temperature']
    print(f'Generation config: {generation_config}')

    writer = open(output_pth, "a", encoding='utf-8')
    print('using model ',modelname)

    for batch_idx, batch in enumerate(dataloader_iterator, start=1):
        batch, lines = prompt_wrapper.wrap(batch)
        input_message = batch[0]
        if batch_idx >= max_num: return
        if batch_idx < start: continue
        print('\n',batch_idx)
        # print(batch[0])
        attempt = 0
        success = False

        while attempt < 3 and not success:  
            try:
                response = _chat_stream(model, tokenizer, query=input_message, history=[],generation_config=generation_config)
                # print(response)
                # response = response.split('assistant')[-1]
                print(response)
                line = lines[0]
                response = [response]
                for idx, _r in enumerate(response):
                    line[f"answer_{idx}"] = _r
                # print(response)
                # print(line)
                writer.write(json.dumps(line, ensure_ascii=False) + "\n")
                success = True  
            except Exception as e:  
                print(f"An error occurred: {e}")
                time.sleep(1) 
                attempt += 1 

        if not success:
            line = lines[0]
            print(f"Failed to process item {batch_idx} after 3 attempts.")
            # break
            line[f"answer_{idx}"] = 'failed!'
            # print(response)
            # print(line)
            writer.write(json.dumps(line, ensure_ascii=False) + "\n")

    # 删除模型实例，释放显存
    del model
    del tokenizer
    # 清除缓存
    torch.cuda.empty_cache()
    return



if __name__ == '__main__':
    
    ### define prompt template
    task1 = '''以下是患者的详细病历记录，请为病例讨论会准备一份简要总结，重点包括主诉、病史概要及病例特点。'''

    task2 = task3 = '''患者病历如下，请完成诊断分析，包括明确诊断、支持诊断的病史依据及鉴别诊断。'''

    task4 = '''根据患者的现有病情，给出最佳的精神科药物治疗建议。'''

    task5_qa_split = '''请阅读以下患者病程记录，然后回答问题。'''

    task5_lt_split = task5_mc_CMExam = ''''''


    USE_COT = False

    for TASK_ID in ['1','2','3','4','5','5_qa_split','5_mc_split','5_mc_CMExam']:

        SHOT_NUM = 0
        max_num = 101
        
        dataset_path = './data/fewshot{}_task{}/CMB-Exam-a-psychAiD.json'.format(SHOT_NUM,TASK_ID)

        
        print('-'*25,'Testing task{}'.format(TASK_ID),'-'*25)
        task_prefix = 'task_prefix'
        if USE_COT:
            # TASK_ID = TASK_ID + '_CoT'
            exec('task_prefix = task{}'.format(TASK_ID+ '_CoT'))
        else:
            exec('task_prefix = task{}'.format(TASK_ID))
        print('using prompt:')
        print(task_prefix)

        dataloader_iterator = get_dataloader_iterator(dataset_path)
        prompt_template = Prompt()
        prompt_wrapper = PromptWrapper(
            None,
            prompt_template.instruction_template_with_fewshot,
            conv_collater=None,
            use_cot=False,
            prompt_prefix = task_prefix
        )
        
        # run eval
        os.makedirs('./result/API/{}shot/'.format(SHOT_NUM),exist_ok=True)
        gen_with_psychgpt('./result/API/{}shot/task{}_psychfound.json'.format(SHOT_NUM,TASK_ID),dataloader_iterator,prompt_wrapper,max_num=10000,start=0,modelname='psychfound-v1')