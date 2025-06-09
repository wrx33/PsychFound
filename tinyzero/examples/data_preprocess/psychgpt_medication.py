"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""
import sys
sys.path.append('../../')

import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
# from tinyzero.verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import os
import shutil
import logging
import random

def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    r"""Works like os.makedirs() but supports hdfs.

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.

    Args:
        name (str): directory to create
        mode (int): file mode bits
        exist_ok (bool): if True, do not raise an exception if the directory already exists
        kwargs: keyword arguments for hdfs

    """
    if _is_non_local(name):
        # TODO(haibin.lin):
        # - handle OSError for hdfs(?)
        # - support exist_ok for hdfs(?)
        _mkdir(name, **kwargs)
    else:
        os.makedirs(name, mode=mode, exist_ok=exist_ok)


def _mkdir(file_path: str) -> bool:
    """hdfs mkdir"""
    if file_path.startswith("hdfs"):
        _run_cmd(_hdfs_cmd(f"-mkdir -p {file_path}"))
    else:
        os.makedirs(file_path, exist_ok=True)
    return True

def copy(src: str, dst: str, **kwargs) -> bool:
    r"""Works like shutil.copy() for file, and shutil.copytree for dir, and supports hdfs.

    Copy data and mode bits ("cp src dst"). Return the file's destination.
    The destination may be a directory.
    If source and destination are the same file, a SameFileError will be
    raised.

    Arg:
        src (str): source file path
        dst (str): destination file path
        kwargs: keyword arguments for hdfs copy

    Returns:
        str: destination file path

    """
    if _is_non_local(src) or _is_non_local(dst):
        # TODO(haibin.lin):
        # - handle SameFileError for hdfs files(?)
        # - return file destination for hdfs files
        return _copy(src, dst)
    else:
        if os.path.isdir(src):
            return shutil.copytree(src, dst, **kwargs)
        else:
            return shutil.copy(src, dst, **kwargs)

def gen_dataset(
    num_samples: int,
    num_operands: int = 6,
    max_target: int = 1000,
    min_number: int = 1,
    max_number: int = 100,
    operations: List[str] = ['+', '-', '*', '/'],
    seed_value: int = 42,
) -> List[Tuple]:
    """Generate dataset for countdown task.
    
    Args:
        num_samples: Number of samples to generate
        num_operands: Number of numbers provided in each sample
        max_target: Maximum value for target number
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        operations: List of allowed operations
        seed_value: Random seed for reproducibility
        
    Returns:
        List of tuples containing (target, numbers, solution)
    """
    seed(seed_value)
    samples = []
    
    for _ in tqdm(range(num_samples)):
        # Generate random target
        target = randint(1, max_target)
        
        # Generate random numbers
        numbers = [randint(min_number, max_number) for _ in range(num_operands)]
        
        
        samples.append((target, numbers))
    
    return samples

def make_prefix_en(dp, template_type):
    diagnosis = dp['diagnosis']
    patient_info = dp['patient_info']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A clinical consultation between a User and an Assistant. The user provides the patient's comprehensive information, including medical history, past medical history, family history, and test results, etc, and the Assistant provides a diagnosis. The assistant first analyzes the information provided, considers possible conditions, and then suggests the most likely diagnosis. 
User: Given the patient's information {patient_info}, what is the most likely diagnosis? Please explain your reasoning. Show your reasoning process in <think> </think> tags. And return the final diagnosis in <answer> </answer> tags. For example, <answer> 双相情感障碍，目前为混合状态 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful clinical assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Given the patient's information {patient_info}, what is the most likely diagnosis? Please explain your reasoning. Show your reasoning process in <think> </think> tags. And return the final diagnosis in <answer> </answer> tags. For example, <answer> 双相情感障碍，目前为混合状态 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix



def make_prefix(dp, template_type):
    medication = dp['medication']
    patient_info = dp['patient_info']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""一个User与Assistant之间的临床咨询。User提供患者的全面信息，包括病史，既往史，家族史和检验检查结果等信息，Assistant则提供用药建议。Assistant首先分析提供的信息，考虑可能的病情，然后给出最适合的用药建议。
User: 根据患者的病历信息： {patient_info}。\n该患者最适合的精神科用药方案是什么？（可选的药物包括但不限于：抗抑郁药：阿戈美拉汀，草酸艾司西酞普兰，氟伏沙明，氟硫西汀，西酞普兰，阿米替林，多塞平，氟西汀，氯米帕明，米那普仑，帕罗西汀，曲唑酮，文拉法辛，氟哌噻吨，米氮平，安非他酮，度洛西汀，马普替林，米安色林，舍曲林。抗精神病药：阿立哌唑，氨磺必利，奥氮平，布南色林，奋乃静，氟哌啶醇，喹硫平，利培酮，氯氮平，帕利哌酮，舒必利，五氟利多，硫必利，鲁拉西酮，氯丙嗪，哌甲酯，哌罗匹隆，齐拉西酮，托莫西汀。抗焦虑药：坦度螺酮，普瑞巴林，丁螺环酮。镇静、抗躁狂类药物：丙戊酸钠，碳酸锂，苯海索，阿普唑仑，艾司唑仑，奥沙西泮，苯巴比妥，地西泮，酒石酸唑吡坦，劳拉西泮，氯硝西泮，马来酸咪达唑仑，硝西泮，佐匹克隆，扎来普隆，卡马西平，拉莫三嗪。）。请详细解释你的思考推理过程。在<think> </think>标签中展示你的推理过程，并在<answer> </answer>标签中给出最终用药建议。例如：<answer>丙戊酸, 阿立哌唑, 艾司西酞普兰</answer>
Assistant: 让我一步一步来解决这个问题。
<think>"""
        
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\n你是一个专业的精神科临床助手。在回答用户的问题时，应该首先结合专业临床知识思考推理过程，然后向用户提供可靠答案。<|im_end|>\n<|im_start|>user\n 根据患者的病历信息： {patient_info}。\n该患者最适合的精神科用药方案是什么？（可选的药物包括但不限于：抗抑郁药：阿戈美拉汀，草酸艾司西酞普兰，氟伏沙明，氟硫西汀，西酞普兰，阿米替林，多塞平，氟西汀，氯米帕明，米那普仑，帕罗西汀，曲唑酮，文拉法辛，氟哌噻吨，米氮平，安非他酮，度洛西汀，马普替林，米安色林，舍曲林。抗精神病药：阿立哌唑，氨磺必利，奥氮平，布南色林，奋乃静，氟哌啶醇，喹硫平，利培酮，氯氮平，帕利哌酮，舒必利，五氟利多，硫必利，鲁拉西酮，氯丙嗪，哌甲酯，哌罗匹隆，齐拉西酮，托莫西汀。抗焦虑药：坦度螺酮，普瑞巴林，丁螺环酮。镇静、抗躁狂类药物：丙戊酸，碳酸锂，苯海索，阿普唑仑，艾司唑仑，奥沙西泮，苯巴比妥，地西泮，酒石酸唑吡坦，劳拉西泮，氯硝西泮，马来酸咪达唑仑，硝西泮，佐匹克隆，扎来普隆，卡马西平，拉莫三嗪。）。请详细解释你的思考推理过程。在<think> </think>标签中展示你的推理过程，并在<answer> </answer>标签中给出最终用药建议。例如：<answer>丙戊酸, 阿立哌唑, 艾司西酞普兰</answer> <|im_end|>\n<|im_start|>assistant\n让我一步一步来解决这个问题。\n<think>"""

    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/sjtu/wrx/code/TinyZero-main/data/medication-instruct-hint-0228-1200-shuffle-2000')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=200)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    parser.add_argument('--data_path', type=str, default="/home/sjtu/wrx/code/LLaMA-Factory-0729/mydata/psychgpt_sft_0805/task4_new.jsonl")

    args = parser.parse_args()

    data_source = 'psychgpt_medication'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                content = json.loads(line)
                content_format = {"medication": content['conversations'][1]['value'], "patient_info": content['conversations'][0]['value']}
                
                if len(content_format['patient_info']) > 1200:
                    continue
                else:
                    yield content_format
                # yield json.loads(line)
    
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(len(raw_dataset))

    random_indices = random.sample(range(len(raw_dataset)), TRAIN_SIZE + TEST_SIZE)
    train_indices = random_indices[:TRAIN_SIZE]
    test_indices = random_indices[TRAIN_SIZE:]

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(train_indices)
    test_dataset = raw_dataset.select(test_indices)

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "medication": example['medication'],
                "patient_info": example['patient_info']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "medication",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
