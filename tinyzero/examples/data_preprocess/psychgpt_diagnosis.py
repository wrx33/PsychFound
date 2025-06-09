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

def make_prefix_icd(dp, template_type):
    diagnosis = dp['diagnosis']
    patient_info = dp['patient_info']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""一个User与Assistant之间的临床咨询。User提供患者的全面信息，包括病史，既往史，家族史和检验检查结果等信息，Assistant则提供诊断意见。Assistant首先分析提供的信息，考虑可能的病情，然后给出最可能的诊断。
User: 根据患者的病历信息：{patient_info}。按照ICD-10临床标准，该患者最可能的精神科诊断是什么？（部分ICD-10编码如下："F20.0 偏执型精神分裂症",
    "F20.1 青春型精神分裂症",
    "F20.2 紧张型精神分裂症",
    "F20.3 未分化型精神分裂症",
    "F20.4 精神分裂症后抑郁",
    "F20.5 残留型精神分裂症",
    "F20.6 单纯型精神分裂症",
    "F20.8 其他精神分裂症",
    "F20.9 精神分裂症，未特指",
    "F21 分裂型障碍",
    "F22.0 妄想性障碍",
    "F22.8 其他持续性妄想性障碍",
    "F22.9 持续性妄想性障碍，未特指",
    "F23.0 急性多形性精神病性障碍，不伴精神分裂症症状",
    "F23.1 急性多形性精神病性障碍，伴精神分裂症症状",
    "F23.2 急性精神分裂症样精神病性障碍",
    "F23.3 其他急性以妄想为主的精神病性障碍",
    "F23.8 其他急性短暂性精神病性障碍",
    "F23.9 急性短暂性精神病性障碍，未特指",
    "F24 感应性妄想性障碍",
    "F25.0 分裂情感性障碍，躁狂型",
    "F25.1 分裂情感性障碍，抑郁型",
    "F25.2 分裂情感性障碍，混合型",
    "F25.8 其他分裂情感性障碍",
    "F25.9 分裂情感性障碍，未特指",
    "F28 其他非器质性精神病性障碍",
    "F29 未特指的非器质性精神病",
    
    "F30.0 轻躁狂",
    "F30.1 躁狂，不伴精神病性症状",
    "F30.2 躁狂，伴精神病性症状",
    "F30.8 其他躁狂发作",
    "F30.9 躁狂发作，未特指",
    "F31.0 双相情感障碍，当前为轻躁狂发作",
    "F31.1 双相情感障碍，当前为不伴精神病性症状的躁狂发作",
    "F31.2 双相情感障碍，当前为伴精神病性症状的躁狂发作",
    "F31.3 双相情感障碍，当前为轻度或中度抑郁发作",
    "F31.4 双相情感障碍，当前为不伴精神病性症状的重度抑郁发作",
    "F31.5 双相情感障碍，当前为伴精神病性症状的重度抑郁发作",
    "F31.6 双相情感障碍，当前为混合发作",
    "F31.7 双相情感障碍，当前为缓解状态",
    "F31.8 其他双相情感障碍",
    "F31.9 双相情感障碍，未特指",
    "F32.0 轻度抑郁发作",
    "F32.1 中度抑郁发作",
    "F32.2 重度抑郁发作，不伴精神病性症状",
    "F32.3 重度抑郁发作，伴精神病性症状",
    "F32.8 其他抑郁发作",
    "F32.9 抑郁发作，未特指",
    "F33.0 复发性抑郁障碍，当前为轻度发作",
    "F33.1 复发性抑郁障碍，当前为中度发作",
    "F33.2 复发性抑郁障碍，当前为不伴精神病性症状的重度发作",
    "F33.3 复发性抑郁障碍，当前为伴精神病性症状的重度发作",
    "F33.4 复发性抑郁障碍，当前为缓解状态",
    "F33.8 其他复发性抑郁障碍",
    "F33.9 复发性抑郁障碍，未特指",
    "F34.0 环性心境",
    "F34.1 恶劣心境",
    "F34.8 其他持续性心境[情感]障碍",
    "F34.9 持续性心境[情感]障碍，未特指",
    "F38.0 其他单次发作的心境[情感]障碍",
    "F38.1 其他复发性心境[情感]障碍",
    "F38.8 其他特指的心境[情感]障碍",
    "F39 未特指的心境[情感]障碍"）。请解释你的推理过程。在<think> </think>标签中展示你的推理过程，并在<answer> </answer>标签中给出最终诊断。例如，<answer> 双相情感障碍，当前为混合发作 </answer>。
Assistant: 让我一步一步来解决这个问题。
<think>"""
        
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\n你是一个专业的精神科临床助手。在回答用户的问题时，应该首先在脑海中思考推理过程，然后向用户提供答案。<|im_end|>\n<|im_start|>user\n 根据患者的病历信息： {patient_info}。\n按照ICD-10临床标准，该患者最可能的精神科诊断是什么？（部分ICD-10编码如下："F20.0 偏执型精神分裂症",
    "F20.1 青春型精神分裂症",
    "F20.2 紧张型精神分裂症",
    "F20.3 未分化型精神分裂症",
    "F20.4 精神分裂症后抑郁",
    "F20.5 残留型精神分裂症",
    "F20.6 单纯型精神分裂症",
    "F20.8 其他精神分裂症",
    "F20.9 精神分裂症，未特指",
    "F21 分裂型障碍",
    "F22.0 妄想性障碍",
    "F22.8 其他持续性妄想性障碍",
    "F22.9 持续性妄想性障碍，未特指",
    "F23.0 急性多形性精神病性障碍，不伴精神分裂症症状",
    "F23.1 急性多形性精神病性障碍，伴精神分裂症症状",
    "F23.2 急性精神分裂症样精神病性障碍",
    "F23.3 其他急性以妄想为主的精神病性障碍",
    "F23.8 其他急性短暂性精神病性障碍",
    "F23.9 急性短暂性精神病性障碍，未特指",
    "F24 感应性妄想性障碍",
    "F25.0 分裂情感性障碍，躁狂型",
    "F25.1 分裂情感性障碍，抑郁型",
    "F25.2 分裂情感性障碍，混合型",
    "F25.8 其他分裂情感性障碍",
    "F25.9 分裂情感性障碍，未特指",
    "F28 其他非器质性精神病性障碍",
    "F29 未特指的非器质性精神病",
    
    "F30.0 轻躁狂",
    "F30.1 躁狂，不伴精神病性症状",
    "F30.2 躁狂，伴精神病性症状",
    "F30.8 其他躁狂发作",
    "F30.9 躁狂发作，未特指",
    "F31.0 双相情感障碍，当前为轻躁狂发作",
    "F31.1 双相情感障碍，当前为不伴精神病性症状的躁狂发作",
    "F31.2 双相情感障碍，当前为伴精神病性症状的躁狂发作",
    "F31.3 双相情感障碍，当前为轻度或中度抑郁发作",
    "F31.4 双相情感障碍，当前为不伴精神病性症状的重度抑郁发作",
    "F31.5 双相情感障碍，当前为伴精神病性症状的重度抑郁发作",
    "F31.6 双相情感障碍，当前为混合发作",
    "F31.7 双相情感障碍，当前为缓解状态",
    "F31.8 其他双相情感障碍",
    "F31.9 双相情感障碍，未特指",
    "F32.0 轻度抑郁发作",
    "F32.1 中度抑郁发作",
    "F32.2 重度抑郁发作，不伴精神病性症状",
    "F32.3 重度抑郁发作，伴精神病性症状",
    "F32.8 其他抑郁发作",
    "F32.9 抑郁发作，未特指",
    "F33.0 复发性抑郁障碍，当前为轻度发作",
    "F33.1 复发性抑郁障碍，当前为中度发作",
    "F33.2 复发性抑郁障碍，当前为不伴精神病性症状的重度发作",
    "F33.3 复发性抑郁障碍，当前为伴精神病性症状的重度发作",
    "F33.4 复发性抑郁障碍，当前为缓解状态",
    "F33.8 其他复发性抑郁障碍",
    "F33.9 复发性抑郁障碍，未特指",
    "F34.0 环性心境",
    "F34.1 恶劣心境",
    "F34.8 其他持续性心境[情感]障碍",
    "F34.9 持续性心境[情感]障碍，未特指",
    "F38.0 其他单次发作的心境[情感]障碍",
    "F38.1 其他复发性心境[情感]障碍",
    "F38.8 其他特指的心境[情感]障碍",
    "F39 未特指的心境[情感]障碍"）。请解释你的推理过程。在<think> </think>标签中展示你的推理过程，并在<answer> </answer>标签中给出最终诊断。例如，<answer> 双相情感障碍，当前为混合发作 </answer>。<|im_end|>\n<|im_start|>assistant\n让我一步一步来解决这个问题。\n<think>"""

    return prefix

def make_prefix_complex(dp, template_type):
    diagnosis = dp['diagnosis']
    patient_info = dp['patient_info']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""一个User与Assistant之间的临床咨询。User提供患者的全面信息，包括病史，既往史，家族史和检验检查结果等信息，Assistant则提供诊断意见。Assistant首先分析提供的信息，考虑可能的病情，然后给出最可能的诊断。
User: 根据患者的病历信息：{patient_info}。按照ICD-10临床标准，该患者最可能的精神科诊断是什么？（部分ICD-10编码如下："F20.0 偏执型精神分裂症","F20.1 青春型精神分裂症","F20.2 紧张型精神分裂症","F20.3 未分化型精神分裂症","F20.4 精神分裂症后抑郁","F20.5 残留型精神分裂症","F20.6 单纯型精神分裂症","F20.8 其他精神分裂症","F20.9 精神分裂症，未特指","F21 分裂型障碍","F22.0 妄想性障碍","F22.8 其他持续性妄想性障碍","F22.9 持续性妄想性障碍，未特指","F23.0 急性多形性精神病性障碍，不伴精神分裂症症状","F23.1 急性多形性精神病性障碍，伴精神分裂症症状","F23.2 急性精神分裂症样精神病性障碍","F23.3 其他急性以妄想为主的精神病性障碍","F23.8 其他急性短暂性精神病性障碍","F23.9 急性短暂性精神病性障碍，未特指","F24 感应性妄想性障碍","F25.0 分裂情感性障碍，躁狂型","F25.1 分裂情感性障碍，抑郁型","F25.2 分裂情感性障碍，混合型","F25.8 其他分裂情感性障碍","F25.9 分裂情感性障碍，未特指","F28 其他非器质性精神病性障碍","F29 未特指的非器质性精神病","F30.0 轻躁狂","F30.1 躁狂，不伴精神病性症状","F30.2 躁狂，伴精神病性症状","F30.8 其他躁狂发作","F30.9 躁狂发作，未特指","F31.0 双相情感障碍，当前为轻躁狂发作","F31.1 双相情感障碍，当前为不伴精神病性症状的躁狂发作","F31.2 双相情感障碍，当前为伴精神病性症状的躁狂发作","F31.3 双相情感障碍，当前为轻度或中度抑郁发作","F31.4 双相情感障碍，当前为不伴精神病性症状的重度抑郁发作","F31.5 双相情感障碍，当前为伴精神病性症状的重度抑郁发作","F31.6 双相情感障碍，当前为混合发作","F31.7 双相情感障碍，当前为缓解状态","F31.8 其他双相情感障碍","F31.9 双相情感障碍，未特指","F32.0 轻度抑郁发作","F32.1 中度抑郁发作","F32.2 重度抑郁发作，不伴精神病性症状","F32.3 重度抑郁发作，伴精神病性症状","F32.8 其他抑郁发作","F32.9 抑郁发作，未特指","F33.0 复发性抑郁障碍，当前为轻度发作","F33.1 复发性抑郁障碍，当前为中度发作","F33.2 复发性抑郁障碍，当前为不伴精神病性症状的重度发作","F33.3 复发性抑郁障碍，当前为伴精神病性症状的重度发作","F33.4 复发性抑郁障碍，当前为缓解状态","F33.8 其他复发性抑郁障碍","F33.9 复发性抑郁障碍，未特指","F34.0 环性心境","F34.1 恶劣心境","F34.8 其他持续性心境[情感]障碍","F34.9 持续性心境[情感]障碍，未特指","F38.0 其他单次发作的心境[情感]障碍","F38.1 其他复发性心境[情感]障碍","F38.8 其他特指的心境[情感]障碍","F39 未特指的心境[情感]障碍"）。请解释你的推理过程。在<think> </think>标签中展示你的推理过程，并在<answer> </answer>标签中给出最终诊断。例如：<answer>F20.0 偏执型精神分裂症</answer>
Assistant: 让我一步一步来解决这个问题。
<think>"""
        
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\n你是一个专业的精神科临床助手。在回答用户的问题时，应该首先结合专业临床知识思考推理过程，然后向用户提供可靠答案。<|im_end|>\n<|im_start|>user\n 根据患者的病历信息： {patient_info}。\n按照ICD-10临床标准，该患者最可能的精神科诊断是什么？（部分ICD-10编码如下："F20.0 偏执型精神分裂症","F20.1 青春型精神分裂症","F20.2 紧张型精神分裂症","F20.3 未分化型精神分裂症","F20.4 精神分裂症后抑郁","F20.5 残留型精神分裂症","F20.6 单纯型精神分裂症","F20.8 其他精神分裂症","F20.9 精神分裂症，未特指","F21 分裂型障碍","F22.0 妄想性障碍","F22.8 其他持续性妄想性障碍","F22.9 持续性妄想性障碍，未特指","F23.0 急性多形性精神病性障碍，不伴精神分裂症症状","F23.1 急性多形性精神病性障碍，伴精神分裂症症状","F23.2 急性精神分裂症样精神病性障碍","F23.3 其他急性以妄想为主的精神病性障碍","F23.8 其他急性短暂性精神病性障碍","F23.9 急性短暂性精神病性障碍，未特指","F24 感应性妄想性障碍","F25.0 分裂情感性障碍，躁狂型","F25.1 分裂情感性障碍，抑郁型","F25.2 分裂情感性障碍，混合型","F25.8 其他分裂情感性障碍","F25.9 分裂情感性障碍，未特指","F28 其他非器质性精神病性障碍","F29 未特指的非器质性精神病","F30.0 轻躁狂","F30.1 躁狂，不伴精神病性症状","F30.2 躁狂，伴精神病性症状","F30.8 其他躁狂发作","F30.9 躁狂发作，未特指","F31.0 双相情感障碍，当前为轻躁狂发作","F31.1 双相情感障碍，当前为不伴精神病性症状的躁狂发作","F31.2 双相情感障碍，当前为伴精神病性症状的躁狂发作","F31.3 双相情感障碍，当前为轻度或中度抑郁发作","F31.4 双相情感障碍，当前为不伴精神病性症状的重度抑郁发作","F31.5 双相情感障碍，当前为伴精神病性症状的重度抑郁发作","F31.6 双相情感障碍，当前为混合发作","F31.7 双相情感障碍，当前为缓解状态","F31.8 其他双相情感障碍","F31.9 双相情感障碍，未特指","F32.0 轻度抑郁发作","F32.1 中度抑郁发作","F32.2 重度抑郁发作，不伴精神病性症状","F32.3 重度抑郁发作，伴精神病性症状","F32.8 其他抑郁发作","F32.9 抑郁发作，未特指","F33.0 复发性抑郁障碍，当前为轻度发作","F33.1 复发性抑郁障碍，当前为中度发作","F33.2 复发性抑郁障碍，当前为不伴精神病性症状的重度发作","F33.3 复发性抑郁障碍，当前为伴精神病性症状的重度发作","F33.4 复发性抑郁障碍，当前为缓解状态","F33.8 其他复发性抑郁障碍","F33.9 复发性抑郁障碍，未特指","F34.0 环性心境","F34.1 恶劣心境","F34.8 其他持续性心境[情感]障碍","F34.9 持续性心境[情感]障碍，未特指","F38.0 其他单次发作的心境[情感]障碍","F38.1 其他复发性心境[情感]障碍","F38.8 其他特指的心境[情感]障碍","F39 未特指的心境[情感]障碍"）。请解释你的思考推理过程。在<think> </think>标签中展示你的推理过程，并在<answer> </answer>标签中给出最终诊断。例如：<answer>F20.0 偏执型精神分裂症</answer> <|im_end|>\n<|im_start|>assistant\n让我一步一步来解决这个问题。\n<think>"""
    return prefix

def make_prefix_simple(dp, template_type):
    diagnosis = dp['diagnosis']
    patient_info = dp['patient_info']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""一个User与Assistant之间的临床咨询。User提供患者的全面信息，包括病史，既往史，家族史和检验检查结果等信息，Assistant则提供诊断意见。Assistant首先分析提供的信息，考虑可能的病情，然后给出最可能的诊断。
User: 根据患者的病历信息：{patient_info}。按照ICD-10临床标准，该患者最可能的精神科诊断是什么？（部分ICD-10编码如下："F20 精神分裂症","F21 分裂型障碍","F22 妄想性障碍","F23 急性短暂性精神病性障碍","F24 感应性妄想性障碍","F25 分裂情感性障碍","F28 其他非器质性精神病性障碍","F29 未特指的非器质性精神病","F30 躁狂发作","F31 双相情感障碍","F32 抑郁发作","F33 复发性抑郁障碍","F34 持续性心境[情感]障碍","F38 其他心境[情感]障碍","F39 未特指的心境[情感]障碍"）。请解释你的推理过程。在<think> </think>标签中展示你的推理过程，并在<answer> </answer>标签中给出最终诊断。例如：<answer>F20 精神分裂症</answer>
Assistant: 让我一步一步来解决这个问题。
<think>"""
        
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\n你是一个专业的精神科临床助手。在回答用户的问题时，应该首先结合专业临床知识思考推理过程，然后向用户提供可靠答案。<|im_end|>\n<|im_start|>user\n 根据患者的病历信息： {patient_info}。\n按照ICD-10临床标准，该患者最可能的精神科诊断是什么？（部分ICD-10编码如下："F20 精神分裂症","F21 分裂型障碍","F22 妄想性障碍","F23 急性短暂性精神病性障碍","F24 感应性妄想性障碍","F25 分裂情感性障碍","F28 其他非器质性精神病性障碍","F29 未特指的非器质性精神病","F30 躁狂发作","F31 双相情感障碍","F32 抑郁发作","F33 复发性抑郁障碍","F34 持续性心境[情感]障碍","F38 其他心境[情感]障碍","F39 未特指的心境[情感]障碍"）。请解释你的思考推理过程。在<think> </think>标签中展示你的推理过程，并在<answer> </answer>标签中给出最终诊断。例如：<answer>F20 精神分裂症</answer> <|im_end|>\n<|im_start|>assistant\n让我一步一步来解决这个问题。\n<think>"""

    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/sjtu/wrx/code/TinyZero-main/data/diagnosis-instruct-hint-icd10-0224-1200-shuffle-800-simple')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=800)
    parser.add_argument('--test_size', type=int, default=80)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    parser.add_argument('--data_path', type=str, default="/home/sjtu/wrx/code/TinyZero-main/data/task2-diagnosis-icd10.jsonl")

    args = parser.parse_args()

    data_source = 'psychgpt_diagnosis'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                content = json.loads(line)
                content_format = {"diagnosis": content['conversations'][1]['value'], "patient_info": content['conversations'][0]['value']}
                
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
            question = make_prefix_simple(example, template_type=args.template_type)
            solution = {
                "diagnosis": example['diagnosis'],
                "patient_info": example['patient_info']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "diagnosis",
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
