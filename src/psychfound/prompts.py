from __future__ import annotations

DIAGNOSIS_CATEGORIES = (
    "F20 精神分裂症", "F21 分裂型障碍", "F22 妄想性障碍", "F23 急性短暂性精神病性障碍",
    "F24 感应性妄想性障碍", "F25 分裂情感性障碍", "F28 其他非器质性精神病性障碍",
    "F29 未特指的非器质性精神病", "F30 躁狂发作", "F31 双相情感障碍", "F32 抑郁发作",
    "F33 复发性抑郁障碍", "F34 持续性心境[情感]障碍", "F38 其他心境[情感]障碍",
    "F39 未特指的心境[情感]障碍",
)

MEDICATION_CATEGORIES = (
    "抗抑郁药", "抗精神病药", "抗焦虑药", "镇静催眠药", "心境稳定剂", "其他精神科用药",
)


def rl_prompt(patient_info: str, task: str, precision: str) -> str:
    """Render the pre-templated Qwen prompt used for clinical GRPO rollouts."""
    if task == "diagnosis":
        noun = "诊断"
        question = "按照ICD-10临床标准，该患者最可能的精神科诊断是什么？"
        if precision == "category":
            example = "F20 精神分裂症"
            choice_text = "（诊断类别包括：" + "，".join(DIAGNOSIS_CATEGORIES) + "）。"
        else:
            example = "F20.0 偏执型精神分裂症"
            choice_text = "请给出完整的ICD-10亚型编码和诊断名称。"
    elif task == "medication":
        noun = "用药建议"
        question = "该患者最适合的精神科用药方案是什么？"
        if precision == "category":
            example = "抗抑郁药, 抗精神病药"
            choice_text = "（药物类别包括：" + "，".join(MEDICATION_CATEGORIES) + "）。"
        else:
            example = "丙戊酸, 阿立哌唑, 艾司西酞普兰"
            choice_text = "请给出药物通用名称。"
    else:
        raise ValueError(f"Unsupported task: {task}")
    system = (
        "你是一个专业的精神科临床助手。在回答用户的问题时，应该首先结合专业临床知识"
        "思考推理过程，然后向用户提供可靠答案。"
    )
    user = (
        f"根据患者的病历信息：{patient_info.strip()}。\n{question}{choice_text}"
        f"请详细解释你的思考推理过程。在<think> </think>标签中展示你的推理过程，"
        f"并在<answer> </answer>标签中给出最终{noun}。例如：<answer>{example}</answer>"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n让我一步一步来解决这个问题。\n<think>"
    )
