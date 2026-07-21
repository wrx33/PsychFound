from __future__ import annotations

import re
from dataclasses import dataclass

_OUTPUT = re.compile(r"^\s*<think>(?P<think>.+?)</think>\s*<answer>(?P<answer>.+?)</answer>\s*$", re.DOTALL)
_ICD = re.compile(r"\bF\d{2}(?:\.\d{1,2})?\b", re.IGNORECASE)
_SEP = re.compile(r"[,，、;；/\n]+")
KEYWORDS = {
    "diagnosis": ("症状", "病程", "严重", "功能", "排除", "鉴别"),
    "medication": ("症状", "诊断", "既往", "合并", "禁忌", "相互作用", "副作用", "不良反应"),
}


@dataclass(frozen=True)
class Reward:
    format: float
    reasoning: float
    accuracy: float

    @property
    def total(self) -> float:
        return self.format + self.reasoning + 2 * self.accuracy


def _text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


def _drug_set(value: str) -> set[str]:
    return {re.sub(r"\s+", "", part).strip("。.") for part in _SEP.split(value) if part.strip()}


def score(completion: object, target: str, task: str, precision: str = "subtype") -> Reward:
    match = _OUTPUT.fullmatch(_text(completion))
    if not match:
        return Reward(0.0, 0.0, 0.0)
    think, answer = match.group("think"), match.group("answer")
    reasoning = sum(word in think for word in KEYWORDS[task]) / len(KEYWORDS[task])
    if task == "diagnosis":
        predicted, expected = _ICD.search(answer), _ICD.search(target)
        if not predicted or not expected:
            accuracy = 0.0
        elif precision == "category":
            accuracy = float(predicted.group(0).upper().split(".")[0] == expected.group(0).upper().split(".")[0])
        else:
            accuracy = float(predicted.group(0).upper() == expected.group(0).upper())
    else:
        predicted, expected = _drug_set(answer), _drug_set(target)
        if not predicted or not expected:
            accuracy = 0.0
        elif precision == "category":
            accuracy = len(predicted & expected) / len(expected)
        else:
            p, r = len(predicted & expected) / len(predicted), len(predicted & expected) / len(expected)
            accuracy = 2 * p * r / (p + r) if p + r else 0.0
    return Reward(1.0, reasoning, accuracy)


def clinical_reward(completions, target, task, precision, **kwargs):
    del kwargs
    values = []
    for completion, expected, task_name, level in zip(completions, target, task, precision):
        rendered = _text(completion)
        if not rendered.lstrip().startswith("<think>"):
            rendered = "<think>" + rendered
        values.append(score(rendered, expected, task_name, level).total)
    return values
