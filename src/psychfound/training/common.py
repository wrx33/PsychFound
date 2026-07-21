from __future__ import annotations

from dataclasses import fields
from typing import Any


def supported_arguments(cls, values: dict[str, Any]) -> dict[str, Any]:
    """Keep configuration portable across compatible trainer patch releases."""
    names = {field.name for field in fields(cls)}
    unknown = sorted(values.keys() - names)
    if unknown:
        raise ValueError(f"Unsupported {cls.__name__} options: {', '.join(unknown)}")
    return values


def lora_config(values: dict[str, Any]):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(values.get("rank", 256)),
        lora_alpha=int(values.get("alpha", values.get("rank", 256) * 2)),
        lora_dropout=float(values.get("dropout", 0.0)),
        target_modules=values.get("target_modules", "all-linear"),
        bias="none",
    )
