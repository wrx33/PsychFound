from __future__ import annotations

from pathlib import Path
from typing import Any

from psychfound.config import load_config
from psychfound.data.io import ClinicalExample, iter_sharegpt

from .common import lora_config, supported_arguments
from .merge import merge_adapter

IGNORE_INDEX = -100
QWEN_DEFAULT_SYSTEM = "You are a helpful assistant."


def qwen_prompt_and_completion(
    example: ClinicalExample,
    *,
    eos_token: str,
    default_system: str = QWEN_DEFAULT_SYSTEM,
) -> tuple[str, str]:
    """Render the one-turn Qwen ChatML layout used throughout PsychFound SFT."""
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        f"<|im_start|>user\n{example.patient_info}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt, f"{example.target}{eos_token}"


def encode_supervised_example(
    example: ClinicalExample,
    tokenizer: Any,
    *,
    max_prompt_length: int,
    max_completion_length: int,
    default_system: str = QWEN_DEFAULT_SYSTEM,
) -> dict[str, list[int]]:
    """Tokenize separately so prompt tokens are masked and truncation matches the original implementation."""
    if not tokenizer.eos_token or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token and eos_token_id (Qwen uses <|im_end|>).")
    prompt, _ = qwen_prompt_and_completion(
        example, eos_token=tokenizer.eos_token, default_system=default_system
    )
    source_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Encode the response independently so every prompt token can be masked.
    target_ids = tokenizer.encode(example.target, add_special_tokens=False) + [tokenizer.eos_token_id]
    source_ids = source_ids[:max_prompt_length]
    target_ids = target_ids[:max_completion_length]
    input_ids = source_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(source_ids) + target_ids,
    }


class SupervisedDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]):
        import torch

        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [IGNORE_INDEX] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_sft(config_path: str | Path, *, dry_run: bool = False) -> dict[str, object]:
    config = load_config(config_path, strict_env=not dry_run)
    summary = {
        "stage": config.get("stage", "sft"),
        "model": config["model_name_or_path"],
        "dataset": config["dataset_path"],
        "output_dir": config["output_dir"],
        "method": "LoRA",
        "model_output": config.get("merged_output_dir", config["output_dir"]),
    }
    if dry_run:
        return summary

    from datasets import Dataset
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    rows = list(iter_sharegpt(config["dataset_path"]))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    values = dict(config.get("training", {}))
    values.setdefault("output_dir", config["output_dir"])
    values.setdefault("report_to", "none")
    max_prompt_length = int(values.pop("max_prompt_length", 4096))
    max_completion_length = int(values.pop("max_completion_length", 2048))
    default_system = str(config.get("template", {}).get("default_system", QWEN_DEFAULT_SYSTEM))
    encoded = [
        encode_supervised_example(
            row,
            tokenizer,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            default_system=default_system,
        )
        for row in rows
    ]
    dataset = Dataset.from_list(encoded)
    validation_size = float(config.get("validation_size", 0.0))
    if validation_size:
        split = dataset.train_test_split(test_size=validation_size, seed=int(config.get("seed", 42)))
        train_dataset, eval_dataset = split["train"], split["test"]
        values.setdefault("eval_strategy", "steps")
    else:
        train_dataset, eval_dataset = dataset, None
        values.pop("eval_strategy", None)
        values.pop("eval_steps", None)

    dtype = torch.bfloat16 if values.get("bf16") else None
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"], torch_dtype=dtype, trust_remote_code=False
    )
    if values.get("gradient_checkpointing"):
        model.config.use_cache = False
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(config.get("lora", {})))
    args = TrainingArguments(**supported_arguments(TrainingArguments, values))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(config["output_dir"])
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config["output_dir"])
    trainer.accelerator.wait_for_everyone()
    if config.get("merged_output_dir") and trainer.is_world_process_zero():
        merge_adapter(config["model_name_or_path"], config["output_dir"], config["merged_output_dir"])
    trainer.accelerator.wait_for_everyone()
    summary["records"] = len(rows)
    return summary
