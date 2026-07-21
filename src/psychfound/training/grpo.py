from __future__ import annotations

import os
from pathlib import Path

from psychfound.config import load_config
from psychfound.rewards import clinical_reward

from .common import supported_arguments


def train_grpo(config_path: str | Path, *, dry_run: bool = False) -> dict[str, object]:
    config = load_config(config_path, strict_env=not dry_run)
    summary = {
        "stage": config.get("stage", "grpo"),
        "model": config["model_name_or_path"],
        "train_data": config["train_data"],
        "output_dir": config["output_dir"],
        "reward_weights": [1, 1, 2],
        "model_output": config["output_dir"],
    }
    if dry_run:
        return summary

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    train_dataset = load_dataset("json", data_files=config["train_data"], split="train")
    eval_dataset = None
    if config.get("eval_data"):
        eval_dataset = load_dataset("json", data_files=config["eval_data"], split="train")
    values = dict(config.get("training", {}))
    values.setdefault("output_dir", config["output_dir"])
    values.setdefault("remove_unused_columns", False)
    values.setdefault("report_to", "none")
    max_prompt_length = int(values.pop("max_prompt_length", 512))
    expected_world_size = int(config.get("expected_world_size", 8))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != expected_world_size and not config.get("allow_world_size_mismatch", False):
        raise ValueError(
            f"GRPO batch semantics require WORLD_SIZE={expected_world_size}; got {world_size}. "
            "Set allow_world_size_mismatch only for intentional ablations."
        )
    completions_per_device = int(values.get("per_device_train_batch_size", 1))
    generations = int(values.get("num_generations", 1))
    global_prompt_batch = completions_per_device * world_size // generations
    expected_prompt_batch = int(config.get("global_prompt_batch_size", 8))
    if global_prompt_batch != expected_prompt_batch:
        raise ValueError(
            f"Effective prompt batch is {global_prompt_batch}, expected {expected_prompt_batch}; "
            "per-device batch counts generated completions in TRL."
        )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    for split_name, dataset in (("train", train_dataset), ("eval", eval_dataset)):
        if dataset is None:
            continue
        overflow = [
            index
            for index, prompt in enumerate(dataset["prompt"])
            if len(tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_length
        ]
        if overflow:
            raise ValueError(
                f"{split_name} contains {len(overflow)} prompts longer than max_prompt_length="
                f"{max_prompt_length}; first index: {overflow[0]}. RL prompts are not silently truncated."
            )

    args = GRPOConfig(**supported_arguments(GRPOConfig, values))
    trainer = GRPOTrainer(
        model=config["model_name_or_path"],
        reward_funcs=clinical_reward,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(config["output_dir"])
    summary["records"] = len(train_dataset)
    return summary
