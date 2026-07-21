from __future__ import annotations

from pathlib import Path

from .config import load_config


def generate(config_path: str | Path, prompt: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"], torch_dtype="auto", device_map="auto", trust_remote_code=False
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    settings = dict(config.get("generation", {}))
    with torch.inference_mode():
        output = model.generate(**inputs, **settings)
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

