from __future__ import annotations

from pathlib import Path


def merge_adapter(
    base_model_path: str | Path,
    adapter_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Merge a causal-LM LoRA adapter into its base checkpoint on CPU."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_path), torch_dtype="auto", device_map="cpu", trust_remote_code=False
    )
    merged = PeftModel.from_pretrained(base, str(adapter_path)).merge_and_unload()
    merged.save_pretrained(destination, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=False)
    tokenizer.save_pretrained(destination)
    return {
        "base_model": str(base_model_path),
        "adapter": str(adapter_path),
        "output_dir": str(destination),
    }
