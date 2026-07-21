from __future__ import annotations

import json
import random
from pathlib import Path

from psychfound.prompts import rl_prompt

from .io import iter_sharegpt


def prepare_rl_data(
    source: str | Path,
    output_dir: str | Path,
    *,
    task: str,
    precision: str,
    train_size: int,
    test_size: int,
    seed: int = 42,
    max_input_chars: int = 320,
) -> dict[str, object]:
    if task not in {"diagnosis", "medication"}:
        raise ValueError("task must be diagnosis or medication")
    if precision not in {"category", "subtype"}:
        raise ValueError("precision must be category or subtype")
    eligible = [row for row in iter_sharegpt(source) if len(row.patient_info) <= max_input_chars]
    requested = train_size + test_size
    if requested <= 0 or len(eligible) < requested:
        raise ValueError(f"requested {requested} examples but found {len(eligible)} eligible records")
    selected = random.Random(seed).sample(eligible, requested)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: list) -> None:
        with (destination / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                value = {
                    "prompt": rl_prompt(row.patient_info, task, precision),
                    "task": task,
                    "target": row.target,
                    "precision": precision,
                }
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    write("train", selected[:train_size])
    write("test", selected[train_size:])
    metadata = {
        "task": task, "precision": precision, "seed": seed,
        "train_records": train_size, "test_records": test_size,
        "source": str(Path(source).resolve()), "max_input_chars": max_input_chars,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
