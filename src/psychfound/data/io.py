from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ClinicalExample:
    patient_info: str
    target: str


def _turns(record: object, line: int) -> list[dict[str, object]]:
    if isinstance(record, list):
        value = record
    elif isinstance(record, dict) and isinstance(record.get("conversations"), list):
        value = record["conversations"]
    else:
        raise ValueError(f"line {line}: expected a list or a conversations field")
    if not all(isinstance(turn, dict) for turn in value):
        raise ValueError(f"line {line}: every turn must be an object")
    return value


def iter_sharegpt(path: str | Path) -> Iterator[ClinicalExample]:
    with Path(path).open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {number}: invalid JSON ({exc.msg})") from exc
            turns = _turns(record, number)
            human = next((str(x.get("value", "")).strip() for x in turns if x.get("from") == "human"), "")
            assistant = next((str(x.get("value", "")).strip() for x in turns if x.get("from") == "gpt"), "")
            if not human or not assistant:
                raise ValueError(f"line {number}: one non-empty human/gpt pair is required")
            yield ClinicalExample(human, assistant)


def validate_sharegpt(path: str | Path) -> dict[str, int]:
    rows = list(iter_sharegpt(path))
    if not rows:
        raise ValueError("dataset is empty")
    return {
        "records": len(rows),
        "max_input_chars": max(map(lambda row: len(row.patient_info), rows)),
        "max_target_chars": max(map(lambda row: len(row.target), rows)),
    }


def to_messages(example: ClinicalExample) -> dict[str, object]:
    return {"messages": [
        {"role": "user", "content": example.patient_info},
        {"role": "assistant", "content": example.target},
    ]}

