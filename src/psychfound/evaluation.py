from __future__ import annotations

import json
import re
from pathlib import Path

from .rewards import score


def evaluate_predictions(path: str | Path) -> dict[str, object]:
    totals = {"records": 0, "format": 0.0, "reasoning": 0.0, "accuracy": 0.0, "reward": 0.0}
    by_task: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            required = {"prediction", "reference", "task"}
            if not required <= row.keys():
                raise ValueError(f"line {number}: missing {', '.join(sorted(required - row.keys()))}")
            result = score(row["prediction"], row["reference"], row["task"], row.get("precision", "subtype"))
            totals["records"] += 1
            totals["format"] += result.format
            totals["reasoning"] += result.reasoning
            totals["accuracy"] += result.accuracy
            totals["reward"] += result.total
            task = by_task.setdefault(row["task"], {"records": 0, "accuracy": 0.0})
            task["records"] += 1
            task["accuracy"] += result.accuracy
    if not totals["records"]:
        raise ValueError("prediction file is empty")
    count = totals["records"]
    summary = {key: (value / count if key != "records" else value) for key, value in totals.items()}
    summary["by_task"] = {
        name: {"records": int(values["records"]), "accuracy": values["accuracy"] / values["records"]}
        for name, values in by_task.items()
    }
    return summary

