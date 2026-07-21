import json
from pathlib import Path

import pytest

from psychfound.data import iter_sharegpt, prepare_rl_data, validate_sharegpt


def write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_validate_and_prepare_rl(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"conversations": [{"from": "human", "value": f"病例 {i}"}, {"from": "gpt", "value": "F31.4 双相情感障碍"}]}
        for i in range(5)
    ]
    write_jsonl(source, rows)
    assert validate_sharegpt(source)["records"] == 5
    metadata = prepare_rl_data(source, tmp_path / "rl", task="diagnosis", precision="category",
                               train_size=3, test_size=1, seed=7)
    assert metadata["train_records"] == 3
    prepared = json.loads((tmp_path / "rl/train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert prepared["task"] == "diagnosis"
    assert isinstance(prepared["prompt"], str)
    assert prepared["prompt"].startswith("<|im_start|>system")
    assert prepared["prompt"].endswith("<think>")


def test_invalid_role_is_rejected(tmp_path):
    source = tmp_path / "bad.jsonl"
    write_jsonl(source, [[{"from": "huamn", "value": "x"}, {"from": "gpt", "value": "y"}]])
    with pytest.raises(ValueError, match="human/gpt"):
        list(iter_sharegpt(source))
