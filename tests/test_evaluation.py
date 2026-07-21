import json

from psychfound.evaluation import evaluate_predictions


def test_evaluation_summary(tmp_path):
    path = tmp_path / "predictions.jsonl"
    row = {
        "task": "diagnosis", "precision": "subtype", "reference": "F31.4 双相情感障碍",
        "prediction": "<think>症状、病程、严重程度、功能、排除、鉴别。</think><answer>F31.4 双相情感障碍</answer>",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = evaluate_predictions(path)
    assert summary["accuracy"] == 1.0
    assert summary["by_task"]["diagnosis"]["records"] == 1
