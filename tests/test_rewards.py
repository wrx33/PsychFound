from psychfound.rewards import clinical_reward, score


def test_diagnosis_reward_weights():
    completion = (
        "<think>综合症状、病程、严重程度、功能损害、排除条件和鉴别诊断。</think>"
        "<answer>F31.4 双相情感障碍</answer>"
    )
    result = score(completion, "F31.4 双相情感障碍", "diagnosis")
    assert result.format == result.reasoning == result.accuracy == 1.0
    assert result.total == 4.0


def test_progressive_diagnosis_precision():
    completion = "<think>分析症状与病程。</think><answer>F31.4 双相情感障碍</answer>"
    assert score(completion, "F31.5 双相情感障碍", "diagnosis", "category").accuracy == 1.0
    assert score(completion, "F31.5 双相情感障碍", "diagnosis", "subtype").accuracy == 0.0


def test_medication_f1_is_order_independent():
    completion = "<think>分析诊断、症状、既往、合并、禁忌、相互作用、副作用和不良反应。</think><answer>喹硫平，碳酸锂</answer>"
    assert score(completion, "碳酸锂,喹硫平", "medication").total == 4.0


def test_trainer_reward_signature():
    values = clinical_reward(
        ["<think>症状</think><answer>F20.0</answer>"], ["F20.0"], ["diagnosis"], ["subtype"]
    )
    assert len(values) == 1 and values[0] > 0


def test_trainer_reward_restores_seeded_think_prefix():
    values = clinical_reward(
        ["症状、病程、严重程度、功能损害、排除条件和鉴别诊断。</think><answer>F31.4</answer>"],
        ["F31.4"],
        ["diagnosis"],
        ["subtype"],
    )
    assert values == [4.0]
