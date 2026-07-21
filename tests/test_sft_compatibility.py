from psychfound.data.io import ClinicalExample
from psychfound.training.sft import IGNORE_INDEX, encode_supervised_example, qwen_prompt_and_completion


class CharacterTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 999

    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text)))


def test_qwen_template_and_assistant_only_labels():
    example = ClinicalExample("病例", "答复")
    prompt, completion = qwen_prompt_and_completion(example, eos_token="<|im_end|>")
    assert prompt.startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>")
    assert prompt.endswith("<|im_start|>assistant\n")
    assert completion == "答复<|im_end|>"
    encoded = encode_supervised_example(
        example, CharacterTokenizer(), max_prompt_length=4096, max_completion_length=2048
    )
    assert encoded["labels"][: len(prompt)] == [IGNORE_INDEX] * len(prompt)
    assert encoded["labels"][len(prompt) :] == list(range(len(example.target))) + [999]


def test_prompt_and_completion_have_independent_limits():
    example = ClinicalExample("病例" * 100, "答复" * 100)
    encoded = encode_supervised_example(
        example, CharacterTokenizer(), max_prompt_length=40, max_completion_length=20
    )
    assert len(encoded["input_ids"]) == 60
    assert encoded["labels"][:40] == [IGNORE_INDEX] * 40
