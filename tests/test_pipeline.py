from pathlib import Path

from psychfound.training import run_pipeline


def test_complete_pipeline_dry_run():
    root = Path(__file__).parents[1]
    results = run_pipeline(root / "configs/pipeline.yaml", dry_run=True)
    assert [row["name"] for row in results] == [
        "knowledge_injection", "reasoning_cold_start", "diagnosis_category_grpo",
        "diagnosis_subtype_grpo", "medication_category_grpo",
        "medication_exact_grpo", "clinical_adaptation",
    ]
    assert results[1]["model"].endswith("outputs/01_knowledge_sft_merged")
    assert results[-1]["model"].endswith("outputs/03d_medication_exact_grpo")


def test_bounded_pipeline_dry_run():
    root = Path(__file__).parents[1]
    results = run_pipeline(root / "configs/pipeline.yaml", dry_run=True,
                           start_at="diagnosis_category_grpo", stop_after="diagnosis_subtype_grpo")
    assert len(results) == 2
