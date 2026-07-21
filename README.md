# PsychFound: A Domain-Adapted Large Language Model for Psychiatric Clinical Practice

**PsychFound** is a clinician-oriented large language model designed to support
psychiatric clinical workflows, including clinical-text understanding,
diagnostic reasoning, differential diagnosis, medication recommendation and
longitudinal management.

This repository provides an independent, end-to-end implementation of the
development workflow described in:

> Wang, R. et al. A domain-adapted large language model to support clinicians
> in psychiatric clinical practice. *Nature Machine Intelligence* **8**,
> 690-707 (2026). https://doi.org/10.1038/s42256-026-01224-w

---

## Key features

- A single `psychfound` command for data validation, training, inference and evaluation.
- Domain knowledge injection, reasoning cold start, progressive GRPO and
  full-cycle clinical SFT in one coherent training system.
- Deterministic format, clinical-reasoning and accuracy rewards weighted `1:1:2`.
- Explicit separation between public PsychCorpus data and protected clinical data.
- Reproducible stage configurations without machine-specific paths or credentials.
- Unit-tested data contracts, prompts, rewards, pipeline orchestration and metrics.

> [!IMPORTANT]
> PsychFound is a research and clinical decision-support system. It is not an
> autonomous diagnostic or prescribing system. Qualified clinicians must review
> all outputs and retain authority over final clinical decisions.

---

## Development workflow

```text
PsychCorpus
    │
    ▼
Professional knowledge SFT
    │
    ▼
Expert reasoning cold-start SFT
    │
    ├── Diagnosis: ICD-10 category GRPO → subtype GRPO
    │
    └── Medication: drug-category GRPO → exact-drug GRPO
    │
    ▼
Full-cycle clinical multitask SFT
    │
    ▼
PsychBench and clinical evaluation
```

The seven executable stages are declared in `configs/pipeline.yaml`. Each stage
can be inspected, resumed or run independently.

---

## Installation

Python 3.10 or newer and a CUDA-enabled Linux environment are recommended for
training. The reference configuration uses eight NVIDIA A100 GPUs.

```bash
git clone https://github.com/wrx33/PsychFound.git
cd PsychFound

python -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"
```

For inference only:

```bash
pip install -e ".[inference]"
```

Inspect the CLI:

```bash
psychfound --help
psychfound --version
```

---

## Data preparation

### ShareGPT SFT format

Knowledge, cold-start and clinical multitask SFT datasets use JSONL records:

```json
{"conversations": [
  {"from": "human", "value": "Patient information or clinical question"},
  {"from": "gpt", "value": "Reference response"}
]}
```

Validate a dataset before training:

```bash
psychfound validate-data --input data/PsychCorpus/PsychCorpus.jsonl
```

Cold-start targets should contain the structured reasoning contract:

```text
<think>Clinical reasoning process</think><answer>Final decision</answer>
```

### GRPO data

Prepare the first diagnosis curriculum round:

```bash
psychfound prepare-rl \
  --task diagnosis \
  --precision category \
  --input /secure/path/diagnosis.jsonl \
  --output /secure/path/diagnosis-category \
  --train-size 800 \
  --test-size 80 \
  --seed 42
```

Repeat with `--precision subtype` for exact ICD-10 subtypes. Medication uses the
same command with `--task medication`; the category round uses medication-class
targets and the subtype round uses exact generic drug names.

The command writes pre-templated Qwen ChatML prompts with an initial `<think>`
prefix, matching the input contract used by the GRPO rollout stage.

Each output directory contains `train.jsonl`, `test.jsonl` and `metadata.json`.

---

## Configuration

Copy `.env.example` to a secure environment configuration and set paths without
committing it:

```bash
cp .env.example .env
set -a
source .env
set +a
```

The repository provides one reference configuration for PsychFound:

- LoRA rank 256 on all linear projection layers.
- Qwen ChatML with the default system message, assistant-only SFT labels and
  independently enforced prompt and response limits.
- SFT learning rate `1e-5`, cosine scheduling, warm-up ratio `0.1`, five epochs,
  per-device batch size 4, 4,096 prompt tokens and 2,048 response tokens.
- Full-parameter GRPO with eight generations per prompt, global prompt batch 8,
  PPO clipping at `0.2`, group-normalized advantages, response-token masking,
  low-variance KL, learning rate `1e-6` and 512/512 prompt/response limits.
- Eight-process FSDP training, vLLM rollout, tensor parallelism 2, rollout
  temperature `0.6`, five epochs and checkpoint/evaluation interval 45.
- Format, clinical-reasoning keyword and task-accuracy rewards weighted `1:1:2`.
- A positive KL-penalty coefficient of `0.001`.
- Inference temperature `0.1`, top-p `0.75` and up to 4,096 generated tokens.

TRL `loss_type: bnpo` is used because its response-token masked batch mean is
consistent with the token-masked policy loss used by the GRPO actor. See
`docs/TRAINING_IMPLEMENTATION.md` for the objective and batch-size mapping.

Each SFT stage saves both its adapter and an automatically merged checkpoint.
The pipeline exports the merged/full checkpoint path to the next stage, so no
intermediate model environment variables are required during an end-to-end run.

---

## Training

Preview every stage without loading a model or allocating a GPU:

```bash
psychfound pipeline --dry-run
```

Run the complete development pipeline:

```bash
accelerate launch \
  --config_file configs/accelerate/grpo_fsdp.yaml \
  -m psychfound pipeline
```

This launches every stage on eight processes. For inexpensive orchestration checks, use the
single-process `--dry-run` command above.

Run a bounded stage range:

```bash
psychfound pipeline \
  --start-at diagnosis_category_grpo \
  --stop-after diagnosis_subtype_grpo
```

Run one SFT configuration directly:

```bash
psychfound train-sft --config configs/01_knowledge_sft.yaml
```

SFT configurations merge adapters automatically. To merge an existing adapter
manually:

```bash
psychfound merge-adapter \
  --base-model /path/to/base \
  --adapter outputs/01_knowledge_sft \
  --output outputs/01_knowledge_sft_merged
```

Run GRPO with the eight-process FSDP configuration:

```bash
accelerate launch \
  --config_file configs/accelerate/grpo_fsdp.yaml \
  -m psychfound train-grpo \
  --config configs/03a_diagnosis_category_grpo.yaml
```

The default colocated vLLM configuration uses the same GPUs as training. If it
does not fit on a particular CUDA/vLLM combination, switch to TRL's server mode;
that changes systems behavior and should be recorded in the run manifest.

Completed pipeline runs write a timestamped manifest under `outputs/runs/`.

---

## Inference

Set `PSYCHFOUND_MODEL`, then run:

```bash
psychfound infer \
  --config configs/inference.yaml \
  --prompt "Please provide a structured diagnostic analysis for the following case: ..."
```

The model checkpoint can also be loaded directly with Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/PsychFound"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)

messages = [{"role": "user", "content": "Your clinical prompt"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=4096, temperature=0.1, top_p=0.75)
print(tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

---

## Evaluation

The lightweight evaluator accepts JSONL predictions:

```json
{"task": "diagnosis", "precision": "subtype", "prediction": "<think>...</think><answer>F31.4 ...</answer>", "reference": "F31.4 ..."}
```

```bash
psychfound evaluate --predictions outputs/predictions.jsonl
```

It reports structured-output compliance, clinical-reasoning coverage, task
accuracy, aggregate reward and task-stratified accuracy. Full PsychBench
evaluation data are released separately from protected development data.

---

## Project structure

```text
PsychFound/
├── configs/                  # Stage, inference and distributed-training settings
├── data/PsychCorpus/         # Public domain-knowledge data location
├── docs/                     # Architecture and data-governance notes
├── scripts/                  # Convenience validation script
├── src/psychfound/
│   ├── data/                 # Data contracts and RL preparation
│   ├── training/             # SFT, adapter merge, GRPO and pipeline orchestration
│   ├── rewards.py            # Psychiatric task rewards
│   ├── inference.py          # Model generation
│   ├── evaluation.py         # Reproducible metrics
│   └── cli.py                # Unified public interface
└── tests/                    # Unit tests
```

---

## Privacy and responsible use

- Never commit identifiable or re-identifiable patient records.
- Keep PsychClinical and prospective-study data in access-controlled storage.
- Record dataset hashes and verify that training cases do not overlap PsychBench.
- Do not write raw clinical prompts or model responses to public experiment logs.
- Report model uncertainty and failed runs, not only the best checkpoint.
- Use outputs as recommendations requiring clinician review.

See `docs/DATA_GOVERNANCE.md` for the release checklist.

---

## License

This repository is released under the MIT License. Clinical datasets and model
checkpoints may be subject to separate access and usage restrictions.

---

## Citation

```bibtex
@article{wang2026psychfound,
  title   = {A domain-adapted large language model to support clinicians in psychiatric clinical practice},
  author  = {Wang, Ruoxi and others},
  journal = {Nature Machine Intelligence},
  volume  = {8},
  pages   = {690--707},
  year    = {2026},
  doi     = {10.1038/s42256-026-01224-w}
}
```

---

## Acknowledgements

The original PsychFound research and engineering process benefited from the
training abstractions and practical fine-tuning experience provided by
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and from the compact
reinforcement-learning design and GRPO experimentation patterns demonstrated by
[TinyZero](https://github.com/Jiayi-Pan/TinyZero). We gratefully acknowledge
their open-source contributions.
