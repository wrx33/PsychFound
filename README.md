# PsychFound: A Domain-Adapted and Clinician-Oriented Language Model for Real-World Psychiatric Clinical Practice

**PsychFound** is a clinician-oriented large language model (LLM) designed to support full-spectrum psychiatric clinical tasks. Built upon expert-curated corpora and real-world clinical data, it provides evidence-based, structured decision support for diagnosis, treatment, and prognosis management in mental health care.

---

## 🔍 Key Features

- **End-to-end support** for psychiatric clinical workflows, including:
  - Diagnostic reasoning and differential diagnosis
  - Medication planning and contraindication analysis
  - Prognosis monitoring and follow-up suggestions
- **Expert-aligned**: Fine-tuned on 42,266 anonymized EHRs and evaluated via a real-world prospective study and multi-level reader study.
- **Open Resources**: Includes an open psychiatric corpus and benchmarking dataset (PsychCorpus, PsychBench).

---

## 📁 Repository Structure

```bash
.
├── data/                # Open-source datasets
├── scripts/             # Preprocessing, training, evaluation scripts
├── model/               # Checkpoints and config files
├── examples/            # Clinical usage examples
├── docs/                # Figures, paper summary
└── README.md
```

## 🚀Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/PsychFound.git
cd PsychFound
```

### 2. Install dependencies

```bash
pip install -e ".[torch,metrics]" --no-build-isolation
```

### 3. Data preparation

This project adopts a **three-stage development framework**, and corresponding datasets are used for each stage. Below we describe the data preparation process for each phase:

#### 1) Professional Knowledge Injection (Stage 1)

We release the dataset used in the first phase, named PsychCorpus, located at:

```bash
data/PsychCorpus
```

**PsychCorpus** is a domain-specific corpus constructed from publicly available and expert-curated resources, including clinical guidelines, standard textbooks, and high-quality academic publications in psychiatry. This corpus serves as the foundational knowledge source to infuse general domain expertise into the model.

#### 2) Real-World Clinical Adaptation (Stage 2 & 3)

The dataset used in the second and third phases is **PsychClinical**, constructed from real-world de-identified electronic health records (EHRs). Due to privacy and regulatory constraints, **PsychClinical is not publicly available**.

For those who wish to replicate the pipeline or perform training with private EHR data, we suggest organizing your data in the following format for SFT (Supervised Fine-tuning) and RL (Reinforcement Learning):

**For SFT:**

```
{
	"conversations":[
		{"from": "human",
		 "value": "your content"}
		{"from": "gpt",
		 "value": "your content"}
	]
}
```

**For RL:**

Run `python ./scripts/data_preprocess/rl_diagnosis.py --local_dir {path_to_your_dataset}` to prepare RL dataset for diagnosis task using your dataset.

You can prepare your cold-start data as following:

```
{
	"conversations":[
		{"from": "human",
		 "value": "your content"}
		{"from": "gpt",
		 "value": "<think>your content</think> <answer>your content</answer>"}
	]
}
```

### 4. Training 

**For SFT:**

```bash
llamafactory-cli train ./scripts/train_sft/sft_lora.yaml # LoRA
llamafactory-cli train ./scripts/train_sft/sft_full.yaml # Full parameters
```

**For RL:**

```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=diagnosis-psychfound-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_rl/rl_diagnosis.sh
```

### 5. Inference

You can use our checkpoints in ./model/checkpoints/psychfound_v1

You can also change model_name_or_path to the path to your own checkpoints.

```bash
llamafactory-cli chat ./scripts/inference/inference.yaml
```

## 📄License

This repository is released under the MIT License.

## 📫 Citation

If you use PsychFound in your research, please cite:

```
Coming soon!
```

### 🤝 Acknowledgments

This project benifits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [TinyZero](https://github.com/Jiayi-Pan/TinyZero).