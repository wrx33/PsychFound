# PsychFound: A Domain-Adapted and Clinician-Oriented Language Model for Real-World Psychiatric Clinical Practice

**PsychFound** is a clinician-oriented large language model (LLM) designed to support full-spectrum psychiatric clinical tasks. Built upon expert-curated corpora and real-world clinical data, it provides evidence-based, structured decision support for diagnosis, treatment, and prognosis management in mental health care.

---

## 🔍 Key Features

- **End-to-end support** for psychiatric clinical workflows, including:
  - Diagnostic reasoning and differential diagnosis
  - Medication planning and contraindication analysis
  - Prognosis monitoring and follow-up suggestions
- **Expert-aligned**: Fine-tuned on 64,588 anonymized EHRs and evaluated via a real-world prospective study and multi-level reader study.
- **Open Resources**: Includes an open psychiatric corpus and benchmarking dataset (PsychCorpus, PsychBench).

---

## Quick Start

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

You can use our checkpoints from huggingface:  wangrx33/PsychFound_v1. 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "path to PsychFound",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("path to PsychFound")

prompt = """
Please complete the diagnostic analysis report, which should include the following:
- Diagnostic conclusion.
- Clinical and examination evidence supporting the diagnosis
- Differential diagnosis and exclusion analysis.

### Patient Information:
Female, 26 years old. **Current Medical History:** In January 2012, while in Grade 9, the patient experienced depression, fatigue, drowsiness, and a general feeling of weakness due to academic pressure. She found it difficult to think and complete homework, and could not understand the teacher's lectures. Her nighttime sleep was poor, often falling asleep around 2 or 3 AM, with shallow sleep and frequent awakenings (about 4 times) per night, occurring approximately once a week. She interacted well with others and managed to perform well in the high school entrance exam. In November 2012, after entering a prestigious high school in Inner Mongolia, she felt that others were more capable, and her symptoms reappeared, although her mood improved slightly compared to before. She remained in this state for a long time, with no significant impact on her studies or life. In November 2013, she experienced an unprovoked episode of increased mood, energy, and cognitive speed, resulting in a significant improvement in academic performance. Two to three weeks later, her symptoms of depression, fatigue, and drowsiness worsened. In November 2015, after entering university, she felt a lax lifestyle and developed a strong sense of self-reproach, feeling sad about the future, with a significant worsening of depressive symptoms, including thoughts of life having no meaning and suicidal ideation. She experienced fatigue and drowsiness, lost interest in playing computer games, and found it difficult to complete homework assigned by teachers. Her nighttime sleep was poor, with difficulty falling asleep and shallow sleep almost every night. She interacted well with others. In November 2017, after watching a program about depression, she sought treatment at the First Affiliated Hospital of Fujian Medical University, where she was diagnosed with \"bipolar disorder.\" She did not receive medication treatment and instead opted for further evaluation before taking medication. She was first treated at our hospital in January 2018, diagnosed with recurrent depressive disorder, and was discharged after 2 months of treatment with escitalopram (20 mg/day) and quetiapine fumarate (12.5 mg/day), experiencing a severe episode without psychotic symptoms. Post-discharge, her symptoms were unstable, and she was readmitted for the same diagnosis and treatment, and was discharged in May 2017 with significant improvement. After discharge, she adhered to medication and had occasional low moods but could maintain normal daily life and studies. In May 2022, she witnessed her father's sudden death at home, and her mood deteriorated significantly, often crying silently, with a decline in academic performance and inability to continue schooling. She had suicidal thoughts without a plan and was constantly worried that her mother would suddenly fall ill and leave her. She could maintain normal daily life and eating habits, with sleep still acceptable. In the past two weeks, she has not experienced fever, abdominal pain, diarrhea, nausea, or vomiting, nor has she experienced coma or convulsions. Her appetite is normal, sleep is acceptable, and bowel and bladder functions are normal, with no significant changes in weight.\n\n**Past Medical History:**\nNegative.\n\n**Family History:**\nNegative.\n\n**Physical Examination, Auxiliary Examinations, and Mental Status Examination:**\nPhysical examination revealed no obvious abnormalities. Auxiliary examinations pending. Physical examination revealed no obvious abnormalities.
"""

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'''
(<think>\nI need to review the patient's medical records and complete a diagnostic analysis report. First, I will carefully read the provided medical information and then proceed with a step-by-step analysis to ensure no critical points are overlooked.\n\nThe patient is a 26-year-old female whose medical history began in early 2012 during junior high school, when she experienced symptoms such as depression, fatigue, difficulty thinking, and sleep issues due to academic pressure. Despite these symptoms, she achieved excellent results in the high school entrance exam. Similar symptoms recurred during high school, but her mood improved. In November 2013, she experienced a two-to-three-week period of elevated mood, increased energy, and faster thinking, followed by a worsening of depressive symptoms. During her university years, she experienced intense feelings of self-reproach, worsening depression, and suicidal thoughts, leading to a diagnosis of bipolar disorder without medication. In 2018, she was diagnosed with recurrent depressive disorder and received pharmacological treatment. After her father's death in 2022, her depression worsened, and she was readmitted to the hospital.\n\nFirst, the diagnostic conclusion needs to consider whether the patient has bipolar disorder or recurrent depressive disorder. Previous diagnoses at other hospitals indicated bipolar disorder, while this hospital diagnosed recurrent depressive disorder. The key point is whether there is a clear episode of mania or hypomania. According to the patient's history, in November 2013, she experienced a two-to-three-week period of elevated mood, increased energy, and increased confidence, which aligns with the criteria for a hypomanic episode (DSM-5 requires hypomanic episodes to last at least four days, though this can vary by diagnostic criteria). However, symptoms lasting two to three weeks exceed the typical duration of a hypomanic episode (usually at least four days, but no more than one week). Based on the description, this episode may still qualify as hypomania, supporting a diagnosis of bipolar disorder. Confirmation of the duration is needed.\n\nSecond, differential diagnoses need to consider recurrent depressive disorder, adjustment disorder, anxiety disorders, etc. Other factors that could cause mood swings, such as thyroid issues or other somatic diseases, must be ruled out. The patient’s past medical history is negative, and physical examination results are normal, with pending auxiliary tests. These factors may need to be excluded.\n\nAdditionally, the patient has clear precipitating factors (academic pressure, father's death), but her depressive episodes are prolonged and cyclical, with mood swings, especially a possible hypomanic episode in 2013, followed by worsening depression. This more closely aligns with bipolar disorder rather than simple depression. However, previous diagnoses are conflicting, requiring further analysis.\n\nSupporting a diagnosis of bipolar disorder includes a clear episode of elevated mood lasting two to three weeks, accompanied by functional improvement (improved academic performance), followed by worsening depression. This fits the criteria for bipolar II disorder, which includes hypomanic episodes and depressive episodes. Recurrent depressive disorder does not include episodes of mania or hypomania. Therefore, previous diagnoses may have been incorrect or misinterpreted.\n\nIn differential diagnosis, recurrent depressive disorder must be ruled out. If there is a hypomanic episode, it cannot be diagnosed as simple depression. Confirmation is needed that the episode meets hypomanic criteria: duration and sufficient symptoms (such as reduced sleep, increased activity). The patient experienced elevated mood, increased energy, faster thinking, and improved academic performance lasting two to three weeks, which may qualify as hypomania, supporting a diagnosis of bipolar disorder.\n\nFurthermore, the patient's response to treatment may influence the diagnosis. For example, antidepressants may induce mania, but this is not mentioned in the patient's history. The patient is currently on escitalopram and quetiapine, possibly for depression and mood stabilization.\n\nSummary: The diagnosis may be bipolar disorder, currently in a depressive episode. Support for this includes the hypomanic episode in 2013, followed by worsening depression. Recurrent depressive disorder is ruled out due to the presence of a hypomanic phase. Other conditions such as adjustment disorder or anxiety disorder need to be excluded, but the patient's symptoms are more consistent with bipolar disorder.\n\n</think>
    
### Diagnostics Analysis Report
#### **Diagnostic Conclusion**
The patient is diagnosed with **Bipolar Disorder, currently in a severe depressive episode**.
#### **Clinical and Examination Evidence Supporting the Diagnosis**
1. **Core Symptoms of Depression**: The patient exhibits persistent depressive symptoms such as mood low, loss of interest, suicidal ideation, and fatigue. These symptoms have been present since childhood, with episodic exacerbation during stressful events (e.g., academic pressure, family crisis).
2. **Evidence of Manic Episodes**: Historical records show that the patient experienced a period of elevated mood, energy, and cognitive speed (November 2013). However, the subsequent worsening of depressive symptoms suggests the presence of bipolar II disorder rather than a manic episode. This supports the differential diagnosis in favor of bipolar disorder.
3. **Family and Personal History**: Negative family history and stable personal history support the diagnosis of bipolar disorder.
4. **Response to Treatment**: The patient's response to SSRI (escitalopram) and antipsychotics (quetiapine) indicates the effectiveness of treatment in bipolar disorder, further supporting the diagnosis.
5. **Exclusion of Other Disorders**: No evidence of organic brain disorders, substance use, or other mental disorders makes bipolar disorder the primary diagnosis.
#### **Differential Diagnosis and Exclusion Analysis**
1. **Depressive Disorder (Unipolar Depression)**: The patient's history includes episodes of depression with no history of manic or hypomanic episodes. However, the presence of manic episodes in the past necessitates a differential diagnosis.
2. **Anxiety Disorders**: While the patient may experience anxiety, the core symptoms of bipolar disorder (mood swings, mood instability) are more consistent with bipolar disorder.
3. **Substance-Induced Disorders**: The patient does not have a history of substance use, making this diagnosis less likely.
4. **Organic Brain Disorders**: Negative physical examination and imaging results rule out organic causes.
5. **Schizophrenia**: The patient's history does not support the presence of psychotic symptoms, making schizophrenia less likely.
#### **Conclusion**
Based on the patient's clinical presentation, historical records, and treatment response, the diagnosis of Bipolar Disorder, currently in a severe depressive episode, is supported. Further examination of auxiliary tests will help rule out other potential causes of symptoms.
)
''' 
```

(Reasoning ability may be slightly reduced in English interaction compared with Chinese interaction.)

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

Recommended CUDA version: 12.0

VRAM requirements (7B):
Training with AdamW and LoRA (r=8): ~14G 

Reasoning: ~10G

### 3. Data preparation

This project adopts a **three-stage development framework**, and corresponding datasets are used for each stage. Below we describe the data preparation process for each phase:

#### 1) Professional Knowledge Injection (Stage 1)

We release part of the dataset used in the first phase, named PsychCorpus, located at:

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

The `data/dataset_info.json` contains all available datasets. If you are using a custom dataset, please **make sure** to add a *dataset description* in `dataset_info.json` and specify `dataset: dataset_name` before training to use it.

**For RL:**

Refer to `python ./tinyzero/examples/data_preprocess/psychfound_diagnosis.py --local_dir {path_to_your_dataset}` to prepare your RL dataset.

You can organize your cold-start data as following:

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
python -m src.llamafactory.cli train ./examples/train_lora/sft_lora.yaml # LoRA
python -m src.llamafactory.cli train ./examples/train_full/sft_full.yaml # Full parameters
```

**For RL:**

```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=diagnosis-psychfound-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./tinyzero/scripts/train_rl_diagnosis.sh
```

### 5. Inference

You can also change model_name_or_path to the path to your own checkpoints.

```bash
python -m src.llamafactory.cli chat ./scripts/inference/inference.yaml
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