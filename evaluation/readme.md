## Evaluate PsychFound on PsychBench

### Requirements
#### For PsychFound
```
torch>=2.2.2
transformers>=4.43.2
tokenizers>=0.21.0
```
#### For Medical Named Entity Recognition (MNER) model
```
Keras==2.2.4
matplotlib==3.4.0
pandas==1.2.3
tensorflow==1.14.0
tqdm==4.61.2
```

### Evaluation Procedure
1. **Prepare the Evaluation Data**
   Download the full evaluation dataset from the [PsychBench](https://github.com/wangrx33/PsychBench) repository, or organize your own test data following the same format as provided in the `./data` directory. Each test instance should be structured as follows:

```json
[
    {
        "conversations": [
            {
                "from": "human",
                "value": "Evaluation question"
            },
            {
                "from": "gpt",
                "value": "Reference answer"
            }
        ],
        "fewshot_examples": "Optional few-shot examples if applicable",
        "id": "question-id"
    },
    ...
]
```

2. **Run the Evaluation Script**
   Execute the evaluation script by running:

   ```bash
   python run_eval.py
   ```

3. **Compute Evaluation Metrics**
   Use the `calc_metrics.ipynb` Jupyter notebook to compute detailed performance metrics. The notebook provides step-by-step instructions for calculating standard evaluation scores.
