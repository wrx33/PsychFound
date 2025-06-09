---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:36421
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 丙戊酸钠缓释片的成份
  sentences:
  - 马来酸氟伏沙明片的禁忌：1、对马来酸氟伏沙明或其他辅料过敏者禁用；2、本品禁与单氨氧化酶抑制剂（MAOIs）联合应用，如果病人由服用单氨氧化酶抑制剂改服本品，治疗初期应注意：如为不可逆转的单氨氧化酶抑制剂，至少应停药2周；如为可逆转的单氨氧化酶抑制剂（如吗氯贝胺）可于停药后1天改服本品；3、若停用本品治疗，在改用单氨氧化酶抑制剂之前至少应停药1周。......
  - 盐酸文拉法辛胶囊的成份：本品主要成分为：盐酸文拉法辛化学名称：(±)-1-[2-(二甲胺基)-1-(4-甲氧苯基)乙基]-环己醇盐酸盐
  - 奥氮平片的批准文号：H20150142
- source_sentence: 盐酸氟西汀胶囊的儿童用药
  sentences:
  - 利培酮片的注意事项：1．患有心血管疾病的人(如心衰、心肌梗死、传导异常、脱水、失血或脑血管疾病) 应慎用，从小剂量开始并应逐渐加大剂量(见【用法用量】)2．由于本品具有α
    受体阻断活性， 因此在用药初期和加药速度过快时会发生（体位性）低血压，此时则应考虑减量。3．同其他具有多巴胺受体拮抗剂性质的药物相似，引起迟发性运动障碍，其特征为有节律的不随意运动，
    主要见于舌及面部。如果出现迟发性运动障碍，应停止服用所有的抗精神病药。4．已有报道指出，服用经典的抗精神病药会出现恶性综合征，其特征为高热、颤抖、意识改变和肌酸磷酸酶水平升高。此时应停用包括本品在内的所有抗精神病药物。5．患有帕金森氏综合征的病人应慎用本品，因为在理论上该药会引起此病的恶化。6．经典的抗精神病药会降低癫痫的发作阈值，
    故患有癫痫的病人应慎用本品。7．服用本品的患者应避免进食过多，以免发胖。8．鉴于本品对中枢神经系统的作用，在与其它作用于中枢的药物同时服用时应慎用。9．本品对需要警觉性的活动有所影响。因此，在了解到患者对该药的敏感性前，建议患者不应驾驶汽车或操作机器。......
  - 巴戟天寡糖胶囊的性状：本品为硬胶囊，内容物为类白色至浅黄色颗粒；味甜。
  - 复方苁蓉益智股囊的禁忌：尚不明确。
- source_sentence: 米氮平片的生产企业
  sentences:
  - 复方苁蓉益智股囊的规格：征粒袋0.3g
  - 血府逐瘀胶囊的不良反应：尚不明确
  - 米氮平片的成份：主要组成成分米氮平。
- source_sentence: 五氟利多片的用法用量
  sentences:
  - 盐酸氟西汀胶囊的用法用量：【老年患者用药】 增加剂量应当慎重，日剂量一般不宜超过40mg。最高推荐剂量是60mg。见。
  - 奥氮平片的儿童用药：尚无在 18 岁以下人群中的研究情况。
  - 基础信息
- source_sentence: 米氮平片的执行标准
  sentences:
  - 多巴丝肼的规格：片剂（胶囊剂）：每片（粒）125mg（左旋多巴100mg和苄丝肼25mg）250mg（左旋多巴200mg和苄丝肼50mg）。控释片：每片125mg。分散片：每片125mg。
  - 棕榈酸帕利哌酮注射液（善思达）的规格：（1） 0.25ml：25mg，（2）0.5ml：50mg ，（3）0.75ml：75mg，（4）1.0ml：100mg，（5）1.5ml：150mg（按帕利哌酮计）
  - 棕榈酸帕利哌酮注射液（善思达）的执行标准：JX20100263
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '米氮平片的执行标准',
    '棕榈酸帕利哌酮注射液（善思达）的执行标准：JX20100263',
    '棕榈酸帕利哌酮注射液（善思达）的规格：（1） 0.25ml：25mg，（2）0.5ml：50mg ，（3）0.75ml：75mg，（4）1.0ml：100mg，（5）1.5ml：150mg（按帕利哌酮计）',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 36,421 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | float                                                          |
  | details | <ul><li>min: 8 tokens</li><li>mean: 14.77 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 100.69 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.16</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                   | sentence_1                                                                                                                       | label            |
  |:-----------------------------|:---------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>氢溴酸西酞普兰片的适应证</code>    | <code>舒眠胶囊的执行标准：国家药品标准WS3-733(Z-137)-2001Z</code>                                                                                | <code>0.0</code> |
  | <code>注射用利培酮微球（恒德）的成份</code> | <code>磷酸钠盐口服溶液的成份：本品为复方制剂，其组分为磷酸二氢钠和磷酸氢二钠。</code>                                                                                | <code>0.0</code> |
  | <code>富马酸喹硫平缓释片的成份</code>    | <code>富马酸喹硫平缓释片的成份：本品主要成份：为11-[4-[2-(2-羟乙氧基)乙基-1-哌嗪基]]二苯骈[b,f][1,4]硫氮杂䓬富马酸盐（2:1）分子式:(C21H25N3O2S)2·C4H4O4分子量:883.08......</code> | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 20
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 20
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch   | Step  | Training Loss |
|:-------:|:-----:|:-------------:|
| 0.2196  | 500   | 0.0949        |
| 0.4392  | 1000  | 0.07          |
| 0.6588  | 1500  | 0.0568        |
| 0.8783  | 2000  | 0.0483        |
| 1.0979  | 2500  | 0.0441        |
| 1.3175  | 3000  | 0.04          |
| 1.5371  | 3500  | 0.0401        |
| 1.7567  | 4000  | 0.0377        |
| 1.9763  | 4500  | 0.0344        |
| 2.1959  | 5000  | 0.0327        |
| 2.4155  | 5500  | 0.0327        |
| 2.6350  | 6000  | 0.0312        |
| 2.8546  | 6500  | 0.0314        |
| 3.0742  | 7000  | 0.0302        |
| 3.2938  | 7500  | 0.0274        |
| 3.5134  | 8000  | 0.028         |
| 3.7330  | 8500  | 0.0285        |
| 3.9526  | 9000  | 0.0276        |
| 4.1722  | 9500  | 0.0262        |
| 4.3917  | 10000 | 0.0248        |
| 4.6113  | 10500 | 0.0258        |
| 4.8309  | 11000 | 0.0267        |
| 5.0505  | 11500 | 0.0233        |
| 5.2701  | 12000 | 0.0217        |
| 5.4897  | 12500 | 0.0233        |
| 5.7093  | 13000 | 0.0238        |
| 5.9289  | 13500 | 0.0251        |
| 6.1484  | 14000 | 0.0211        |
| 6.3680  | 14500 | 0.022         |
| 6.5876  | 15000 | 0.0224        |
| 6.8072  | 15500 | 0.0224        |
| 7.0268  | 16000 | 0.0222        |
| 7.2464  | 16500 | 0.0203        |
| 7.4660  | 17000 | 0.0211        |
| 7.6856  | 17500 | 0.0209        |
| 7.9051  | 18000 | 0.021         |
| 8.1247  | 18500 | 0.0193        |
| 8.3443  | 19000 | 0.0207        |
| 8.5639  | 19500 | 0.0188        |
| 8.7835  | 20000 | 0.0203        |
| 9.0031  | 20500 | 0.02          |
| 9.2227  | 21000 | 0.0191        |
| 9.4422  | 21500 | 0.0184        |
| 9.6618  | 22000 | 0.0198        |
| 9.8814  | 22500 | 0.019         |
| 10.1010 | 23000 | 0.0191        |
| 10.3206 | 23500 | 0.0198        |
| 10.5402 | 24000 | 0.0181        |
| 10.7598 | 24500 | 0.0183        |
| 10.9794 | 25000 | 0.0181        |
| 11.1989 | 25500 | 0.0184        |
| 11.4185 | 26000 | 0.0175        |
| 11.6381 | 26500 | 0.0194        |
| 11.8577 | 27000 | 0.0178        |
| 12.0773 | 27500 | 0.0176        |
| 12.2969 | 28000 | 0.0176        |
| 12.5165 | 28500 | 0.018         |
| 12.7361 | 29000 | 0.0175        |
| 12.9556 | 29500 | 0.018         |
| 13.1752 | 30000 | 0.0168        |
| 13.3948 | 30500 | 0.018         |
| 13.6144 | 31000 | 0.0184        |
| 13.8340 | 31500 | 0.0157        |
| 14.0536 | 32000 | 0.0178        |
| 14.2732 | 32500 | 0.0159        |
| 14.4928 | 33000 | 0.0172        |
| 14.7123 | 33500 | 0.0175        |
| 14.9319 | 34000 | 0.0171        |
| 15.1515 | 34500 | 0.0166        |
| 15.3711 | 35000 | 0.0168        |
| 15.5907 | 35500 | 0.0174        |
| 15.8103 | 36000 | 0.0167        |
| 16.0299 | 36500 | 0.0163        |
| 16.2495 | 37000 | 0.017         |
| 16.4690 | 37500 | 0.0158        |
| 16.6886 | 38000 | 0.0159        |
| 16.9082 | 38500 | 0.0174        |
| 17.1278 | 39000 | 0.0157        |
| 17.3474 | 39500 | 0.0161        |
| 17.5670 | 40000 | 0.015         |
| 17.7866 | 40500 | 0.0167        |
| 18.0061 | 41000 | 0.0176        |
| 18.2257 | 41500 | 0.0141        |
| 18.4453 | 42000 | 0.0164        |
| 18.6649 | 42500 | 0.017         |
| 18.8845 | 43000 | 0.016         |
| 19.1041 | 43500 | 0.0168        |
| 19.3237 | 44000 | 0.0179        |
| 19.5433 | 44500 | 0.0158        |
| 19.7628 | 45000 | 0.0147        |
| 19.9824 | 45500 | 0.0154        |


### Framework Versions
- Python: 3.10.16
- Sentence Transformers: 3.3.1
- Transformers: 4.47.1
- PyTorch: 2.5.1+cu124
- Accelerate: 1.2.1
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->