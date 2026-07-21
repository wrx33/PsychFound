# Training implementation

PsychFound uses one end-to-end training interface for knowledge injection,
reasoning enhancement and clinical adaptation.

## Supervised fine-tuning

- Qwen ChatML is rendered explicitly with the default system message.
- Prompt tokens receive label `-100`; only assistant response tokens and the
  final EOS token contribute to cross-entropy.
- Prompts and responses are tokenized separately and limited to 4,096 and 2,048
  tokens, respectively.
- PEFT LoRA uses rank 256, alpha 512, dropout 0, no bias, and all linear
  attention/feed-forward projections.
- The optimizer uses a learning rate of `1e-5`, cosine scheduling and a warm-up
  ratio of 0.1.
- Each SFT stage saves its adapter and creates a merged full-model checkpoint on
  rank 0. The pipeline passes the merged checkpoint to the next stage.

## GRPO

The reference configuration samples 8 prompts and 8 responses per prompt,
producing 64 completions per optimizer update on 8 GPUs. TRL counts completions
in `per_device_train_batch_size`, so the configured value is 8 per device:

`8 completions/device × 8 devices ÷ 8 generations = 8 prompts`.

The GRPO stages use:

- group-normalized reward advantages;
- token-level importance ratios and symmetric clipping with epsilon 0.2;
- response-token masks and BNPO batch normalization;
- one policy iteration per rollout batch;
- a positive low-variance KL penalty with coefficient `0.001`;
- full-parameter bf16 training, gradient checkpointing and gradient norm 1.0;
- 512-token prompt and response limits;
- vLLM rollout with tensor parallelism 2 and temperature 0.6;
- eight-process FSDP full sharding and CPU offload.

## Reward function

The total reward is

`format_reward + reasoning_reward + 2 × accuracy_reward`.

Format reward checks the required `<think>...</think><answer>...</answer>`
contract. Reasoning reward measures coverage of task-specific clinical concepts.
Diagnosis accuracy is calculated at ICD-10 category level in the first round and
subtype level in the second. Medication accuracy uses category-target recall in
the first round and exact-drug F1 in the second.

## Reproducibility limits

Distributed collective order, CUDA kernels, vLLM sampling streams and library
patch releases can affect exact floating-point values. Reproduction checks
should compare token IDs and labels, rewards and advantages on fixed
completions, a single optimizer-step parameter delta, and final PsychBench
metrics. Sampled output-text equality alone is not a reliable parity test.
