# GPT-OSS Examples

This directory contains example scripts for GPT-OSS 20B language models.

For model introduction and architecture details, see the GPT-OSS documentation.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

See the [conversion.sh](conversion.sh) script for checkpoint conversion examples.

- **Import**: Use `openai/gpt-oss-20b` as the source Hugging Face model.
- **Export**: Use `unsloth/gpt-oss-20b-BF16` as the reference HF model for export because the exported Megatron checkpoint is unquantized (bf16), which matches that repo's format.

### Import HF → Megatron

To import the HF model to your desired Megatron path:

```bash
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model openai/gpt-oss-20b \
    --megatron-path ${WORKSPACE}/models/gpt-oss-20b \
    --trust-remote-code
```

### Export Megatron → HF

The export uses `unsloth/gpt-oss-20b-BF16` as the reference so the saved HF checkpoint matches that unquantized format:

```bash
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model unsloth/gpt-oss-20b-BF16 \
    --megatron-path ${WORKSPACE}/models/gpt-oss-20b/iter_0000000 \
    --hf-path ${WORKSPACE}/models/gpt-oss-20b-hf-export
```

### Round-trip Validation

Multi-GPU round-trip validation between formats:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id unsloth/gpt-oss-20b-BF16 \
    --megatron-load-path ${WORKSPACE}/models/gpt-oss-20b/iter_0000000 \
    --tp 2 --pp 2 \
    --trust-remote-code
```

## Training

All GPT-OSS training modes use the unified [training entry point](../../../scripts/training/README.md):
`scripts/training/train.sh`. The examples below select the GPT-OSS 20B library recipe with
`--model gpt_oss_20b`. Alternatively, pass a complete `--recipe`; `--model` and `--recipe` are mutually exclusive.

Set the deployment environment once:

```bash
export CONTAINER_IMAGE=/path/to/container.sqsh
export HF_TOKEN=your_huggingface_token
export HF_HOME=/shared/cache/huggingface
export NEMO_HOME=/shared/cache/nemo
```

Add `--account`, `--partition`, `--nodes`, and `--gpus-per-node` for the target Slurm allocation. Environment variables
and paths are not forwarded automatically: use `--env NAME` for each exported value the job needs and `--mount HOST`
for a same-path mount, or `--mount HOST:CONTAINER` when the paths differ.

### Pretrain on DCLM

DCLM must already be preprocessed into Megatron indexed `.bin/.idx` files. Missing data is an error and never falls
back to mock data.

```bash
./scripts/training/train.sh \
    --nodes 2 --gpus-per-node 8 \
    --account ACCOUNT --partition PARTITION \
    --env HF_TOKEN --env HF_HOME --mount "${HF_HOME}" \
    --mount /data/dclm --mount /data/dclm-cache \
    --model gpt_oss_20b \
    --mode pretrain --dataset dclm \
    --dataset-path /data/dclm \
    --dataset-cache /data/dclm-cache \
    train.train_iters=1000 \
    train.global_batch_size=128
```

Preprocess raw DCLM using the [DCLM tutorial](../../../tutorials/data/dclm/README.md).

Use `--dataset mock` for a model/launcher smoke test.

### SFT on OpenMathInstruct-2

The thinking preset puts chain-of-thought tokens in the analysis channel, the final answer in the final channel, and
enables offline packing. Packing is performed by the dataset builder on global rank 0; there is no separate packing
job.

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 8 \
    --account ACCOUNT --partition PARTITION \
    --env HF_TOKEN --env HF_HOME --env NEMO_HOME \
    --mount "${HF_HOME}" --mount "${NEMO_HOME}" --mount "${WORKSPACE}" \
    --model gpt_oss_20b \
    --mode sft --dataset openmathinstruct2-thinking \
    --from ${WORKSPACE}/models/gpt-oss-20b \
    --cp 2 \
    train.train_iters=1000 \
    train.global_batch_size=128 \
    optimizer.lr=5e-6 \
    optimizer.min_lr=5e-7
```

With `--cp 2`, the launcher configures packed-sequence padding to a multiple of four for THD context parallelism.

### LoRA on OpenMathInstruct-2

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 1 \
    --account ACCOUNT --partition PARTITION \
    --env HF_TOKEN --env HF_HOME --env NEMO_HOME \
    --mount "${HF_HOME}" --mount "${NEMO_HOME}" --mount "${WORKSPACE}" \
    --model gpt_oss_20b \
    --mode lora --dataset openmathinstruct2-thinking \
    --from ${WORKSPACE}/models/gpt-oss-20b \
    train.train_iters=1000 \
    optimizer.lr=1e-4 \
    peft.dim=16
```

Use a complete `--recipe` when you need a hardware- or topology-specific library recipe instead of the model's default
recipe. Do not pass `--model` with `--recipe`.

Multi-node SFT and LoRA require a dataset cache on shared storage so every rank sees the materialized JSONL and packed
artifacts. Forward the cache environment variable with `--env NAME` and mount its directory explicitly.

### Expected Training Dynamics
We provide a [Weights & Biases report](https://api.wandb.ai/links/nvidia-nemo-fw-public/xs3rmk4t) for the expected loss curves and grad norms.

## Inference

See [inference.sh](inference.sh) for text generation with:
- Hugging Face checkpoint (`unsloth/gpt-oss-20b-BF16`)
- Imported Megatron checkpoint (after [conversion.sh](conversion.sh) import)
- Exported HF checkpoint (after conversion export)
- **SFT (finetuned) checkpoint**: set `SFT_CHECKPOINT` to the result directory produced by the
  [training entry point](../../../scripts/training/README.md) and run:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/inference/text_generation.py \
    --hf_model_path unsloth/gpt-oss-20b-BF16 \
    --megatron_model_path ${WORKSPACE}/results/gpt_oss_20b_finetune_tp2_pp2_ep4_spTrue_cp1 \
    --prompt "Hello, how are you?" \
    --max_new_tokens 64 \
    --tp 2 --pp 2 --ep 2 --etp 1 \
    --use-legacy-generation \
    --attention-backend local \
    --trust-remote-code
```

TP×PP×EP must equal `--nproc_per_node`. Adjust parallelism to match your SFT run.

## Evaluation

### GSM8K (zero-shot chain-of-thought)

Evaluate a SFT checkpoint on GSM8K using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via the NeMo evaluation framework:

```bash
python /opt/Evaluator/scripts/evaluation_with_nemo_run.py  \
    --megatron_checkpoint ${WORKSPACE}/results/<checkpoint_dir>/<iter> \
    --evaluation_result_dir ${WORKSPACE}/results/eval_<run_name> \
    --serving_backend ray \
    --endpoint_type chat \
    --eval_task gsm8k_cot_instruct \
    --nodes 1 --devices 8 \
    --tensor_parallelism_size 2 \
    --pipeline_parallelism_size 1 \
    --expert_model_parallel_size 4 \
    --batch_size 8 --parallel_requests 8 \
    --additional_args="--legacy_model_format"
```

Replace `<checkpoint_dir>/<iter>` with your SFT result path (e.g. `gpt_oss_20b_openmathinstruct2_gsm8k_finetune_tp2_pp2_ep4_spTrue_cp1/iter_0001000`).
The script deploys an inference server, runs lm-eval against it, and writes results to `<evaluation_result_dir>/megatron_model/results_*.json`.
Scores are reported as `flexible-extract` (flex) and `strict-match` (strict) accuracy.

## SFT Tuning Learnings (GSM8K)

Findings from hyperparameter tuning on GPT-OSS 20B × OpenMathInstruct-2:

- **Chat template**: Must match at both train and eval time.
- **Analysis channel format**: Use `generated_solution` from OpenMathInstruct-2 as the `analysis` channel and put only the final answer in `final`, rather than mixing both in `final`. This should be better theoretically since it matches what the channel architecture is designed for — the model reasons freely in `analysis` and commits to the final answer in `final`.
  - Plain: `<|start|>assistant<|channel|>final<|message|>{CoT} #### N<|end|>`
  - Analysis: `<|start|>assistant<|channel|>analysis<|message|>{CoT}<|end|>` + `<|start|>assistant<|channel|>final<|message|>#### N<|end|>`
- **Packed sequences**: Eliminates padding waste; reduced a 1-epoch run from ~17 h to within 4 h on 2 nodes × 8 H100. The dataset builder prepares and caches packed data automatically.
- **Hyperparameters** — strict-match improved **86.05% → 93.6%** by:
  - `global_batch_size`: 8 → 128
  - `train_iters`: 1 full epoch
  - `min_lr`: 0 → 1/10 × `max_lr` (e.g. 5e-7 when `lr=5e-6`)
