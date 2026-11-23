# Nemotron 3.5 Super VL: IITC LoRA fine-tuning

[VEGA IITC](https://huggingface.co/datasets/zhourax977/VEGA) contains
questions about documents with interleaved text and images. This example fine-tunes
Nemotron 3.5 Super VL with Bridge's standard LoRA recipe on 10,000 examples,
then evaluates answer overlap and picture-reference accuracy.

## Prepare the data

Set `HF_MODEL` to a downloaded Nemotron 3.5 Super VL checkpoint. Preparation
loads only its processor to check sample lengths.

```bash
export EXAMPLE="$PWD/examples/models/nemotron/nemotron_3_5_super_vl/iitc/src"
export IITC_ROOT="$PWD/work/super-vl-iitc"
export HF_MODEL=/path/to/hf-checkpoint

uv run --no-project python "$EXAMPLE/prepare_data.py" --download \
  --source "$IITC_ROOT/source" --images "$IITC_ROOT/images" \
  --output "$IITC_ROOT/energon" --processor "$HF_MODEL"
```

The published IITC-4K/8K splits contain 374,575 training examples
(192,442 + 182,133) and 1,330 test examples (672 + 658).
For illustration, we select 10,000 for training and reserve 210 for validation
from the training splits. Final evaluation uses all 658 IITC-8K test examples.
The original selection used seeded hash ranking, excluded test-paper/question
and duplicate-image overlap, and kept complete examples within 16K tokens.
The preparation script now creates a fresh illustrative subset with seed
20261004, paper-disjoint splits, question deduplication, and the same token
limit; it does not recreate the exact historical selection. Text/image order
is preserved in Energon shards. The image archive is about 46 GB.

## Train through the Bridge CLI

Run the following command on two nodes with eight H100 GPUs each, using the
default LoRA recipe with CLI overrides.

```bash
export OUTPUT_DIR="$IITC_ROOT/lora-run"
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --nodes=2 --ntasks=16 --ntasks-per-node=8 \
  uv run --no-project python \
  scripts/training/run_recipe.py \
  --recipe nemotron_35_super_vl_peft_16gpu_h100_bf16_config \
  --mode lora --step-func nemotron_omni_step \
  --max_steps 625 --seq_length 16384 \
  --pretrained_checkpoint "$HF_MODEL" \
  --save_dir "$OUTPUT_DIR/checkpoints" --save_interval 50 \
  dataset.path="$IITC_ROOT/energon" dataset.num_workers=0 \
  dataset.task_encoder.hf_processor_path="$HF_MODEL" \
  dataset.task_encoder.hf_processor_revision=null \
  dataset.task_encoder.use_temporal_video_embedder=false \
  model.temporal_patch_dim=1 model.separate_video_embedder=false \
  model.temporal_ckpt_compat=false \
  model.recompute_granularity=selective \
  'model.recompute_modules=[layernorm,moe]' \
  scheduler.lr_warmup_iters=10 scheduler.lr_decay_iters=625 \
  train.exit_duration_in_mins=210 \
  checkpoint.hf_source_path="$HF_MODEL" checkpoint.hf_trust_remote_code=true \
  checkpoint.save_optim=true checkpoint.save_rng=true checkpoint.async_save=false \
  logger.wandb_project=super-vl-iitc logger.wandb_exp_name=iitc-lora \
  logger.wandb_save_dir="$OUTPUT_DIR/wandb" \
  logger.save_config_filepath="$OUTPUT_DIR/resolved-config.yaml"
```

Recipe defaults supply rank/alpha 32/32, dropout 0, language-only adapter
targets, global/micro batch 16/1, TP/PP/EP 4/2/8, BF16, seed 1234, and
LR 1e-4 → 0. Vision tower and projector are frozen.
625 optimizer updates × 16 samples = 10,000 sample presentations, about
one epoch's worth. `16384` is the training sequence limit in tokens.

## Merge and export to HF format after training

After step 625, merge the adapters with the base weights and export to HF
before evaluation.

Convert the base checkpoint once, then use Bridge's existing
[merge script](../../../examples/peft/merge_lora.py) on the same two-node allocation, keeping the training parallelism:

```bash
export BASE_NATIVE="$OUTPUT_DIR/base-native"
export MERGED_HF="$OUTPUT_DIR/merged-hf"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun --nodes=2 --ntasks=16 --ntasks-per-node=8 bash -c \
  'RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID uv run --no-project python "$@"' \
  _ scripts/conversion/run_conversion.py import --device gpu \
  --hf-model "$HF_MODEL" --megatron-path "$BASE_NATIVE" \
  --tp 4 --pp 2 --ep 8 --etp 1 --torch-dtype bfloat16 \
  --trust-remote-code --low-memory-save

srun --nodes=2 --ntasks=16 --ntasks-per-node=8 bash -c \
  'RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID uv run --no-project python "$@"' \
  _ examples/peft/merge_lora.py \
  --lora-checkpoint "$OUTPUT_DIR/checkpoints/iter_0000625" \
  --pretrained "$BASE_NATIVE" --hf-model-path "$HF_MODEL" \
  --output "$MERGED_HF" --tp 4 --pp 2 --ep 8
```

## Evaluate the original and merged checkpoints

Run one process on an eight-GPU node with `rouge==1.0.1`.
Use the full IITC-8K test split (658 examples); 8K labels its context-length
category, not the number of examples. Keep the original processor, image order,
chat template (`enable_thinking=False`), and greedy 512-token decoding matched.
The evaluator saves generations and metrics for each checkpoint.

```bash
uv run --no-project python "$EXAMPLE/evaluate.py" \
  --model "$HF_MODEL" --processor "$HF_MODEL" \
  --source "$IITC_ROOT/source/IITC_8k_test.json" --images "$IITC_ROOT/images" \
  --output "$IITC_ROOT/eval-original"
uv run --no-project python "$EXAMPLE/evaluate.py" \
  --model "$MERGED_HF" --processor "$HF_MODEL" \
  --source "$IITC_ROOT/source/IITC_8k_test.json" --images "$IITC_ROOT/images" \
  --output "$IITC_ROOT/eval-finetuned"
```

Picture-reference accuracy requires exactly one correct `[Picture N]` citation;
ROUGE-L is mean reference-answer overlap. The scoring follows
[VEGA's evaluator](https://github.com/zhourax/VEGA/blob/96d4be247fb4385b23265ac4f0b6079a9225698d/eval/IITC.py),
with the custom prompt defined in [evaluate.py](../../../examples/models/nemotron/nemotron_3_5_super_vl/iitc/src/evaluate.py); report that
prompt difference when sharing scores.

## Results

Historical results using the original selection, evaluated on all 658
IITC-8K held-out test examples:

| Checkpoint | Picture-reference accuracy | ROUGE-L × 100 |
| --- | ---: | ---: |
| Original | 85.71% | 35.83 |
| Bridge LoRA, step 625 | 87.99% | 54.01 |
