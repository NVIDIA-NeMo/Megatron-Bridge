# Nemotron 3.5 Super VL: IITC LoRA fine-tuning

Train on 10,000 interleaved image/text examples using Bridge's standard LoRA
recipe. The recorded results on all 658 IITC-8K test examples were:

| Checkpoint | Picture-reference accuracy | ROUGE-L × 100 |
| --- | ---: | ---: |
| Original | 85.71% | 35.83 |
| Bridge LoRA, step 625 | 87.99% | 54.01 |

Both evaluations used the same **custom English single-citation prompt**.
The accuracy gain is inconclusive (95% CI: −0.16 to +4.85 percentage points).
ROUGE-L measures text overlap; its increase alone does not prove better visual
grounding. These are historical results; the CLI workflow below has not been
rerun through a complete training/export/evaluation cycle. A different starting
checkpoint will not necessarily reproduce these scores.

## Prepare the data

From a configured Bridge checkout, set `HF_MODEL` to a local HF-format
Nemotron 3.5 Super VL checkpoint compatible with the recipe, including its
processor and model code. Use the checkpoint as supplied; no alias-creation
or checkpoint-layout preparation step is required by this tutorial.
Record the checkpoint revision for reproducibility.
On the recorded Bridge revision, the recipe factory also loads its configured
default HF model before CLI overrides; that configuration must be accessible
or cached. The local checkpoint override alone does not remove this dependency.

```bash
export EXAMPLE="$PWD/examples/models/nemotron/nemotron_3/super_vl/iitc"
export IITC_ROOT="$PWD/work/super-vl-iitc"
export HF_MODEL=/path/to/hf-checkpoint
export OUTPUT_DIR="$IITC_ROOT/step-625"

uv run --no-project python "$EXAMPLE/prepare_data.py" --download \
  --source "$IITC_ROOT/source" --images "$IITC_ROOT/images" \
  --output "$IITC_ROOT/energon-16k"
```

[selection.csv](iitc/selection.csv) is simply the fixed list of source-row IDs
and hashes identifying the exact 10,000 training and 210 reserved validation
examples and their order. The preparation script checks the hashes before
packaging; Bridge itself does not require this file. Preparation downloads
[VEGA](https://huggingface.co/datasets/zhourax977/VEGA) at revision
`417bd55ab869a872cef9ae4f12a9b2073f619ac4`, preserves text/image order, and writes
indexed Energon shards. Use new output directories; the image archive is about
46 GB. Training retains VEGA's original instruction, including its Chinese
fallback clause; evaluation uses the same English instruction for both models.

## Train through the Bridge CLI

Use two nodes with eight H100 80GB GPUs each. Run this command once per node,
with `NODE_RANK=0` or `1` and the same `MASTER_ADDR` pointing to node 0. Paths
and the configured Bridge environment must be available on both nodes.
Authenticate W&B before launching.

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --no-project python -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port=29501 \
  scripts/training/run_recipe.py \
  --recipe nemotron_35_super_vl_peft_16gpu_h100_bf16_config \
  --mode lora --step-func nemotron_omni_step \
  --max_steps 625 --seq_length 16384 \
  --pretrained_checkpoint "$HF_MODEL" \
  --save_dir "$OUTPUT_DIR/checkpoints" --save_interval 50 \
  dataset.path="$IITC_ROOT/energon-16k" dataset.num_workers=0 \
  dataset.task_encoder.hf_processor_path="$HF_MODEL" \
  dataset.task_encoder.hf_processor_revision=null \
  dataset.task_encoder.use_temporal_video_embedder=false \
  model.temporal_patch_dim=1 model.separate_video_embedder=false \
  model.temporal_ckpt_compat=false \
  model.recompute_granularity=full model.recompute_method=uniform \
  model.recompute_num_layers=1 \
  scheduler.lr_warmup_iters=10 scheduler.lr_decay_iters=625 \
  train.exit_duration_in_mins=210 \
  checkpoint.hf_source_path="$HF_MODEL" checkpoint.hf_trust_remote_code=true \
  checkpoint.save_optim=true checkpoint.save_rng=true checkpoint.async_save=false \
  logger.wandb_project=super-vl-iitc logger.wandb_exp_name=iitc-step-625 \
  logger.wandb_save_dir="$OUTPUT_DIR/wandb" \
  logger.save_config_filepath="$OUTPUT_DIR/resolved-config.yaml"
```

Recipe defaults supply rank/alpha **32/32**, dropout **0**, language-only adapter
targets, global/micro batch **16/1**, TP/PP/EP **4/2/8**, BF16, seed **1234**, and
LR **1e-4 → 0**. Vision tower and projector are frozen. For interrupted training,
repeat with `--load_dir "$OUTPUT_DIR/checkpoints"`. Add `--dry-run` to inspect
the resolved configuration without training. No custom `train.py` is needed.

## Merge/export limitation

The command above saves a **native Bridge checkpoint**. On this revision,
pipeline-parallel HF export after PEFT training can fail with
`Can't pickle local object ... peft_pre_wrap_hook`: shared export code fails
to remove nested training hooks before sending configuration between ranks.
This is an export serialization bug; it does not establish a training or
dataloader failure. It can affect other models using the same export path.

The historical experiment also added compatibility aliases to its source HF
checkpoint. Those additions created extra strict-export requirements and are
not part of this tutorial. Image-only training disables the video embedder;
exporting a complete multimodal HF checkpoint also requires handling that
omitted branch. These layout concerns are separate from the serialization bug.

Library fixes are separate from this tutorial PR. There is **no verified
standalone merge/export command here**, and full Bridge/HF numerical parity
has not been established. Evaluation requires an already merged, standalone
HF checkpoint (`MERGED_HF`); a native checkpoint cannot be passed directly.

## Evaluate the original and merged checkpoints

Run one process on an eight-GPU node, using Transformers 5.12.1 and
`rouge==1.0.1`. Keep the original processor, all 658 test rows, image order,
chat template (`enable_thinking=False`), and greedy 512-token decoding matched.
The evaluator records full answers and rejects mismatched paired inputs.

```bash
export MERGED_HF=/path/to/already-merged-hf-checkpoint

uv run --no-project python "$EXAMPLE/evaluate.py" \
  --model "$HF_MODEL" --processor "$HF_MODEL" \
  --source "$IITC_ROOT/source/IITC_8k_test.json" --images "$IITC_ROOT/images" \
  --output "$IITC_ROOT/eval-original"
uv run --no-project python "$EXAMPLE/evaluate.py" \
  --model "$MERGED_HF" --processor "$HF_MODEL" \
  --source "$IITC_ROOT/source/IITC_8k_test.json" --images "$IITC_ROOT/images" \
  --output "$IITC_ROOT/eval-625"
uv run --no-project python "$EXAMPLE/evaluate.py" \
  --compare "$IITC_ROOT/eval-original" "$IITC_ROOT/eval-625" \
  --output "$IITC_ROOT/comparison-625.json"
```

Picture-reference accuracy requires exactly one correct `[Picture N]` citation;
ROUGE-L is mean reference-answer overlap. The scoring follows
[VEGA's evaluator](https://github.com/zhourax/VEGA/blob/96d4be247fb4385b23265ac4f0b6079a9225698d/eval/IITC.py),
with the custom prompt defined in [evaluate.py](iitc/evaluate.py); report that
prompt difference when sharing scores.
