# Nemotron 3.5 Super VL: IITC LoRA fine-tuning

[VEGA IITC](https://huggingface.co/datasets/zhourax977/VEGA) contains
questions about documents with interleaved text and images. Answers include
supporting picture references such as `[Picture 2]`. This example fine-tunes
Nemotron 3.5 Super VL with Bridge's standard LoRA recipe on 10,000 examples,
then evaluates answer overlap and picture-reference accuracy.

## Prepare the data

From a configured Bridge checkout, download an available Nemotron 3.5 Super VL
checkpoint from Hugging Face and set `HF_MODEL` to its downloaded directory.

```bash
export EXAMPLE="$PWD/examples/models/nemotron/nemotron_3/super_vl/iitc"
export IITC_ROOT="$PWD/work/super-vl-iitc"
export HF_MODEL=/path/to/hf-checkpoint
export OUTPUT_DIR="$IITC_ROOT/lora-run"

uv run --no-project python "$EXAMPLE/prepare_data.py" --download \
  --source "$IITC_ROOT/source" --images "$IITC_ROOT/images" \
  --output "$IITC_ROOT/energon"
```

We select **10,000 training and 210 validation examples** from the published
IITC-4K/8K **training** splits; the test splits remain separate.
[selection.csv](iitc/selection.csv) fixes the selected rows and their order for
reproducibility. Preparation preserves text/image order and writes Energon
shards. The image archive is about 46 GB; use new output directories.

## Train through the Bridge CLI

Use a Slurm allocation with two nodes, each with eight H100 80GB GPUs, shared
paths, and the configured Bridge environment. Authenticate W&B, then launch
once from the allocation with `srun`; Bridge derives ranks from Slurm.

```bash
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
  model.recompute_granularity=full model.recompute_method=uniform \
  model.recompute_num_layers=1 \
  scheduler.lr_warmup_iters=10 scheduler.lr_decay_iters=625 \
  train.exit_duration_in_mins=210 \
  checkpoint.hf_source_path="$HF_MODEL" checkpoint.hf_trust_remote_code=true \
  checkpoint.save_optim=true checkpoint.save_rng=true checkpoint.async_save=false \
  logger.wandb_project=super-vl-iitc logger.wandb_exp_name=iitc-lora \
  logger.wandb_save_dir="$OUTPUT_DIR/wandb" \
  logger.save_config_filepath="$OUTPUT_DIR/resolved-config.yaml"
```

Recipe defaults supply rank/alpha **32/32**, dropout **0**, language-only adapter
targets, global/micro batch **16/1**, TP/PP/EP **4/2/8**, BF16, seed **1234**, and
LR **1e-4 → 0**. Vision tower and projector are frozen.
**625 optimizer updates × 16 samples = 10,000 sample presentations**, about
one epoch's worth. `16384` is the training sequence limit in tokens.

## Merge and export after training

Merge the **step-625** adapters into the original base weights and export a
standalone HF checkpoint **before inference**. The training CLI saves native
adapters; it does not automatically produce a merged model. The following
small export script uses Bridge's loading and merge/export APIs on eight GPUs.

```bash
export ADAPTER_CKPT="$OUTPUT_DIR/checkpoints/iter_0000625"
export MERGED_HF="$OUTPUT_DIR/merged-hf"
cat > "$OUTPUT_DIR/merge_export.py" <<'PYTHON'
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.bridge.peft.utils import create_peft, create_peft_hook, load_peft_adapter_checkpoint
from megatron.bridge.training.utils.checkpoint_utils import read_run_config
from megatron.bridge.utils.common_utils import maybe_initialize_distributed

maybe_initialize_distributed()
source = os.environ["HF_MODEL"]
checkpoint = os.environ["ADAPTER_CKPT"]
bridge = AutoBridge.from_hf_pretrained(source, trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=True)
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 2
provider.expert_model_parallel_size = 4
provider.expert_tensor_parallel_size = 1
provider.sequence_parallel = True
provider.params_dtype = provider.pipeline_dtype = torch.bfloat16
provider.bf16 = True
provider.temporal_patch_dim = 1
provider.separate_video_embedder = False
provider.temporal_ckpt_compat = False
provider.moe_token_dispatcher_type = "alltoall"
provider.moe_permute_fusion = False
provider.finalize()
peft_config = read_run_config(f"{checkpoint}/run_config.yaml")["peft"]
peft = create_peft(peft_config)
provider.register_pre_wrap_hook(create_peft_hook(peft, training=False))
provider.initialize_model_parallel(seed=1234)
model = provider.provide_distributed_model(wrap_with_ddp=False)
load_peft_adapter_checkpoint(model, checkpoint, peft)
bridge.save_hf_pretrained(
    model, os.environ["MERGED_HF"], source_path=source,
    merge_adapter_weights=True, strict=True, weight_dtype=torch.bfloat16,
)
dist.barrier()
dist.destroy_process_group()
PYTHON
srun --nodes=1 --ntasks=8 --ntasks-per-node=8 \
  uv run --no-project python "$OUTPUT_DIR/merge_export.py"
```

## Evaluate the original and merged checkpoints

Run one process on an eight-GPU node with `rouge==1.0.1`.
Use the **full IITC-8K test split (658 examples)**; 8K labels its context-length
category, not the number of examples. Keep the original processor, image order,
chat template (`enable_thinking=False`), and greedy 512-token decoding matched.
The evaluator records full answers and rejects mismatched paired inputs.

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

Compare the saved answers (CPU only; this is **not a third inference run**):

```bash
uv run --no-project python "$EXAMPLE/evaluate.py" \
  --compare "$IITC_ROOT/eval-original" "$IITC_ROOT/eval-finetuned" \
  --output "$IITC_ROOT/comparison.json"
```

Picture-reference accuracy requires exactly one correct `[Picture N]` citation;
ROUGE-L is mean reference-answer overlap. The scoring follows
[VEGA's evaluator](https://github.com/zhourax/VEGA/blob/96d4be247fb4385b23265ac4f0b6079a9225698d/eval/IITC.py),
with the custom prompt defined in [evaluate.py](iitc/evaluate.py); report that
prompt difference when sharing scores.

## Results

Evaluation on all **658 IITC-8K held-out test examples**:

| Checkpoint | Picture-reference accuracy | ROUGE-L × 100 |
| --- | ---: | ---: |
| Original | 85.71% | 35.83 |
| Bridge LoRA, step 625 | 87.99% | 54.01 |

Step 625 logged **10,000 consumed training samples** (625 updates × global
batch size 16). This counter measures consumed samples, not distinct dataset
rows visited.
