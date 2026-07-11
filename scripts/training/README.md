# Training entry point

Megatron Bridge training uses one public launcher:

```bash
./scripts/training/train.sh [launch options] [training options] [KEY=VALUE overrides]
```

`train.sh` invokes `setup_experiment.py` on a Slurm login node and submits `run_recipe.py` directly through Slurm. The
setup layer only owns resources, the container, explicitly forwarded environment variables, and explicit mounts.
Recipe selection, dataset construction, and ConfigContainer overrides are resolved inside the training environment.

The former `launch_with_nemo_run.py` and `launch_with_sbatch.sh` entry points have been removed.

## Selection rules

Choose exactly one of a complete recipe or a model selector. `--recipe` and `--model` are mutually exclusive. Every
invocation requires one of `--mode pretrain`, `--mode sft`, `--mode lora`, or `--mode dora`.

### Complete recipe

A complete recipe already identifies the model and default training configuration, so do not also pass `--model`:

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 8 \
    --account ACCOUNT --partition PARTITION \
    --container-image /path/to/container.sqsh \
    --recipe llama32_1b_pretrain_config \
    --mode pretrain --dataset mock
```

### Model selector

The model selector combines the model stem and mode to load the corresponding library recipe. For example,
`--model gpt_oss_20b --mode sft` loads `gpt_oss_20b_sft_config`; LoRA and DoRA load the model's PEFT recipe and set the
requested adapter scheme.

```bash
./scripts/training/train.sh \
    --nodes 2 --gpus-per-node 8 \
    --account ACCOUNT --partition PARTITION \
    --container-image /path/to/container.sqsh \
    --model gpt_oss_20b \
    --mode pretrain --dataset mock
```

The default forward step is `llm_step`. Pass `--step-func NAME` explicitly for a recipe that needs another registered
forward step.

Pass `--hf-path` (or `--hf_path`) when the selected recipe factory accepts an alternate Hugging Face model path, and
`--seq-length` (or `--seq_length`) when it accepts a sequence-length argument.

## Dataset presets

`--dataset` names the data users intend to train on rather than the internal dataset config class.

| Dataset | Mode | Behavior |
|---|---|---|
| `mock` | pretrain | In-memory mock GPT data |
| `dclm` | pretrain | Preprocessed Megatron `.bin/.idx` data; never falls back to mock |
| `rp2` | pretrain | Preprocessed Megatron `.bin/.idx` data |
| `c4` | pretrain | Preprocessed Megatron `.bin/.idx` data |
| `squad` | sft/lora/dora | Hugging Face SQuAD preset |
| `squad-packed` | sft/lora/dora | SQuAD with offline sequence packing |
| `openmathinstruct2` | sft/lora/dora | OpenMathInstruct-2 prompt/completion preset |
| `openmathinstruct2-thinking` | sft/lora/dora | Analysis/final channel format with offline packing |
| `gsm8k` | sft/lora/dora | GSM8K preset |
| `local-jsonl` | sft/lora/dora | Local prompt-completion JSONL selected by `dataset.dataset_root` |
| `preloaded-vlm` | sft/lora/dora | Local VLM JSON/JSONL selected by dataset path overrides |

### DCLM

DCLM must be preprocessed into indexed Megatron data. Select it with one of:

```bash
--dataset dclm --dataset-path /data/dclm/dclm_01_01_text_document
--dataset dclm --dataset-path /data/dclm
```

Every selected prefix must have matching `.bin` and `.idx` files. `--dataset-cache` selects the index cache. Missing or
incomplete data fails before distributed training starts.

For preprocessing instructions, see [the DCLM tutorial](../../tutorials/data/dclm/README.md).

### Text-only model validation matrix

The [text-only pretrain inventory](../../examples/models/text-pretrain-validation.md)
tracks the PR #4805 text models under one reproducible contract: two 8-GPU
H100 nodes, real indexed DCLM, 100 iterations, per-step logging, and one W&B
project/group. Use `submit_text_pretrain_validation.py` to render or submit
selected rows through this launcher. The driver renders commands unless
`--submit` is explicitly provided.

### OpenMathInstruct-2

The builder downloads/materializes the Hugging Face source and prepares packed data when
`openmathinstruct2-thinking` is selected. A separate packing Slurm job is not required. For multi-node training, put
the recipe's dataset cache on shared storage, forward its environment variable with `--env NAME`, and mount the shared
path explicitly.

## SFT and LoRA checkpoints

Use `--from` for the pretrained native Megatron checkpoint or local Hugging Face model directory:

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 8 \
    --account ACCOUNT --partition PARTITION --container-image IMAGE \
    --mount /checkpoints \
    --recipe gpt_oss_20b_sft_8gpu_h100_bf16_config \
    --mode sft --dataset openmathinstruct2-thinking \
    --from /checkpoints/gpt-oss-20b
```

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 1 \
    --account ACCOUNT --partition PARTITION --container-image IMAGE \
    --mount /checkpoints \
    --recipe gpt_oss_20b_peft_1gpu_h100_bf16_config \
    --mode lora --dataset openmathinstruct2-thinking \
    --from /checkpoints/gpt-oss-20b
```

## Overrides

Common values have flags and every ConfigContainer field can be overridden with trailing `KEY=VALUE` arguments:

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 8 \
    --account ACCOUNT --partition PARTITION \
    --container-image /path/to/container.sqsh \
    --recipe llama32_1b_pretrain_config \
    --mode pretrain --dataset mock \
    --max-steps 10 --global-batch-size 8 --micro-batch-size 1 \
    optimizer.lr=0.0002 \
    scheduler.lr_warmup_iters=1 \
    scheduler.lr_decay_iters=10
```

Precedence is recipe defaults, dataset preset, common flags, then trailing ConfigContainer overrides.

## Slurm and containers

Required Slurm arguments are:

```text
--nodes
--gpus-per-node
--account
--partition
--container-image (or CONTAINER_IMAGE)
```

Set `CONTAINER_IMAGE` to avoid repeating `--container-image`. Environment variables and filesystem paths are never
forwarded implicitly. Repeat `--env NAME` to forward a variable from the login-node environment, or use
`--env NAME=VALUE`. Inherited names are forwarded through Slurm/container inheritance and are not serialized with
their values into the generated sbatch script; reserve `NAME=VALUE` for non-secret settings. Repeat `--mount HOST` for
the same host and container path, or use
`--mount HOST:CONTAINER` when the paths differ. Mount every dataset, checkpoint, cache, and output path the job needs.

The launcher submits the experiment in detached mode and returns after Slurm accepts the job. Inspect its state and
logs with the cluster's normal `squeue`, `sacct`, and log-file workflow.

Use `--submission-dry-run` to render the NeMo-Run experiment without submitting it. For a rank-local
ConfigContainer dry run, invoke `run_recipe.py` directly:

```bash
uv run python scripts/training/run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --mode pretrain --dataset mock \
    --dry-run --save-config /tmp/config.yaml
```

## Rank-local entry point

`run_recipe.py` remains available for existing distributed environments that already own process launch:

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
    scripts/training/run_recipe.py \
    --recipe vanilla_gpt_pretrain_config \
    --mode pretrain --dataset mock \
    model.tensor_model_parallel_size=2 \
    model.sequence_parallel=true
```

This entry point loads library recipes for functional pretraining, SFT, LoRA, and DoRA.
