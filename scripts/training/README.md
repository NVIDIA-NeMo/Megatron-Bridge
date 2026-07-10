# Training entry point

Megatron Bridge training uses one public launcher:

```bash
./scripts/training/train.sh [launch options] [training options] [KEY=VALUE overrides]
```

`train.sh` invokes `setup_experiment.py`, which launches `run_recipe.py` locally or through Slurm. The setup layer
only owns resources, containers, environment variables, and mounts. Recipe selection, dataset construction, and
ConfigContainer overrides are resolved inside the training environment.

The former `launch_with_nemo_run.py` and `launch_with_sbatch.sh` entry points have been removed.

## Selection rules

Choose either a complete recipe or the shorthand model selector. `--recipe` and `--model` are mutually exclusive.

### Complete recipe

A complete recipe already identifies the model and default training configuration, so do not also pass `--model`:

```bash
./scripts/training/train.sh \
    --local --gpus-per-node 1 \
    --recipe llama32_1b_pretrain_config \
    --mode pretrain --dataset mock
```

### Model selector

The selector resolves a hardware- and topology-specific recipe from the family, model, mode, GPU type, and total GPU
count. `setup_experiment.py` derives the total GPU count from `--nodes * --gpus-per-node`.

```bash
./scripts/training/train.sh \
    --nodes 2 --gpus-per-node 8 --gpu-type h100 \
    --account ACCOUNT --partition PARTITION \
    --container-image /path/to/container.sqsh \
    --source recipes --family gpt_oss --model gpt_oss_20b \
    --mode pretrain --dataset dclm
```

Public modes are `pretrain`, `sft`, `peft`, `lora`, and `dora`. `lora` and `dora` select a PEFT recipe and set the
corresponding PEFT scheme. The legacy `--task` spelling remains a compatibility alias.

## Dataset presets

`--dataset` names the data users intend to train on rather than the internal dataset config class.

| Dataset | Mode | Behavior |
|---|---|---|
| `mock` | pretrain | In-memory mock GPT data |
| `dclm` | pretrain | Preprocessed Megatron `.bin/.idx` data; never falls back to mock |
| `squad` | SFT/PEFT | Hugging Face SQuAD preset |
| `squad-packed` | SFT/PEFT | SQuAD with offline sequence packing |
| `openmathinstruct2` | SFT/PEFT | OpenMathInstruct-2 prompt/completion preset |
| `openmathinstruct2-thinking` | SFT/PEFT | Analysis/final channel format with offline packing |
| `gsm8k` | SFT/PEFT | GSM8K preset |

The internal values `llm-pretrain`, `llm-pretrain-mock`, `llm-finetune`, and the VLM dataset types remain available
for compatibility and advanced use.

### DCLM

DCLM must be preprocessed into indexed Megatron data. Select it with one of:

```bash
--dataset dclm --dataset-path /data/dclm/dclm_01_01_text_document
--dataset dclm --dataset-path /data/dclm
```

or set `DCLM_DATA_PREFIX` or `DCLM_DATA_DIR`. Every selected prefix must have matching `.bin` and `.idx` files.
`--dataset-cache` or `DCLM_CACHE` selects the index cache. Missing or incomplete data fails before distributed
training starts.

For preprocessing instructions, see [the DCLM tutorial](../../tutorials/data/dclm/README.md).

### OpenMathInstruct-2

The builder downloads/materializes the Hugging Face source and prepares packed data when
`openmathinstruct2-thinking` is selected. A separate packing Slurm job is not required. Multi-node training must set
`NEMO_DATASETS_CACHE` or `NEMO_HOME` to shared storage.

## SFT and LoRA checkpoints

Use `--from` for the pretrained native Megatron checkpoint or local Hugging Face model directory:

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 8 --gpu-type h100 \
    --account ACCOUNT --partition PARTITION --container-image IMAGE \
    --recipe gpt_oss_20b_sft_8gpu_h100_bf16_config \
    --mode sft --dataset openmathinstruct2-thinking \
    --from /checkpoints/gpt-oss-20b
```

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 1 --gpu-type h100 \
    --account ACCOUNT --partition PARTITION --container-image IMAGE \
    --recipe gpt_oss_20b_peft_1gpu_h100_bf16_config \
    --mode lora --dataset openmathinstruct2-thinking \
    --from /checkpoints/gpt-oss-20b
```

## Overrides

Common values have flags and every ConfigContainer field can be overridden with trailing `KEY=VALUE` arguments:

```bash
./scripts/training/train.sh \
    --local --gpus-per-node 1 \
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
--container-image (or MB_CONTAINER_IMAGE)
```

Use repeated `--mount host:container` and `--env KEY=VALUE` values for deployment-specific settings. The launcher
automatically inherits the standard HF/NeMo/DCLM/uv cache variables and authentication variables without logging
their values. Existing dataset, checkpoint, cache, and output paths passed through common flags are mounted at the
same path automatically.

Use `--dry-run` to render the NeMo-Run experiment without submitting it. For a rank-local ConfigContainer dry run,
invoke `run_recipe.py` directly:

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

Performance recipes under `megatron.bridge.perf_recipes` remain throughput benchmark configurations, typically on
mock data. Use library recipes for functional pretraining, SFT, and PEFT.
