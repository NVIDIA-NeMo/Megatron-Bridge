# Training entry points

Megatron Bridge training provides a small public Slurm launcher:

```bash
./scripts/training/train.sh [launch options] [training options] [KEY=VALUE overrides]
```

`train.sh` invokes `setup_experiment.py` on a Slurm login node and submits `run_recipe.py` directly through Slurm. The
setup layer only owns resources, the container, explicitly forwarded environment variables, and explicit mounts.
Recipe selection, dataset construction, and ConfigContainer overrides are resolved inside the training environment.
Without an active virtual environment, the shell entry point creates an isolated `nemo-run` environment rather than
resolving the full GPU training dependency set on the login node.

`launch_with_nemo_run.py` and `launch_with_sbatch.sh` remain available for their existing specialized workflows; `train.sh` is the compact recipe-oriented path.

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
| `megatron-indexed` | pretrain | Local Megatron `.bin/.idx` data; never falls back to mock |
| `squad` | sft/lora/dora | Hugging Face SQuAD preset |
| `openmathinstruct2` | sft/lora/dora | OpenMathInstruct-2 prompt/completion preset |
| `openmathinstruct2-thinking` | sft/lora/dora | OpenMathInstruct-2 analysis/final channel format |
| `gsm8k` | sft/lora/dora | GSM8K preset |
| `local-jsonl` | sft/lora/dora | Local prompt-completion JSONL selected by `--dataset-root` |
| `cord-v2` | sft/lora/dora | CORD-v2 receipt VQA preset |
| `medpix` | sft/lora/dora | MedPix medical VQA preset |
| `raven` | sft/lora/dora | RAVEN visual reasoning preset |
| `rdr` | sft/lora/dora | RDR visual reasoning preset |
| `llava-video-178k` | sft/lora/dora | LLaVA-Video preset; requires `--media-root` |
| `local-vlm` | sft/lora/dora | Local VLM JSON/JSONL selected by explicit path flags |

### Megatron indexed data

Any pretraining corpus must first be converted to Megatron indexed data. Select one or more prefixes with:

```bash
--dataset megatron-indexed --dataset-path /data/dclm/dclm_01_01_text_document
--dataset megatron-indexed --dataset-path /data/dclm
```

Every selected prefix must have matching `.bin` and `.idx` files. `--dataset-cache` selects the index cache. Missing or
incomplete data fails before distributed training starts.

The launcher does not infer the source corpus from the files. For one preprocessing example, see
[the DCLM tutorial](../../tutorials/data/dclm/README.md).

### OpenMathInstruct-2

`openmathinstruct2` uses prompt/completion records. `openmathinstruct2-thinking` changes only the semantic output
format: chain-of-thought goes to the analysis channel and the answer goes to the final channel.

### Offline packing

Offline packing is a text SFT option, not a dataset name. Add `--offline-packing` to `squad`, either OpenMathInstruct-2
format, `gsm8k`, or `local-jsonl`. The launcher aligns packed padding for the selected CP/TP and sequence-parallel
configuration. Packed training requires `--micro-batch-size 1`. The builder materializes packed data automatically, so
a separate packing Slurm job is not required. The selected recipe/model must support packed THD sequences; for example,
GLM-4.5 and Qwen3-Next recipes currently do not. On multiple nodes, keep the dataset cache on shared mounted storage.

### VLM data

The hosted VLM names use the existing Hugging Face dataset adapters and retain the processor and in-batch packing
settings from the selected VLM recipe. `raven`, `rdr`, and `llava-video-178k` derive deterministic 95/5 train and
validation slices; `cord-v2` and `medpix` use their published validation splits. Pass a VLM forward step explicitly.

For local JSON/JSONL, select `local-vlm` and pass `--train-data-path`; optional split paths are
`--validation-data-path` and `--test-data-path`. Rows and media paths must already use the selected processor's
supported conversation schema. `--hf-processor-path` overrides the processor inherited from the VLM recipe.

## SFT and LoRA checkpoints

Use `--from` for the pretrained native Megatron checkpoint or local Hugging Face model directory:

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 8 \
    --account ACCOUNT --partition PARTITION --container-image IMAGE \
    --mount /checkpoints \
    --recipe gpt_oss_20b_sft_8gpu_h100_bf16_config \
    --mode sft --dataset openmathinstruct2-thinking --offline-packing \
    --micro-batch-size 1 \
    --from /checkpoints/gpt-oss-20b
```

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 1 \
    --account ACCOUNT --partition PARTITION --container-image IMAGE \
    --mount /checkpoints \
    --recipe gpt_oss_20b_peft_1gpu_h100_bf16_config \
    --mode lora --dataset openmathinstruct2-thinking --offline-packing \
    --micro-batch-size 1 \
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

Set `CONTAINER_IMAGE` to avoid repeating `--container-image`. On clusters that allocate whole GPU nodes implicitly, pass `--no-gpu-resource-request` to omit the explicit Slurm GPU request while retaining one task per requested GPU. Environment variables and filesystem paths are never
forwarded implicitly. Export credentials in the launcher environment, then repeat `--env NAME` to forward names without
materializing their values in the generated sbatch script. Repeat `--mount HOST` for the same host and container path, or use
`--mount HOST:CONTAINER` when the paths differ. Mount every dataset, checkpoint, cache, and output path the job needs.

The launcher submits the experiment in detached mode and returns after Slurm accepts the job. Inspect its state and
logs with the cluster's normal `squeue`, `sacct`, and log-file workflow.

Use `--dry-run` (or the explicit `--submission-dry-run` spelling) to render the NeMo-Run experiment without
submitting it. For a rank-local ConfigContainer dry run, invoke `run_recipe.py` directly:

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
