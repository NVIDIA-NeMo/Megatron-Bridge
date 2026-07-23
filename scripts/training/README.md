# Training entry points

Megatron Bridge training provides a small public Slurm launcher:

```bash
./scripts/training/train.sh [launch options] [runner options] [KEY=VALUE overrides]
```

`train.sh` invokes `setup_experiment.py` on a Slurm login node and submits `run_recipe.py` directly through Slurm. The
setup layer only owns resources, the container, explicitly forwarded environment variables, and explicit mounts.
Recipe selection, dataset construction, and ConfigContainer overrides are resolved inside the training environment.
Without an active virtual environment, the shell entry point creates an isolated `nemo-run` environment rather than
resolving the full GPU training dependency set on the login node.

`launch_with_nemo_run.py` and `launch_with_sbatch.sh` remain available for their existing specialized workflows; `train.sh` is the compact recipe-oriented path.

## Selection rules

Choose exactly one of a complete recipe or a model selector. `--recipe` and `--model` are mutually exclusive. A complete
recipe is discovered automatically from its exported function name, whether it is a library or benchmark recipe;
there is no separate source flag. A model selector requires one of `--mode pretrain`, `--mode sft`, `--mode lora`, or
`--mode dora`; a conventional complete recipe name infers its mode when `--mode` is omitted.

### Library recipe

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

### Benchmark recipe

The training launcher can run exact exported recipes from `src/megatron/bridge/perf_recipes`, including text
pretraining, text SFT/PEFT, Qwen-VL pretraining, and Wan pretraining. The total allocation must match the GPU count
encoded by the recipe name; the user selects the node shape, and the Slurm partition must provide the requested
hardware:

`benchmark` is the unified runner's user-facing term. The existing `perf_recipes` package and `scripts/performance/`
compatibility paths retain their legacy names.

```bash
./scripts/training/train.sh \
    --nodes 2 --gpus-per-node 8 \
    --account ACCOUNT --partition PARTITION \
    --container-image IMAGE \
    --recipe qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config \
    --mode pretrain
```

Benchmark recipes provide canonical defaults for their dataset, parallelism topology, batch sizes, sequence length,
precision, dispatcher, CUDA-graph settings, and process environment. Trailing `KEY=VALUE` overrides are applied to their
`ConfigContainer` in the same way as library recipes; an overridden run no longer represents the canonical benchmark
configuration. Benchmark recipes retain their selected dataset type, so `--dataset` is not supported on this path.
Recipe environment defaults are installed before the launcher enters the training stack; values explicitly set by the
shell or Slurm environment retain precedence.

The launcher does not infer offline mode, cluster-specific CPU/NUMA binding, Slurm segment sizing, or NCCL fabric
settings from the recipe name. Supply those deployment settings explicitly through the target cluster's launcher
configuration, repeated `--srun-arg` options, or exported values forwarded with `--env NAME`.

The runner selects the registered text, multimodal, audio, omni, or diffusion forward step from the recipe identity,
regardless of which recipe package exports it. The compatibility launcher at
`scripts/performance/setup_experiment.py` remains available for selector-based invocation, dataset replacement, and
specialized legacy controls that are not part of the compact training CLI.

Text SFT/PEFT benchmark recipes retain the flat runner's mock-data default. Qwen-VL and Wan recipes retain their
model-specific dataset configuration. Exported benchmark PEFT recipes are fixed LoRA configs; DoRA remains available
through configurable library recipes. Explicit benchmark dataset replacement remains on the compatibility launcher.

Five legacy duplicate names resolve to the benchmark definition; their library workloads remain available through
the corresponding generic recipe aliases. New recipe names should be unique across both packages.

Text recipes default to `llm_step`; all recipes infer their modality-specific forward step from the same registry.
Pass `--step-func NAME` to override that selection. Common training,
sequence-length, parallelism, optimization, and checkpoint fields also have convenience flags such as
`-ms`/`--max_steps`, `-sl`/`--seq_length`, `-tp`/`--tensor_model_parallel_size`, and `--save_dir`. Use trailing
`KEY=VALUE` overrides for every other `ConfigContainer` field.

## Dataset selection

For library recipes, `--dataset` accepts source selectors and named dataset presets rather than internal dataset config class names.
Each name selects a `DatasetConfig` preset; the config type selects the existing runtime builder. Trailing
`dataset.*` overrides are applied directly after preset selection. Use typed fields for source, preprocessing, packing,
and loader settings; use `dataset.dataset_kwargs={...}` only for extra options consumed by a dataset implementation.
Benchmark recipes retain their recipe-owned dataset and reject `--dataset`.

| Value | Kind | Mode | Behavior |
|---|---|---|---|
| `mock` | Source selector | pretrain | In-memory generated GPT data |
| `megatron-indexed` | Source selector | pretrain | Local Megatron `.bin/.idx` data; never falls back to mock |
| `local-jsonl` | Source selector | sft/lora/dora | Local prompt-completion JSONL selected by `dataset.dataset_root` |
| `local-vlm` | Source selector | sft/lora/dora | Local VLM JSON/JSONL selected through `dataset.source` overrides |
| `squad` | Named preset | sft/lora/dora | Hugging Face SQuAD preset |
| `tulu3` | Named preset | sft/lora/dora | Ai2 Tulu 3 SFT mixture (`allenai/tulu-3-sft-mixture`) |
| `openmathinstruct2` | Named preset | sft/lora/dora | OpenMathInstruct-2 prompt/completion preset |
| `openmathinstruct2-thinking` | Named preset | sft/lora/dora | OpenMathInstruct-2 analysis/final channel format |
| `gsm8k` | Named preset | sft/lora/dora | GSM8K preset |
| `cord-v2` | Named preset | sft/lora/dora | CORD-v2 receipt VQA preset |
| `medpix` | Named preset | sft/lora/dora | MedPix medical VQA preset |
| `raven` | Named preset | sft/lora/dora | RAVEN visual reasoning preset |
| `rdr` | Named preset | sft/lora/dora | RDR visual reasoning preset |
| `llava-video-178k` | Named preset | sft/lora/dora | LLaVA-Video preset; requires `dataset.source.adapter_kwargs.video_root_path` |

### Megatron indexed data

Any pretraining corpus must first be converted to Megatron indexed data. Set one prefix or a list of prefixes directly
on the dataset config:

```bash
--dataset megatron-indexed dataset.data_path=/data/dclm/dclm_01_01_text_document
--dataset megatron-indexed 'dataset.data_path=[/data/dclm/part_1,/data/dclm/part_2]'
```

Every selected prefix must have matching `.bin` and `.idx` files. Use `dataset.path_to_cache=/shared/cache` to select
the index cache.

The launcher does not infer the source corpus from the files. For one preprocessing example, see
[the DCLM tutorial](../../tutorials/data/dclm/README.md).

### OpenMathInstruct-2

`openmathinstruct2` uses prompt/completion records. `openmathinstruct2-thinking` changes only the semantic output
format: chain-of-thought goes to the analysis channel and the answer goes to the final channel.

### Tulu

`tulu3` uses the canonical current Tulu SFT dataset, `allenai/tulu-3-sft-mixture`. The preset reads its published
`train` split and its native `messages` chat schema; it reserves 5% for validation by default. The mixture is licensed
under ODC-BY-1.0, but individual subsets can carry additional terms, including non-commercial restrictions. Review the
Hugging Face dataset card and its linked subset licenses before use.

### Offline packing

Offline packing is a text SFT option, not a dataset name. Set `dataset.enable_offline_packing=true` for `squad`, either
OpenMathInstruct-2 format, `tulu3`, `gsm8k`, or `local-jsonl`. The launcher aligns packed padding for the resolved CP/TP and
sequence-parallel configuration. Packed training requires `train.micro_batch_size=1`. The builder materializes packed
data automatically, so a separate packing Slurm job is not required. The selected recipe/model must support packed THD
sequences; for example, GLM-4.5 and Qwen3-Next recipes currently do not. On multiple nodes, keep the dataset cache on
shared mounted storage.

### VLM data

The hosted VLM names use the existing Hugging Face dataset adapters and retain the processor and in-batch packing
settings from the selected VLM recipe. `raven`, `rdr`, and `llava-video-178k` derive deterministic 95/5 train and
validation slices; `cord-v2` and `medpix` use their published validation splits. The runner selects the registered VLM
forward step from the recipe name; pass `--step-func` only to override it.

For local JSON/JSONL, select `local-vlm` and set
`dataset.source.load_kwargs.data_files.train=/path/to/train.jsonl`. Optional split overrides are
`dataset.validation_source.load_kwargs.data_files.validation` and
`dataset.test_source.load_kwargs.data_files.test`; enable those stages explicitly with `dataset.do_validation=true`
and `dataset.do_test=true`. Rows and media paths must already use the selected processor's supported conversation
schema. `dataset.hf_processor_path` overrides the processor inherited from the VLM recipe.

## SFT and LoRA checkpoints

Set `checkpoint.pretrained_checkpoint` to the pretrained native Megatron checkpoint or local Hugging Face model
directory:

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 8 \
    --account ACCOUNT --partition PARTITION --container-image IMAGE \
    --mount /checkpoints \
    --recipe gpt_oss_20b_sft_8gpu_h100_bf16_config \
    --mode sft --dataset openmathinstruct2-thinking \
    dataset.enable_offline_packing=true \
    train.micro_batch_size=1 \
    checkpoint.pretrained_checkpoint=/checkpoints/gpt-oss-20b
```

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 1 \
    --account ACCOUNT --partition PARTITION --container-image IMAGE \
    --mount /checkpoints \
    --recipe gpt_oss_20b_peft_1gpu_h100_bf16_config \
    --mode lora --dataset openmathinstruct2-thinking \
    dataset.enable_offline_packing=true \
    train.micro_batch_size=1 \
    checkpoint.pretrained_checkpoint=/checkpoints/gpt-oss-20b
```

## Overrides

Every selected recipe's `ConfigContainer` fields can be set with trailing `KEY=VALUE` arguments. Common fields also
accept the convenience flags listed by `run_recipe.py --help`. For example, batch sizes and training duration belong
under `train`, not `model`:

```bash
./scripts/training/train.sh \
    --nodes 1 --gpus-per-node 8 \
    --account ACCOUNT --partition PARTITION \
    --container-image /path/to/container.sqsh \
    --recipe llama32_1b_pretrain_config \
    --mode pretrain --dataset mock \
    -ms 10 -gb 8 -mb 1 \
    --lr 0.0002 --warmup_iters 1 \
    scheduler.lr_decay_iters=10
```

Precedence is recipe defaults, the selected dataset configuration, common convenience arguments, then trailing
`ConfigContainer` overrides. A trailing override therefore wins when it targets the same field as a convenience
argument.

Overrides take a benchmark recipe outside its canonical configuration. Use
`scripts/performance/setup_experiment.py` for specialized workflows that the unified launcher has not migrated yet.

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
The launcher adds no cluster-specific `srun` flags by default. Repeat `--srun-arg=ARG` for every flag required by the
target cluster. For example, a Pyxis/Enroot cluster may use:

```bash
--srun-arg=--mpi=pmix \
--srun-arg=--no-container-mount-home \
--srun-arg=--container-writable
```

The `=` form is required when `ARG` begins with `-`.

The compact launcher does not add rank-command prefixes or extra sbatch parameters such as `segment`. Configure those
through the target cluster integration, or use `scripts/performance/setup_experiment.py` when its compatibility NUMA
and segment policies are required.

The launcher submits the experiment in detached mode and returns after Slurm accepts the job. Inspect its state and
logs with the cluster's normal `squeue`, `sacct`, and log-file workflow.

Use `--dry-run` (or the explicit `--submission-dry-run` spelling) to render the NeMo-Run experiment without
submitting it. For a rank-local ConfigContainer dry run, invoke `run_recipe.py` directly:

```bash
uv run python scripts/training/run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --mode pretrain --dataset mock \
    --dry-run logger.save_config_filepath=/tmp/config.yaml
```

For a benchmark dry run, use the complete flat recipe name and omit `--dataset`. The rank-local dry run discovers the
recipe and validates the final config against the total GPU count encoded in that name without
requiring a live allocation; the submission dry run additionally validates `--nodes` and `--gpus-per-node`.

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

This entry point loads library recipes for pretraining, SFT, LoRA, and DoRA, and all exact exported
benchmark recipes when selected explicitly.
