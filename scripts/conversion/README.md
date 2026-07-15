# Checkpoint conversion launcher

`convert.sh` is the supported command-line entry point for Hugging Face ↔
Megatron checkpoint conversion. It uses NeMo Run for both local execution and
Slurm submission and selects one of two conversion backends:

- `--device cpu`: one process on one node, with model construction and export
  on CPU;
- `--device gpu`: distributed conversion with one process per GPU and TP, PP,
  EP, and ETP support.

Run `./scripts/conversion/convert.sh import --help`,
`./scripts/conversion/convert.sh export --help`, or
`./scripts/conversion/convert.sh roundtrip --help` for the complete CLI.

## Local CPU conversion

Local execution uses the current Megatron Bridge environment and waits for the
conversion to finish.

```bash
./scripts/conversion/convert.sh import \
  --executor local \
  --device cpu \
  --hf-model meta-llama/Llama-3.2-1B \
  --megatron-path /workspace/models/llama32-1b

./scripts/conversion/convert.sh export \
  --executor local \
  --device cpu \
  --hf-model meta-llama/Llama-3.2-1B \
  --megatron-path /workspace/models/llama32-1b/iter_0000000 \
  --hf-path /workspace/models/llama32-1b-hf
```

CPU conversion intentionally supports one process on one node. Allocate more
host memory and CPUs for large checkpoints rather than increasing `--nodes`.

## Distributed GPU conversion on Slurm

The GPU backend uses NeMo Run's torchrun launcher for local execution and
srun-native tasks for Slurm. Users should not wrap the command in `torchrun`,
`srun`, or `sbatch`.

```bash
export HF_TOKEN="$(<${HOME}/HF_TOKEN)"

./scripts/conversion/convert.sh import \
  --executor slurm \
  --device gpu \
  --nodes 1 \
  --gpus-per-node 8 \
  --account ACCOUNT \
  --partition PARTITION \
  --time 01:00:00 \
  --container-image /path/to/megatron-bridge.sqsh \
  --mount /workspace \
  --mount /path/to/Megatron-Bridge:/opt/Megatron-Bridge \
  --env HF_TOKEN \
  --hf-model Qwen/Qwen3-30B-A3B \
  --megatron-path /workspace/models/qwen3-30b-a3b \
  --tp 1 --pp 1 --ep 8 --etp 1
```

For export, add `--hf-path`. Distributed Hugging Face saving is enabled by
default for the GPU backend to avoid gathering the full model on rank zero.
Use `--no-distributed-save` only when the model fits comfortably on rank zero.

No cluster-specific `srun` flags are added by default. If the target cluster
requires extra flags, repeat `--srun-arg=ARG`. For example, a Pyxis/Enroot
cluster may use:

```bash
--srun-arg=--mpi=pmix \
--srun-arg=--no-container-mount-home \
--srun-arg=--container-writable
```

The `=` form is required when `ARG` begins with `-`.

## Distributed round-trip validation

The `roundtrip` command runs
`examples/conversion/hf_megatron_roundtrip_multi_gpu.py` through the same
NeMo Run local or Slurm executor. It compares weights after the Hugging Face →
Megatron → Hugging Face conversion and requires the GPU backend.

```bash
./scripts/conversion/convert.sh roundtrip \
  --executor local \
  --device gpu \
  --gpus-per-node 8 \
  --hf-model-id Qwen/Qwen3-30B-A3B \
  --megatron-load-path /workspace/models/qwen3-30b-a3b/iter_0000000 \
  --tp 1 --pp 1 --ep 8 --etp 1 \
  --trust-remote-code \
  --skip-save
```

`--hf-model` and `--hf-model-id` are aliases, as are `--megatron-path` and
`--megatron-load-path`. Omit the Megatron load path to validate a direct
in-memory conversion from the Hugging Face model. Omit `--skip-save` to write
the round-tripped Hugging Face model, optionally under `--output-dir`, and use
`--megatron-save-path` to also save a Megatron checkpoint.

## CPU conversion on Slurm

CPU mode submits one task and does not request GPUs or GRES:

```bash
./scripts/conversion/convert.sh import \
  --executor slurm \
  --device cpu \
  --nodes 1 \
  --cpus-per-task 64 \
  --mem 1T \
  --account ACCOUNT \
  --partition CPU_PARTITION \
  --container-image /path/to/megatron-bridge.sqsh \
  --mount /workspace \
  --mount /path/to/Megatron-Bridge:/opt/Megatron-Bridge \
  --env HF_TOKEN \
  --hf-model meta-llama/Llama-3.2-1B \
  --megatron-path /workspace/models/llama32-1b
```

`--env` accepts names only. Export values in the launcher environment so
secrets are inherited by Slurm without being materialized in generated job
scripts. Use `--submission-dry-run` to inspect a rendered job and `--detach`
when a Slurm command should return immediately after submission. Local execution
always waits so worker failures propagate to the launcher.
