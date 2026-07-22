# K-EXAONE MoE Examples

Scripts for [K-EXAONE-236B-A23B](https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B), a large BF16 sparse MoE language model from LG AI Research.

| Property | Value |
|---|---|
| HF model ID | `LGAI-EXAONE/K-EXAONE-236B-A23B` |
| Architecture | Sparse MoE decoder-only LM (`ExaoneMoeForCausalLM`) |
| Routed experts | 128 |
| Active experts | 8 per token |
| Total params | 236B |
| Active params | 23B |
| Default single-node parallelism | `TP=1, PP=1, EP=8, ETP=1` |
| Default Slurm parallelism | `TP=1, PP=1, EP=16, ETP=1` |

**Requirements:** use `--trust-remote-code` when loading the Hugging Face checkpoint.

## Hardware Requirements

K-EXAONE-236B-A23B is memory intensive in BF16. The single-node scripts default to 8 GPUs with `EP=8`; H200-class GPUs are recommended. Inference or round-trip conversion on 8x H100 80 GB may OOM because parameters, workspaces, and KV cache must fit together.

Key constraints:

- `TP * PP * EP` must equal the number of distributed ranks.
- `EP` must divide 128, the number of routed experts.
- Increase `EP` to reduce expert-parameter memory per rank.

## Inference (Single Node)

[inference.sh](inference.sh) loads the Hugging Face checkpoint, converts it to Megatron in memory, and runs greedy text generation across 8 local GPUs.

```bash
examples/models/exaone/exaone_moe/inference.sh
```

Override generation or parallelism settings with environment variables:

```bash
PROMPT="대한민국의 수도는 어디인가요?" \
MAX_NEW_TOKENS=64 \
NPROC_PER_NODE=8 \
TP=1 PP=1 EP=8 ETP=1 \
examples/models/exaone/exaone_moe/inference.sh
```

## Checkpoint Conversion (Single Node)

[conversion.sh](conversion.sh) runs HF -> Megatron -> HF round-trip conversion across 8 local GPUs. The default `EP=8` assigns 16 of the model's 128 routed experts to each GPU.

```bash
examples/models/exaone/exaone_moe/conversion.sh
```

Set `OUTPUT_DIR` to save the exported Hugging Face checkpoint:

```bash
OUTPUT_DIR=/workspace/models/K-EXAONE-236B-A23B-hf-export \
examples/models/exaone/exaone_moe/conversion.sh
```

## Slurm Inference

[slurm_inference.sh](slurm_inference.sh) runs generation across 16 GPUs on 2 nodes with default `TP=1, PP=1, EP=16`.

```bash
mkdir -p logs
sbatch examples/models/exaone/exaone_moe/slurm_inference.sh
```

Required environment variables:

| Variable | Description |
|---|---|
| `CONTAINER_IMAGE` | Enroot/SquashFS container image path |
| `CONTAINER_MOUNTS` | Optional comma-separated container mounts |
| `WORKDIR` | Repository path inside the container; defaults to `/opt/Megatron-Bridge` |
| `HF_TOKEN` | Hugging Face access token, if required |
| `HF_HOME` | Shared Hugging Face cache directory |
| `UV_CACHE_DIR` | Shared uv cache directory |

The Slurm scripts warm the shared `uv` cache once before all distributed ranks enter the main job.

## Slurm Checkpoint Conversion

[slurm_conversion.sh](slurm_conversion.sh) uses `convert.sh roundtrip` to submit a fixed `TP=1, PP=1, EP=16` config and verify HF ↔ Megatron round-trip conversion across 16 GPUs on 2 nodes. Run the wrapper from a Slurm login node; it submits one job and waits for it by default.

Set the container and account, then launch:

```bash
export CONTAINER_IMAGE=/path/to/container.sqsh
export SLURM_ACCOUNT=<your-account>
export SLURM_PARTITION=batch
# Optional: export CONTAINER_MOUNTS, HF_TOKEN, HF_HOME, and UV_CACHE_DIR before launching.
bash examples/models/exaone/exaone_moe/slurm_conversion.sh
```

The current checkout is mounted automatically at `/opt/Megatron-Bridge` and must be on storage visible from the compute nodes. Forward any cluster-specific `srun` options your scheduler requires, for example:

```bash
bash examples/models/exaone/exaone_moe/slurm_conversion.sh \
    --srun-arg=--mpi=pmix
```

## Script Configuration

| Variable | Description |
|---|---|
| `HF_MODEL_ID` | Hugging Face model ID; defaults to `LGAI-EXAONE/K-EXAONE-236B-A23B` |
| `NPROC_PER_NODE` | Local GPU/rank count for single-node scripts; defaults to 8 |
| `TP` | Tensor parallelism |
| `PP` | Pipeline parallelism |
| `EP` | Expert parallelism |
| `ETP` | Expert tensor parallelism |
| `PROMPT` | Inference prompt |
| `MAX_NEW_TOKENS` | Number of generated tokens |
| `OUTPUT_DIR` | Optional export path for single-node round-trip conversion |

## Validation

Focused EXAONE conversion coverage lives in `tests/functional_tests/test_groups/models/exaone/test_exaone_conversion.py`.