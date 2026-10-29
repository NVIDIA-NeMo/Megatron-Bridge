# GLM-5 Family Examples

Examples for the GLM-5 family — [GLM-5](https://huggingface.co/zai-org/GLM-5) (`zai-org/GLM-5`), [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) (`zai-org/GLM-5.1`), [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) (`zai-org/GLM-5.2`), and [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (`zai-org/GLM-5.3`) — large sparse MoE models with Multi-Latent Attention (MLA) and Dynamic Sparse Attention (DSA).

All four checkpoints use the `GlmMoeDsaForCausalLM` architecture and are handled by `GLM5Bridge`. GLM-5 and GLM-5.1 have identical MoE / MLA / DSA dimensions, while GLM-5.2/5.3 share IndexShare-style DSA index reuse settings. GLM-5.3 uses an FP8 checkpoint and a different chat template; see the [compatibility guide](../../../../docs/models/glm/glm5-2.md#glm-52--glm-53-compatibility) for import/export requirements and validation limits. Full-model GLM-5.3 verification is pending; the GLM-5.2 card and recipes remain specific to GLM-5.2. GLM-5.3-Flash is a separate architecture.

| Property | Value |
|---|---|
| HF model IDs | `zai-org/GLM-5`, `zai-org/GLM-5.1`, `zai-org/GLM-5.2`, `zai-org/GLM-5.3` |
| Architecture | MoE + MLA + DSA (`GlmMoeDsaForCausalLM`) |
| Layers | 78 transformer (first 3 dense, rest MoE) |
| Routed experts | 256, top-8 per token |
| Shared experts | 1 per MoE layer |
| GLM-5.2/5.3 checkpoint parameters | ~753B, including the appended MTP layer |
| GLM-5.2/5.3 checkpoint precision | BF16 / block-scaled FP8, respectively; router biases remain FP32 |

**Requirements:** the repository's supported Transformers version, `fast-hadamard-transform` (CUDA extension, required by DSA)

## Hardware Requirements

The included GLM-5 round-trip conversion example uses **8 nodes (64 GPUs × 80 GB)**. See the [GLM-5.2 model verification card](../../../model_verification_cards/glm5-2/card.yaml) for the hardware and parallelism used by each verified GLM-5.2 workflow. Key constraints include:

- EP must divide 256 (number of routed experts). Valid: 1, 2, 4, 8, 16, 32, 64, 128, 256.
- TP does **not** reduce expert memory — increase EP instead.
- The conversion wrapper uses `TP=1, PP=2, EP=32` on 64 GPUs. PP splits the 78 transformer layers evenly, with 39 layers per stage, and EP places 8 routed experts per GPU.

### Pre-requisites

Install `fast-hadamard-transform` (required by the DSA attention variant) into the project venv from a GPU node:

```bash
pip install --target=.venv/lib/python3.12/site-packages --no-deps --no-build-isolation \
    git+https://github.com/Dao-AILab/fast-hadamard-transform.git
```

The PyPI source distribution is incomplete; install from the git repo.

## Inference (Megatron)

Use the verified `inference` item in the [GLM-5.2 model verification card](../../../model_verification_cards/glm5-2/card.yaml). The card is the canonical source for the pinned checkpoint revision, tested topology, `scripts/inference/infer.sh` command, and expected result.

The verified command selects the `legacy-full-prefix-generation` compatibility task. It recomputes the accumulated prefix at every decoding step because cached inference is not yet supported for AbsorbedMLA.

## Checkpoint Conversion (Round-Trip)

[slurm_conversion.sh](slurm_conversion.sh) uses `convert.sh roundtrip` to submit
HF → Megatron → HF validation and verify weight fidelity. Run it from a Slurm
login node; it waits for the job by default. Round-trip validation runs entirely
in memory and does not write another full checkpoint.

```bash
export CONTAINER_IMAGE=/path/to/container.sqsh
export SLURM_ACCOUNT=your_account
bash examples/models/glm/glm5/slurm_conversion.sh
```

The script uses 8 nodes (64 GPUs) with `TP=1`, `PP=2`, and `EP=32`.

> **Note:** The round-trip verification step (comparing ~63K weight tensors on rank 0)
> may hit shared-filesystem I/O contention at this model scale.

## Conversion Script Configuration

Set these environment variables before submitting the round-trip conversion wrapper:

| Variable | Description |
|---|---|
| `CONTAINER_IMAGE` | Path to Singularity/SquashFS container image |
| `SLURM_ACCOUNT` | Slurm account used for the submitted job |
| `SLURM_PARTITION` | Slurm partition; defaults to `batch` |
| `CONTAINER_MOUNTS` | Optional comma-separated bind mounts for shared storage; the current checkout is mounted automatically at `/opt/Megatron-Bridge` |
| `HF_HOME` | HuggingFace cache directory containing the downloaded `zai-org/GLM-5` model |
| `HF_TOKEN` | HuggingFace access token (for gated model access) |

Pass any cluster-specific `srun` flags after the wrapper, for example
`--srun-arg=--mpi=pmix`. The wrapper forwards them to `convert.sh`; no
NVIDIA-specific `srun` flags are enabled by default.

## Megatron-Core baseline

Use the repository's pinned Megatron-Core revision. It includes DSA dispatch and
the MLA `fuse_input_layernorm=False` specification; these no longer require
manual patches. Remaining production qualification is tracked in
[#5476](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/5476).
