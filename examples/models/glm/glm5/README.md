# GLM-5 Family Examples

Examples for the GLM-5 family — [GLM-5](https://huggingface.co/zai-org/GLM-5) (`zai-org/GLM-5`), [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) (`zai-org/GLM-5.1`), and [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) (`zai-org/GLM-5.2`) — large sparse MoE models with Multi-Latent Attention (MLA) and Dynamic Sparse Attention (DSA).

All three checkpoints use the `GlmMoeDsaForCausalLM` architecture and are handled by `GLM5Bridge`. GLM-5 and GLM-5.1 have identical MoE / MLA / DSA dimensions, while GLM-5.2 adds IndexShare-style DSA index reuse settings.

| Property | Value |
|---|---|
| HF model IDs | `zai-org/GLM-5`, `zai-org/GLM-5.1` |
| Architecture | MoE + MLA + DSA (`GlmMoeDsaForCausalLM`) |
| Layers | 78 transformer (first 3 dense, rest MoE) |
| Routed experts | 256, top-8 per token |
| Shared experts | 1 per MoE layer |
| Total params | ~800B+ (BF16) |
| Active params | ~60B per token |

**Requirements:** `transformers >= 5.2.0`, `fast-hadamard-transform` (CUDA extension, required by DSA)

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

## MCore Patches Required

The DSA attention variant requires two patches to `megatron/core/models/gpt/experimental_attention_variant_module_specs.py`:

1. **DSA dispatch:** Add `elif config.experimental_attention_variant == "dsa"` to `get_experimental_attention_variant_module_spec` to call `get_dsa_module_spec_for_backend`.
2. **MLA metainfo:** Add `metainfo={"fuse_input_layernorm": False}` to the `MLASelfAttention` `ModuleSpec` in `get_dsa_module_spec_for_backend`.
