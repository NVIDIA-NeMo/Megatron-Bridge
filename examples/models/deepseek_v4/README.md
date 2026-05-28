# DeepSeek V4

End-to-end conversion and inference scripts for the DeepSeek V4 family on Megatron Bridge.

The bridge supports four published variants out of the same code path. The on-disk quantisation differs between post-trained (Flash, Pro) and pretrained-only (Flash-Base, Pro-Base) models — see [`docs/models/deepseek/deepseek-v4.md`](../../../docs/models/deepseek/deepseek-v4.md) for the per-variant scheme.

## MCore Dev Branch Requirement

DSv4 imports require MCore changes that are not yet on a tagged release: PR [#3430](https://github.com/NVIDIA/Megatron-LM/pull/3430), PR [#4458](https://github.com/NVIDIA/Megatron-LM/pull/4458), PR [#4481](https://github.com/NVIDIA/Megatron-LM/pull/4481), and PR [#4518](https://github.com/NVIDIA/Megatron-LM/pull/4518), and PR [#4839](https://github.com/NVIDIA/Megatron-LM/pull/4839). Until these merge to Megatron-LM `main` and the bridge submodule pin advances, point `3rdparty/Megatron-LM` at the Megatron-LM `dev` branch:

```bash
./scripts/switch_mcore.sh dev
uv sync
```

Use `./scripts/switch_mcore.sh main` and `uv sync --locked` to return to the pinned main-branch submodule.

| Variant | HF path | Quant scheme | Validation |
|---------|---------|--------------|------------|
| DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash` | FP8 attn + MXFP4 experts | Verified on GB200, last-token logit cosine 0.96-0.99 (short prompts ~0.98, long prompts >1024 tokens ~0.96-0.99) vs official inference |
| DeepSeek-V4-Flash-Base | `deepseek-ai/DeepSeek-V4-Flash-Base` | uniform FP8 (F32 scales) | Verified on GB200, last-real-token logit cosine 0.9866-0.9930, mean 0.9907 vs official inference |
| DeepSeek-V4-Pro | `deepseek-ai/DeepSeek-V4-Pro` | FP8 attn + MXFP4 experts | Import, export, inference verified on GB200 (PP=4 EP=8) and H100 (PP=16 EP=8) |
| DeepSeek-V4-Pro-Base | `deepseek-ai/DeepSeek-V4-Pro-Base` | uniform FP8 (F32 scales) | Same bridge code as Pro; end-to-end untested |

## Examples

- `conversion.sh` imports HF weights into Megatron Bridge and exports Megatron checkpoints back to HF format.
- `inference.sh` runs text generation against an HF or Megatron checkpoint.
- `slurm_pretrain.sh` runs the DeepSeek-V4-Flash pretraining recipes.

Run `bash conversion.sh` after setting `WORKSPACE` and `MODEL_VARIANT`. See each script's header comments for the expected environment variables and `#SBATCH` directives to edit before submitting.

## Pretraining Recipes

See [`slurm_pretrain.sh`](slurm_pretrain.sh) for the Slurm launcher and [`deepseek_v4.py`](../../../src/megatron/bridge/recipes/deepseek/deepseek_v4.py) for recipe definitions.

Available pretraining recipes:

- `deepseek_v4_flash_pretrain_mxfp8_config`: Adam MXFP8
- `deepseek_v4_flash_pretrain_muon_config`: Muon BF16

Before submitting, set `CONTAINER_IMAGE`. For DCLM, also set `DCLM_DATA_DIR` and `DCLM_CACHE`. Use `CONTAINER_MOUNTS` and `EXTRA_PYTHONPATH` for cluster-specific data, checkouts, and Python dependencies.

The bridge's `maybe_modify_loaded_hf_weight` hook dispatches dequantisation by tensor dtype:

- `int8` -> MXFP4 packed nibbles -> `bfloat16` via the E2M1 lookup table and per-row 16-K-tile E8M0 scales
- `float8_e4m3fn` with companion `.scale` -> `bfloat16` via 128x128 block-scale expansion, handling both E8M0 and F32 scale dtypes

No external dequantisation script is required.

## Parallelism Configurations

DSv4 currently requires **TP=1** because MLA tensor parallelism is not supported alongside the DSv4 hybrid attention path. Scale via expert and pipeline parallelism instead.

| Model | TP | PP | EP | GPUs | GPU | Verified |
|-------|---:|---:|---:|-----:|-----|----------|
| DeepSeek-V4-Flash | 1 | 1 | 4 | 4 | GB200 192GB | Import, export, inference |
| DeepSeek-V4-Flash | 1 | 1 | 8 | 8 | H100 80GB | Import, export, inference |
| DeepSeek-V4-Flash-Base | 1 | 1 | 4 | 4 | GB200 192GB | Import, export, inference |
| DeepSeek-V4-Pro | 1 | 4 | 8 | 32 | GB200 192GB | Import, export, inference |
| DeepSeek-V4-Pro | 1 | 16 | 8 | 128 | H100 80GB | Import, export, inference |

## Known Limitations

- **MTP is disabled for inference** via `disable_mtp_for_inference()`. MTP weights are mapped end-to-end and loaded into the Megatron model.

- **Fused mHC is not supported on H100.** Set `use_fused_mhc=False` in the bridge config when running on Hopper GPUs. Fused mHC is enabled by default and works on GB200.

- **`fast_hadamard_transform` is required by the DSA attention variant.** `csa.py` and `dsa.py` import `hadamard_transform` from this package and hard-assert availability — there is no in-tree PyTorch fallback. Install from the Dao-AILab git repo (the PyPI source distribution is incomplete; see the sibling GLM-5 [README](../glm/glm5/README.md#pre-requisites) for the same dependency):

  ```bash
  pip install --no-build-isolation \
      git+https://github.com/Dao-AILab/fast-hadamard-transform.git
  ```

- **Logit parity is verified for Flash and Flash-Base** against the official inference stack at last-real-token logits. The remaining gap is structural, from different attention/HC kernel decompositions and accumulation precisions between MCore and official inference.
