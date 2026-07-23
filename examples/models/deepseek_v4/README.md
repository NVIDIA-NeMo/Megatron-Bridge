# DeepSeek V4

End-to-end conversion and inference scripts for the DeepSeek V4 family on Megatron Bridge.

The bridge supports four published variants out of the same code path. The on-disk quantisation differs between post-trained (Flash, Pro) and pretrained-only (Flash-Base, Pro-Base) models — see [`docs/models/deepseek/deepseek-v4.md`](../../../docs/models/deepseek/deepseek-v4.md) for the per-variant scheme.

## MCore Checkout

The pretraining recipes were tested with Megatron-LM `dev` commit `35f36c7c9dba` plus PR [#4839](https://github.com/NVIDIA/Megatron-LM/pull/4839) (`f04b762406f0` in the OCI test checkout). The Megatron-LM copy inside the current NeMo FW container is not expected to work for these recipes.

```bash
./scripts/switch_mcore.sh dev
uv sync
```

Use `./scripts/switch_mcore.sh main` and `uv sync --locked` to return to the pinned main-branch submodule.

The full-scale `deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config` performance
recipe preserves the stack validated by Megatron Bridge PR
[#4824](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/4824):
`nvcr.io/nvidia/nemo:26.06.01` with Megatron-LM dev commit
`9d46c924dce3818f2b5f894f7380712c780d1801` and the capability-check patch
documented in that PR. The Megatron-LM commit pinned by the current
Megatron Bridge `main` branch does not provide the required DeepSeek V4
performance features, so it is not a supported runtime for that recipe.

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
- `slurm_sft.sh` runs DeepSeek-V4-Flash full SFT end to end (import, then fine-tune) on Hopper or Blackwell, with MTP on or off.

Run `bash conversion.sh` after setting `WORKSPACE` and `MODEL_VARIANT`. See each script's header comments for the expected environment variables and `#SBATCH` directives to edit before submitting.

## Pretraining Recipes

See [`slurm_pretrain.sh`](slurm_pretrain.sh) for the Slurm launcher and [`deepseek_v4.py`](../../../src/megatron/bridge/recipes/deepseek/deepseek_v4.py) for recipe definitions.

Available Blackwell pretraining recipes:

- `deepseek_v4_flash_pretrain_mxfp8_config`: Adam MXFP8
- `deepseek_v4_flash_pretrain_muon_config`: Muon BF16
- `deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config`: 256-GPU GB300
  performance configuration (requires the PR #4824 container and dev-MCore
  stack described above)

`slurm_pretrain.sh` is a GB200 launcher with `TP=1,PP=4,EP=8,CP=1` by default. Indexer loss are disabled for now and is planned for a follow-up.

Before submitting, set `CONTAINER_IMAGE`. For DCLM, also set `DCLM_DATA_DIR` and `DCLM_CACHE`. Use `CONTAINER_MOUNTS` and `EXTRA_PYTHONPATH` for cluster-specific data, checkouts, and Python dependencies.

The bridge's `maybe_modify_loaded_hf_weight` hook dispatches dequantisation by tensor dtype:

- `int8` -> MXFP4 packed nibbles -> `bfloat16` via the E2M1 lookup table and per-row 16-K-tile E8M0 scales
- `float8_e4m3fn` with companion `.scale` -> `bfloat16` via 128x128 block-scale expansion, handling both E8M0 and F32 scale dtypes

No external dequantisation script is required.

## SFT Recipes

Full-parameter SFT of DeepSeek-V4-Flash. See [`slurm_sft.sh`](slurm_sft.sh) for the end-to-end launcher and [`deepseek_v4.py`](../../../src/megatron/bridge/recipes/deepseek/deepseek_v4.py) for the recipe definitions.

| Recipe | MTP | mHC kernel | Hardware |
|--------|-----|------------|----------|
| `deepseek_v4_flash_sft_config` | on | fused (cuTile, sm_100) | Blackwell (Hopper: set `use_fused_mhc=False`) |
| `deepseek_v4_flash_no_mtp_sft_config` | off | fused (cuTile, sm_100) | Blackwell (Hopper: set `use_fused_mhc=False`) |

Both recipes enable fused mHC and fused rope (`use_fused_mhc=True`, `apply_rope_fusion=True`), matching the pretrain recipes. The historical "fused-kernel SFT NaN" reports are both resolved: the fused-mHC report was a confound, and the fused-rope NaN was a bridge config-mapping bug (`partial_rotary_factor` double-applied) fixed by `rotary_percent=1.0` in #4271 — with that fix, full-model (43-layer, real weights) SFT with rope fusion on 8×GB300 matches the unfused control's loss trajectory. The fused mHC cuTile kernel is **sm_100 (Blackwell)**; on Hopper set `use_fused_mhc=False`. The `no_mtp` variant drops the MTP layer and trims `csa_compress_ratios` back to `num_layers` (the bridge appends an MTP-layer ratio that `transformer_config` would otherwise reject).

> There are intentionally **no MXFP8 or Muon SFT recipes**: both were prototyped (mirroring the pretrain recipes) but fail in full-model DSv4-Flash SFT — see Blockers. SFT ships **Adam/bf16**; the pretrain MXFP8/Muon recipes are unaffected.

`slurm_sft.sh` selects the recipe from `MTP` (`on`|`off`); `HARDWARE` (`blackwell`|`hopper`) sets the node topology (on Hopper, also set `use_fused_mhc=False` — the fused mHC kernel is Blackwell-only). It runs the model at `TP=1, PP=4, EP=8` (32 GPUs), and:

1. Imports `deepseek-ai/DeepSeek-V4-Flash` into a Megatron checkpoint (skipped if already present).
2. Runs full SFT from that checkpoint via `scripts/training/run_recipe.py`.

GPUs per node differ by hardware, so 32 GPUs means a different node count:

| Hardware | GPUs/node | Nodes for 32 GPUs | `#SBATCH` |
|----------|-----------|-------------------|-----------|
| GB200 NVL | 4 | 8 | `--nodes=8 --gpus-per-node=4` (default) |
| H100/H200 | 8 | 4 | `--nodes=4 --gpus-per-node=8` |

**Sequences are unpacked (SBHD).** The CSA/DSA indexer asserts `packed_seq_params is None` (`csa.py`), so packed/THD sequences are not yet supported on the sparse layers. The recipes ship an unpacked SQuAD config; do not set `dataset.enable_offline_packing=true`. Select another built-in source with `--dataset gsm8k` or use `--dataset local-jsonl dataset.dataset_root=<path>` for JSONL data — both stay unpacked by default.

**Eval sizing.** Each evaluation draws `validation.eval_iters × global_batch_size` samples; if that exceeds your validation/test split the eval hangs trying to form a batch. `slurm_sft.sh` defaults to a small `EVAL_ITERS=2` and `DO_TEST=false` (the end-of-run test eval is the usual culprit on small test sets) — raise them only when your splits are large enough.

## SFT Status, TODO & Blockers

Validated end to end on **8× GB300 (32 GPU, TP1/PP4/EP8)** with real DeepSeek-V4-Flash weights: HF→Megatron import (FP8/MXFP4→bf16) + full SFT, **MTP on and off**, `lm loss` decreasing with no NaN — at SBHD / bf16 / Adam / 4K. (Those end-to-end runs used `use_fused_mhc=False`; the recipes now default `use_fused_mhc=True`, verified clean by an isolated GB300 re-run and confirmed by a reviewer on GB200.) The items below are **not** implemented and are gated on upstream Megatron-Core (verified against the code and PRs as of 2026-06-02; tracking: [NVIDIA/Megatron-LM#4468](https://github.com/NVIDIA/Megatron-LM/issues/4468)):

| Capability | Status | Gating | Notes |
|------------|--------|--------|-------|
| Packed sequence (**THD**) for DSv4 attention | **Experimental** (unmerged) | mcore PR #5011 (open) | THD *is implemented* in #5011 — it removes the `packed_seq_params is None` asserts in `csa.py`/`dsa.py` and adds the THD index/topk + `cu_seqlens` path. It **grafts cleanly onto our pinned mcore** (0-conflict 3-way merge of #5011) and needs `nvidia-cudnn-frontend[cutedsl]>=1.24.0` installed **`--no-deps`** for the DSA THD kernels (the stock 26.04 container ships an older FE; a full install shadows the container CUDA and breaks TE). The shipped/supported path stays **SBHD**, but an experimental THD (packed-seq) SFT is now **self-validated on 8×GB300** (10-iter, `lm loss` 4.46→1.48, `mtp_1` 13.5→2.70, no NaN) — ahead of #5011 merging. Merged dispatcher-THD #4816 is MoE-side only and does not lift the attention block. |
| CUDA Graphs for DSv4 THD | TODO | (no PR) | Follows THD. |
| Context parallel / long-context (≥64K) | TODO | draft PR #5087 (depends on #5011) | SFT runs CP=1; this is the Phase-3 long-context target. |
| MXFP8 / Muon **SFT** | **Fails (upstream); no SFT recipe shipped** | fp8 numerics; Muon + expert-parallel | Both prototyped (mirroring the pretrain recipes) and tested at full scale on 8×GB300 with `use_fused_mhc=False`: **MXFP8 NaNs at iter-2** (fp8 × hash-MoE/ClampedSwiGLU numerics) and **Muon hits an iter-2 `AssertionError`** (Muon + EP-MoE grad bookkeeping not yet supported upstream). Removed from the shipped recipe set; the **pretrain** MXFP8/Muon recipes remain. SFT ships **Adam/bf16**. |
| Full SFT on **H100-80GB** | needs **≥64 GPU** | hardware | At 32 GPU (TP1/PP4/EP8 ⇒ DP=1) the fp32 master-param buffer can't shard and OOMs. Use **≥64 H100** (PP8 ⇒ DP≥2), or H200-140GB / Blackwell, or PEFT/LoRA. |

Already incorporated in the pinned mcore (no action): dense-loss + per-layer rope-type fix (#5018), CSA/HCA (#4458), Hash MoE/ClampedSwiGLU (#4481), MTP+mHC (#4518), fusion kernels (#4894).

## Container Image

Run inside a container that has the DSv4 prerequisites: Megatron-Bridge on a **`main2dev`** Megatron-LM commit (validated on `ed6b1f65502aec7f2fe27e14a1245c29e435c2a6`; has both `safe_get_world_size` and the DSv4 `csa.py`/`dsa.py`), the DSA dependency `fast_hadamard_transform`, and a pre-built `helpers_cpp`. The [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) — or an image built from `docker/Dockerfile.ci` with the submodule checked out to a `main2dev` commit — provides this. Set `slurm_sft.sh`'s `CONTAINER_IMAGE` to the resulting `.sqsh`.

The first import downloads ~285B of HF weights, so point `HF_HOME` at shared scratch (e.g. `/home/scratch.<user>/HF_HOME`) before submitting so the download persists across jobs, and set `HF_TOKEN` if the repo is gated.

### Disk footprint

The full bf16 model is ~570 GB, so plan storage before submitting:

| Artifact | Size | Notes |
|----------|------|-------|
| HF download cache (`HF_HOME`) | ~150–200 GB | quantized (FP8/MXFP4) |
| Imported Megatron checkpoint (loaded by SFT) | ~570 GB | bf16, `${MEGATRON_DIR}/iter_0000000` |
| Each saved SFT checkpoint | ~570 GB | bf16 model only |

`slurm_sft.sh` does **not** save checkpoints by default (`SAVE_CKPT=0`) — to get training running you only need the base weights *loaded*, not written back — so the whole run fits in ~750 GB (cache + base). Never save optimizer state: distributed Adam for a 285B model is multi-TB (`save_optim=false` is fixed in the script). For a real fine-tune, set `SAVE_CKPT=1` (saving is capped to `KEEP_CKPTS=1` latest checkpoint) and put `HF_HOME` + `MEGATRON_DIR` on shared/project storage so a small (e.g. 2 TB) personal scratch only holds the output.

## Parallelism Configurations

DSv4 currently requires **TP=1** because MLA tensor parallelism is not supported alongside the DSv4 hybrid attention path. Scale via expert and pipeline parallelism instead.

| Model | TP | PP | EP | GPUs | GPU | Verified |
|-------|---:|---:|---:|-----:|-----|----------|
| DeepSeek-V4-Flash | 1 | 1 | 4 | 4 | GB200 192GB | Import, export, inference |
| DeepSeek-V4-Flash | 1 | 1 | 16 | 16 | H100 80GB | Import, export, inference |
| DeepSeek-V4-Flash-Base | 1 | 1 | 4 | 4 | GB200 192GB | Import, export, inference |
| DeepSeek-V4-Pro | 1 | 4 | 8 | 32 | GB200 192GB | Import, export, inference |
| DeepSeek-V4-Pro | 1 | 16 | 8 | 128 | H100 80GB | Import, export, inference |

## Known Limitations

- **MTP is disabled for inference** via `disable_mtp_for_inference()`. MTP weights are mapped end-to-end and loaded into the Megatron model.

- **Fused mHC: use the unfused path for SFT.** The fused cuTile mHC kernel needs sm_100 (not H100) and is on by default in the bridge config; it works for import/inference on Blackwell, but **NaNs in SFT training** (see SFT Blockers). The SFT recipes therefore force `use_fused_mhc=False` — the validated unfused/reference path, which also runs on Hopper.

- **`fast_hadamard_transform` is required by the DSA attention variant.** `csa.py` and `dsa.py` import `hadamard_transform` from this package and hard-assert availability — there is no in-tree PyTorch fallback. Install from the Dao-AILab git repo (the PyPI source distribution is incomplete; see the sibling GLM-5 [README](../glm/glm5/README.md#pre-requisites) for the same dependency):

  ```bash
  pip install --no-build-isolation \
      git+https://github.com/Dao-AILab/fast-hadamard-transform.git
  ```

- **Logit parity is verified for Flash and Flash-Base** against the official inference stack at last-real-token logits. The remaining gap is structural, from different attention/HC kernel decompositions and accumulation precisions between MCore and official inference.
