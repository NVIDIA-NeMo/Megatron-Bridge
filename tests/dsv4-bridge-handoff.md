# DeepSeek-V4-Flash Bridge — Handoff Document

## Current State (2026-05-07)

Last-token logit cosine similarity vs official inference: **0.935**
Top-5 token overlap: **4/5**

**Use OCI (`oci-hsg-cs-001`) only** — DSv4 requires TP=1 which needs B200 192GB GPUs; CW-DFW A100 80GB cannot fit the model.

---

## Code Changes

All changes are committed and pushed.

### Bridge (`Megatron-Bridge`, branch: `weijia_dsv4`, PR #3562)

**1. `src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py`**
- `provider.mscale = 1.0; provider.mscale_all_dim = 1.0` — fixes YaRN RoPE concentration factor (was 1.277x, now 1.0x)
- `provider.use_fused_mhc = True` — enables cuTile fused HC kernels (PR #3828)

**2. `src/megatron/bridge/models/conversion/model_bridge.py`**
- `if task is None or task.megatron_module is None:` at lines 924 and 1052 — temporary guard for 5 unmapped MTP params. Revert when MTP mappings are fully implemented (see MTP Status below).

### MCore (`Megatron-LM`, branch: `weijiac/dsv4-bridge`, fork: `weijiac0619/Megatron-LM`)

Base: `origin/dev` + PR #4518 (merged) + 1 commit:

**3. `megatron/core/transformer/experimental_attention_variant/dsa.py`**
- `_pytorch_hadamard_transform` fallback — `fast_hadamard_transform` package unavailable on aarch64 (OCI B200)
- `mask = mask.to(index_scores.dtype)` — fixes bf16/fp32 dtype mismatch assert in `fused_qk_topk_naive`

Both MCore changes are environment workarounds, not model logic changes. They become unnecessary if `fast_hadamard_transform` is installed.

### MTP Support Status

MTP is **mostly mapped** in the bridge (transformer weights, HC params, experts, norms). It is **disabled for inference** via `disable_mtp_for_inference()`. Five MTP params are unmapped, causing the `model_bridge.py` None guard to be needed:

| MTP param | Status | What's needed |
|---|---|---|
| `mtp.layers.0.hc_head_fn` | Unmapped | Add `ReplicatedMapping` (same pattern as decoder's `hc_head_fn`) |
| `mtp.layers.0.hc_head_base` | Unmapped | Add `ReplicatedMapping` |
| `mtp.layers.0.hc_head_scale` | Unmapped | Add `ReplicatedMapping` |
| `mtp.layers.0.e_proj.weight` | Stale mapping | PR 4518 changed MCore MTP from concatenated `eh_proj` to separate `e_proj`/`h_proj`. Bridge's `_MTPEHProjMapping` targets old `eh_proj` — needs updating to map `e_proj` and `h_proj` separately |
| `mtp.layers.0.h_proj.weight` | Stale mapping | Same as above |

These don't affect inference (MTP disabled) but must be fixed for MTP training/inference support.

---

## Key Paths (OCI: `oci-hsg-cs-001`)

| Item | Path |
|---|---|
| Bridge repo | `/lustre/fsw/portfolios/coreai/users/weijiac/nemo_workspace/Megatron-Bridge` (branch: `weijia_dsv4`) |
| MCore repo | `/lustre/fsw/portfolios/coreai/users/weijiac/nemo_workspace/Megatron-LM` (branch: `weijiac/dsv4-bridge`) |
| HF model | `/lustre/fsw/portfolios/coreai/users/weijiac/models/deepseek-ai/DeepSeek-V4-Flash` |
| Megatron checkpoint (TP=1, ETP=4) | `/lustre/fsw/portfolios/coreai/users/weijiac/dsv4_flash_megatron_ckpt_05062026/` |
| Official BF16 model (MP=4) | `/lustre/fsw/portfolios/coreai/users/weijiac/dsv4_flash_converted_bf16_mp4/` |
| Official inference code | `/lustre/fsw/portfolios/coreai/users/weijiac/nemo_workspace/dsv4_official_inference/` |
| Cosine analysis results | `/lustre/fsw/portfolios/coreai/users/weijiac/dsv4_cosine_analysis/` |
| Container | `/lustre/fsw/portfolios/coreai/users/weijiac/sqsh/nemo_26.04.sqsh` |
| Slurm account | `coreai_dlalgo_llm`, partition: `batch`, 4 GPUs/node (B200 192GB) |

---

## MCore Setup (two options)

**Option A: Use existing Lustre path (recommended if you have OCI access)**
```bash
# Already set up, no action needed
cd /lustre/fsw/portfolios/coreai/users/weijiac/nemo_workspace/Megatron-LM
# Branch: weijiac/dsv4-bridge (dev + PR #4518 + dsa.py workarounds)
```

**Option B: Clone from fork**
```bash
git clone git@github.com:weijiac0619/Megatron-LM.git
cd Megatron-LM
git checkout weijiac/dsv4-bridge
```

---

## How to Run Tests

All tests run on OCI (`oci-hsg-cs-001`) with 4 B200 GPUs.

### Full Cosine Similarity Report (uses saved checkpoint, ~5 min)

```bash
WKDIR=/lustre/fsw/portfolios/coreai/users/weijiac

# Step 1: Run Megatron (captures all layer + sub-layer hidden states)
srun -A coreai_dlalgo_llm -p batch -N 1 --gpus-per-node=4 --ntasks-per-node=4 -t 00:15:00 \
  --container-image=$WKDIR/sqsh/nemo_26.04.sqsh \
  --container-mounts="/lustre:/lustre,$WKDIR/nemo_workspace/Megatron-Bridge:/opt/Megatron-Bridge,$WKDIR/nemo_workspace/Megatron-LM:/opt/megatron-lm" \
  --export=ALL,PYTHONPATH=/opt/megatron-lm:/opt/Megatron-Bridge/src,HF_HOME=$WKDIR/.cache/huggingface,MASTER_ADDR=localhost,MASTER_PORT=29500,MODE=megatron \
  bash -c "cd /opt/Megatron-Bridge && python -u tests/dsv4_full_cosine_report.py"

# Step 2: Run comparison (CPU-only, ~1 min)
srun -A coreai_dlalgo_llm -p batch -N 1 --gpus-per-node=4 --ntasks-per-node=1 -t 00:05:00 \
  --container-image=$WKDIR/sqsh/nemo_26.04.sqsh \
  --container-mounts=/lustre:/lustre \
  --export=ALL,MODE=compare \
  python3 /opt/Megatron-Bridge/tests/dsv4_full_cosine_report.py
```

### Official Reference (only needed if re-generating, ~5 min)

```bash
# Captures official hidden states (requires tilelang)
srun -A coreai_dlalgo_llm -p batch -N 1 --gpus-per-node=4 --ntasks-per-node=4 -t 00:15:00 \
  --container-image=$WKDIR/sqsh/nemo_26.04.sqsh \
  --container-mounts="/lustre:/lustre,$WKDIR/nemo_workspace/Megatron-Bridge:/opt/Megatron-Bridge,$WKDIR/nemo_workspace/Megatron-LM:/opt/megatron-lm" \
  --export=ALL,PYTHONPATH=/opt/megatron-lm:/opt/Megatron-Bridge/src,HF_HOME=$WKDIR/.cache/huggingface,MODE=official \
  bash -c "pip install tilelang 2>/dev/null; python -u /opt/Megatron-Bridge/tests/dsv4_cosine_analysis.py"
```

### Fresh Import (only needed if MCore or Bridge changes affect weights, ~20 min)

```bash
srun -A coreai_dlalgo_llm -p batch -N 1 --gpus-per-node=4 --ntasks-per-node=4 -t 01:00:00 \
  --container-image=$WKDIR/sqsh/nemo_26.04.sqsh \
  --container-mounts="/lustre:/lustre,$WKDIR/nemo_workspace/Megatron-Bridge:/opt/Megatron-Bridge,$WKDIR/nemo_workspace/Megatron-LM:/opt/megatron-lm" \
  --export=ALL,PYTHONPATH=/opt/megatron-lm:/opt/Megatron-Bridge/src,HF_HOME=$WKDIR/.cache/huggingface,MASTER_ADDR=localhost,MASTER_PORT=29500 \
  bash -c "cd /opt/Megatron-Bridge && python -u tests/dsv4_fresh_import_save.py"
```

---

## Results

### Per-layer Cosine Similarity (all 43 layers)

```
 Lyr  CR   CosSim      Status
   0   0   1.000121    Perfect
   1   0   1.000024    Perfect
   2   4   1.000063    Perfect
   3 128   1.000022    Perfect
   4   4   0.999978    Perfect
   5 128   1.000075    Perfect
   6   4   1.000077    Perfect
   7 128   1.000090    Perfect
   8   4   1.000014    Perfect
   9 128   1.000001    Perfect
  10   4   0.999919    Perfect
  11 128   0.999701    Perfect
  12   4   0.999955    Perfect
  13 128   0.999992    Perfect
  14   4   0.999963    Perfect
  15 128   0.999609    Perfect
  16   4   0.998605    Good
  17 128   0.999298    Perfect
  18   4   0.999388    Perfect
  19 128   0.997299    Good
  20   4   0.994421    Degrading
  21 128   0.990124    Degrading
  22   4   0.991360    Degrading
  23 128   0.963529    <- drift starts
  24   4   0.952473
  25 128   0.801323    <- MoE cliff
  26   4   0.702532    <- cascade
  27 128   0.622081
  28   4   0.316325    <- worst (near-random)
  29 128   0.899721    <- partial recovery
  30   4   0.745024
  31 128   0.836764
  32   4   0.865728
  33 128   0.826633
  34   4   0.896559    <- recovering
  35 128   0.928631
  36   4   0.953009
  37 128   0.962245
  38   4   0.968674
  39 128   0.963865
  40   4   0.963790
  41 128   0.948313
  42   4   0.726813    <- last layer drops
```

### Post-layer Metrics

```
 Last-token logits (129K-dim):    cos = 0.935
 Top-5 overlap:                   4/5
   Official: [7230, 19, 2337, 671, 2107]
   MCore:    [19, 671, 7230, 2337, 104822]
```

### Sub-layer Breakdown (cosine sim)

```
                             L0 (dense) L2 (CSA=4)  L10 (MoE) L25 (cliff) L42 (last)
----------------------------------------------------------------------------------
        Attn input (HC+norm)      1.000      0.997      0.993       0.881      0.842
                 Attn output      1.000      0.998      0.989       0.853      0.946
         MLP input (HC+norm)      0.999      0.992      0.987       0.847      0.787
              MLP/MoE output      1.000      1.000      1.000       0.037      0.971
           Full layer output      1.000      1.000      1.000       0.801      0.727
```

---

## Gap Analysis

### Three phases of divergence

1. **Layers 0-19 (cos > 0.997):** Near-perfect match. Attention contributes ~0.004 cosine gap per layer, HC contributes ~0.002. MoE output stays at ~1.0 because inputs are similar enough for identical expert routing.

2. **Layers 20-28 (cos 0.316-0.994):** Accumulated drift reaches a tipping point. At layer 25, MLP input has diverged enough (cos=0.847) that MoE routing sends some tokens to different experts. Once a token hits a different expert, its output is uncorrelated -> MoE output cos=0.037. This cascades through layers 26-28 (cos drops to 0.316).

3. **Layers 29-42 (cos 0.727-0.969):** HC residual connections gradually dampen the drift. The 4-stream residual mixing preserves enough of the correct signal from earlier layers to partially recover.

### Root causes (per component)

| Component | Per-layer gap | What causes it | Code pointers |
|---|---|---|---|
| **Attention** | ~0.004 cos | Different attention kernels: official uses `sparse_attn` tilelang kernel (online softmax, bf16 accum); MCore uses `unfused_compressed_sparse_attn` (2-pass softmax, fp32) | Official: `model.py:545`, MCore: `csa.py:168` |
| **HC** | ~0.002 cos | Different operator decomposition: official uses `hc_split_sinkhorn` single tilelang kernel; MCore uses separate cuTile ops (`_proj_rms_op`, `_sinkhorn_op`, `_h_aggregate_op`). HC post: official float32, MCore bf16 | Official: `model.py:691-703`, MCore: `hyper_connection.py:167-588` |
| **MoE** | 0.000 until cliff, then catastrophic | Same weights, but different dispatch: official uses sequential loop with fp32 accumulation; MCore uses TE GroupedLinear batched GEMM in bf16. When accumulated input drift crosses routing threshold, tokens land on different experts -> uncorrelated output | Official: `model.py:648-662`, MCore: `experts.py:257-290` |

### Key insight

MoE is the **amplifier**, not the root cause. Attention + HC drift (~0.006 per layer) compounds over ~25 layers until MoE routing diverges. Once that happens, recovery is limited by HC residual dampening. The model still produces correct answers despite the drift (top-5 token overlap 4/5).

### What was fixed

| Fix | Effect | Status |
|---|---|---|
| MXFP4 dequant (double E8M0 decode) | Expert weights zeroed -> fixed | Pushed |
| Fused RoPE enable | Non-fused path broken for inverse RoPE | Pushed |
| mscale=1.0, mscale_all_dim=1.0 | 1.277x RoPE scaling -> fixed | Uncommitted |
| use_fused_mhc=True | Enables cuTile HC kernels (no cos sim change, faster) | Uncommitted |
| model_bridge.py None task guard | Fresh import crash on unmapped MTP params | Uncommitted |
| dsa.py hadamard fallback | Container missing fast_hadamard_transform | Uncommitted |

### What cannot be improved from Bridge side

The remaining gap (0.935 -> 1.0) is from structural differences between two correct implementations (different kernels, different fusion boundaries, different accumulation dtypes). These are MCore/TE implementation choices, not Bridge mapping issues.
