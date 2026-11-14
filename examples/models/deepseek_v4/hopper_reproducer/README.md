# DSv4-Flash Hopper (H100) NaN Reproducer

## Bug

DSv4-Flash pretrain hits NaN at **iteration 2** on H100 (SM90). The same config
completes 1k+ steps on GB200 (SM100). The NaN is in the forward loss computation
on the last pipeline stage, after exactly 2 optimizer updates.

**100% reproducible** across all tested optimizer/precision/parallelism/recompute
combinations on Hopper.

## Environment

| Component | Version / Commit |
|-----------|-----------------|
| Cluster | CW-DFW, H100-80GB SXM |
| Container | `nemo_26.06.rc2.sqsh` |
| Bridge | `main` @ `4b68d3e59` |
| MCore | `dev` @ `630956b35` (Jun 2, 2026) |
| SLURM account | `coreai_dlalgo_genai` |
| Data | DCLM preprocessed (any tokenized data works; NaN is data-independent) |

## Quick Reproduce (BF16, full recompute, 32 nodes)

This is the simplest reproducer — no extra dependencies beyond Bridge main + MCore dev.

```bash
# On CW-DFW login node
WKDIR=/lustre/fsw/portfolios/coreai/users/weijiac

# 1. Ensure MCore dev is checked out
cd $WKDIR/nemo_workspace/Megatron-LM
git checkout dev && git pull

# 2. Submit the smoke run (Adam BF16, full recompute, PP=32 EP=8)
cd /path/to/this/directory
MODE=adam_dsaoff \
RUN_TAG=nan-repro \
RECOMPUTE_GRANULARITY=full \
RECOMPUTE_METHOD=uniform \
RECOMPUTE_NUM_LAYERS=1 \
RECOMPUTE_MODULES=null \
sbatch cw_dsv4_hopper_smoke_matrix.sh
```

Expected outcome: NaN at iteration 2, ~18 min after training loop starts.

```
RuntimeError: Rank 255, ..., iteration 2: Unexpected result nan
  (message='found NaN in local forward loss calculation')
```

## Variants Tested (all NaN at iter 2)

| PP | EP | Optimizer | Precision | Recompute | Fused DSA | Eval |
|----|-----|-----------|-----------|-----------|-----------|------|
| 32 | 8 | Adam | BF16 | full | off | on |
| 32 | 8 | Adam | BF16+CP2 | full | off | on |
| 32 | 8 | Adam | blockwise FP8 | full | off | on |
| 32 | 8 | Muon | BF16 | full | off | on |
| 16 | 16 | Adam | BF16 | full | off | on |
| 16 | 16 | Adam | BF16 | selective `[moe_act,mhc]` | on (SM90) | on |
| 16 | 16 | Adam | BF16 | selective `[moe_act,mhc]` | on (SM90) | **off** |

### To reproduce Muon variant:
```bash
MODE=muon_dsaoff RUN_TAG=muon-nan-repro \
RECOMPUTE_GRANULARITY=full RECOMPUTE_METHOD=uniform \
RECOMPUTE_NUM_LAYERS=1 RECOMPUTE_MODULES=null \
sbatch cw_dsv4_hopper_smoke_matrix.sh
```

## Working Comparison (Blackwell)

OCI GB200, job `3021485` — completed 1k steps with:
```
Recipe:     deepseek_v4_flash_pretrain_muon_config
TP=1  PP=8  EP=8  CP=1  (16 GB200 nodes)
Recompute:  selective [moe_act,mhc]
Optimizer:  Muon BF16
DSA fusion: off
```

## Key Observations

1. **Hopper-specific**: same code, same MCore, same Bridge — works on Blackwell, NaN on Hopper.
2. **Optimizer-independent**: both Adam and Muon NaN.
3. **Precision-independent**: BF16, BF16+CP2, blockwise FP8 all NaN.
4. **Recompute-independent**: full recompute and selective recompute both NaN.
5. **Fused/unfused DSA-independent**: both paths NaN.
6. **PP-independent**: PP=16 and PP=32 both NaN.
7. **Eval-independent**: NaN with eval on or off.
8. **Always iteration 2**: after exactly 2 optimizer updates, iteration 2 forward NaN's.
9. **Always last PP stage**: NaN on the MTP + loss ranks.

## Fused DSA Variant (optional, requires extra setup)

The fused DSA + selective recompute variant is notable because it's the only config
that **fits in H100-80GB memory with selective recompute** (unfused selective OOMs
by ~640 MiB). It requires:

1. **MCore patch** — relax SM100 assertion at `transformer_config.py:1455`:
   ```python
   # Change: assert sm[0] >= 10
   # To:     assert sm[0] >= 9
   ```

2. **cudnn-frontend PR263 overlay** — merged overlay with container's cudnn + PR263
   DSA SM90 kernels at: `$WKDIR/training/dsv4_cudnn_merged_overlay/`

3. **FlashMLA H100 build** — x86_64 build at: `$WKDIR/training/flash_mla_h100_site/`

Submit:
```bash
MODE=adam_dsaoff RUN_TAG=fused-dsa-repro PP=16 EP=16 \
RECOMPUTE_GRANULARITY=selective RECOMPUTE_METHOD=null \
RECOMPUTE_NUM_LAYERS=null RECOMPUTE_MODULES="[moe_act,mhc]" \
EXTRA_OVERRIDES="model.apply_dsa_kernel_fusion=true" \
PIPELINE_LAYOUT="Et*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*2|t*2|t*2|t*2|t*2mL" \
CUDNN_DSA_SITE=$WKDIR/training/dsv4_cudnn_merged_overlay \
FLASH_MLA_SITE=$WKDIR/training/flash_mla_h100_site \
sbatch cw_dsv4_hopper_smoke_matrix.sh
```

## Reference W&B Runs

Project: `nvidia-nemo-fw-public/megatron-bridge-dsv4`

| Run ID | Config | Result |
|--------|--------|--------|
| `ajq033g1` | Adam BF16 PP=32 full recomp | NaN iter 2 |
| `fit073gv` | Adam BF16 CP2 PP=32 full recomp | NaN iter 2 |
| `6m42u2cn` | Adam FP8 PP=32 full recomp | NaN iter 2 |
| `75lrwpuq` | Adam BF16 PP=16 fused DSA selective | NaN iter 2 |

## Launcher Script

`cw_dsv4_hopper_smoke_matrix.sh` — configurable via env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `muon_dsaoff` | `adam_dsaoff`, `muon_dsaoff`, `adam_fp8block_dsaoff` |
| `PP` | 32 | Pipeline parallel size |
| `EP` | 8 | Expert parallel size |
| `TRAIN_ITERS` | 20 | Number of training iterations |
| `RECOMPUTE_GRANULARITY` | selective | `selective` or `full` |
| `RECOMPUTE_METHOD` | null | `uniform` for full recompute |
| `RECOMPUTE_NUM_LAYERS` | null | `1` for full recompute |
| `RECOMPUTE_MODULES` | `[moe_act,mhc]` | Modules to selectively recompute |
| `EXTRA_OVERRIDES` | | Additional Hydra overrides |
| `RUN_TAG` | | Suffix for run name |
| `PIPELINE_LAYOUT` | (PP=32 layout) | Custom PP layout string |
| `CUDNN_DSA_SITE` | (PR263 overlay) | Path to cudnn DSA overlay |
| `FLASH_MLA_SITE` | (aarch64 build) | Path to FlashMLA site |
