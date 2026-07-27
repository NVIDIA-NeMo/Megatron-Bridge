# Reproduce: Nemotron 3 Ultra gb300 MXFP8 deterministic bit-wise check

How to reproduce the `nemotron-3-ultra-det-bitwise-check-*` run (the 128-GPU / 32-node
gb300 MXFP8 `deterministic_mode=true` bit-reproducibility check). **No new script — this
is produced by the existing launcher** [`../launch_nemotron_3_ultra_nsys_compare.sh`](../launch_nemotron_3_ultra_nsys_compare.sh):
its **`det-bitwise`** arm emits exactly that `run_script.py … deterministic_mode=true`
command (`ENABLE_NSYS=false`; see the script's `submit_run()` and the `det-bitwise` /
`det-bitwise2` cases). The launcher submits two independent no-nsys deterministic
allocations (`det-bitwise` + `det-bitwise2`) and diffs their per-iter loss to answer
*"does this recipe reproduce bit-for-bit across allocations?"*.

## Reproduce it

Run the existing launcher with the gb300 / MXFP8 recipe knobs (its
`# Recipe selection knobs` block):

```bash
GPU_TYPE=gb300 \
COMPUTE_DTYPE=fp8_mx \            # -c fp8_mx (underscore)
NGPUS=128 GN=4 \                  # 128 GPU = 32 gb300 nodes
KEEP_RECIPE_DISPATCHER=1 \        # keep native HybridEP (else setup_experiment forces alltoall)
NVTE_CPU_OFFLOAD_V1=1 \           # fine-grained activation offload (gb300 fp8 path)
NCCL_CUMEM_ENABLE=1 \            # HybridEP MNNVL fabric needs cuMem
HF_HUB_OFFLINE=0 \               # online is fine at <= 128 GPU
TRAIN_ITERS=50 \
ACCOUNT=<account> PARTITION=<partition> \
CONTAINER_IMAGE=<gb300 NeMo image validated for MXFP8, e.g. 26.08.rc2> \
REPO_ROOT=<absolute path to this checkout> \
HF_CACHE=<shared HF cache dir> \
HF_TOKEN=<token> WANDB_API_KEY=<key> \
PYTHON=<interpreter with nemo_run> \
bash scripts/performance/launch_nemotron_3_ultra_nsys_compare.sh
```

Cluster note: set `GRES=gpu:N` (e.g. `GRES=gpu:4`) if your partition does not
auto-allocate GPUs; the launcher only auto-sets GRES on `oci-hsg-cs-001`.

## What the `det-bitwise` arm adds (not visible from the env inputs above)

The env knobs above map 1:1 to `--gpu`/`-c`/`-ng`/`-gn` etc.; what the arm injects on
top — and you can't see in your own command — is the determinism block:

| Source | Produces in the command |
|---|---|
| `det-bitwise` arm | `model.deterministic_mode=true model.cross_entropy_loss_fusion=false` + `-E NCCL_ALGO=Ring -E NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 -E CUBLAS_WORKSPACE_CONFIG=:4096:8 -E MAMBA_DETERMINISTIC=1`, `ENABLE_NSYS=false` |
| other launcher defaults | `model.moe_router_fusion=true`, `train.fill_uninitialized_memory=false`, `train.manual_gc*`, the `logger.*` tensorboard overrides |

## Read the result

The launcher waits for `det-bitwise` and `det-bitwise2`, then diffs their per-iter
`lm loss` / grad-norm from the two job logs (see its tail section). Bit-exact ⇒ every
sampled iter matches to the last printed digit; any divergence ⇒ the two allocations do
**not** reproduce. Job IDs and the diff are printed at the end and land under `$OUT_DIR`.

## Prerequisites / gotchas

- **Sync the submodule first:** `git submodule update --init --recursive` (a stale
  `3rdparty/Megatron-LM` fails at import: `No module named 'megatron.training.models.gpt'`).
- **Secrets:** the launcher's INFO log echoes `HF_TOKEN` / `WANDB_API_KEY` in plaintext —
  never paste real tokens into a shared transcript; rotate if you do.
- **HF cache:** at > 128 GPU, pre-stage the cache and set `HF_HUB_OFFLINE=1` (online at
  scale trips the HF rate limit → NCCL cascade).

This is the deterministic-reproducibility check (`deterministic_mode=true`). It is
separate from the cross-process **first-divergence trace** (`DET_TRACE_*` +
`diff_streams.py`); see [`../../../docs/determinism-debug-tool.md`](../../../docs/determinism-debug-tool.md).
