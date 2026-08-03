# [BUG] `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` does not cover the cuDNN grouped-MLP `dscales` path

**Component:** Transformer Engine — `transformer_engine/pytorch/ops/fused/grouped_mlp.py`
**Version:** 2.18.0+e7c550c5 (also present in 2.19.0.dev0)
**Hardware:** GB300 (sm_100), MXFP8 block-scaled grouped MLP

## Summary

TE has **two** implementations that compute the routing-probability gradient
(`dscales` / `dprob`). One is guarded by `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`; the other,
which is what actually runs on this path, is not. The flag is documented as enforcing
deterministic execution, and here it silently does not.

## The inconsistency

**Triton implementation — guarded, refuses to run:**

```python
# transformer_engine/pytorch/triton/grouped_dbias_dscales.py:35
if _is_deterministic_mode():
    raise RuntimeError(
        "grouped_dbias Triton kernel uses non-deterministic atomic adds "
        "and cannot be used when deterministic execution is requested "
        "(NVTE_ALLOW_NONDETERMINISTIC_ALGO=0). "
        "Disable determinism or use a deterministic fallback."
    )
```

backed by `tl.atomic_add` in the underlying kernel.

**cuDNN implementation — unguarded:**

`ops/fused/grouped_mlp.py` passes `dprob_tensor` into
`cudnn.grouped_gemm_dsrelu_wrapper_sm100`, which accumulates it with a hardware FADD atomic
plus an order-dependent subtile sum. `grep NVTE_ALLOW_NONDETERMINISTIC_ALGO` over
`ops/fused/grouped_mlp.py` returns **nothing**.

So TE explicitly classifies this reduction as nondeterministic, then runs a different
implementation of the same reduction without checking the flag.

## Reproduction

Nemotron-3-Ultra MXFP8, 256×GB300, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` set. Two identical
runs diverge; `dprob` is the first divergent value in the backward on 256/256 ranks
(`grouped_mlp.py:1836`, rel `|Δsum| ≈ 1.5e-6`, bit-identical inputs). Full analysis in the
companion cuDNN report.

## Impact

Users who set the flag reasonably believe determinism is enforced. It is not, and the failure
is silent — no warning, no error. This cost several days of investigation precisely because
the flag was set and assumed effective.

## No configuration avoids the path

- `moe_apply_probs_on_input` — asserts `moe_router_topk == 1`
  (`megatron/core/transformer/moe/experts.py:729`); MoE models with topk > 1 cannot use it.
- `_grouped_mlp_unit_activation_scale` — only ever *read* (`grouped_mlp.py:1060`), never set,
  and forced `False` when `num_groups != 1` (line 1062).
- `NVTE_CUTEDSL_FUSED_GROUPED_MLP=0` — disables the fused path, but hits
  `RuntimeError: ScaledSReLU(activation_recompute_in_mlp=True) requires the fused grouped MLP
  path` whenever `recompute_modules` contains `moe_act`.

## Requested fix

**Short term** — honour the flag. Either raise, matching the Triton path, or fall back to a
deterministic implementation:

```python
if not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"))):
    raise RuntimeError(
        "cuDNN grouped-MLP dscales/dprob uses non-deterministic atomic adds and an "
        "order-dependent subtile reduction; not available under "
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO=0"
    )
```

An explicit failure is strictly better than silently returning nondeterministic gradients
under a flag that promises otherwise.

**Preferred** — make the path deterministic. A verified fix is in the companion cuDNN report:
per-N-tile single-writer slots plus a fixed-order reduction, with `use_dynamic_sched=False`.
TE's side is small:

```python
_det_nslots = (fc1_weight_shape[0] + 255) // 256   # kernel asserts mma_tiler N == 256
dscales_tensor = torch.zeros((scales_tensor.shape[0], _det_nslots, 1),
                             dtype=torch.float32, device=scales_tensor.device)
...
grad_scales = fc2_dgrad_kernel_out["dprob_tensor"].sum(dim=1).view(-1)
```

Measured result: **0 divergent records across 256/256 ranks** (13,706,980 compared), and
15/15 wandb metrics bitwise identical, with `lm loss` matching the stock atomic build to
~1e-5 relative — the expected reassociation difference.

## Secondary observation

`ops/fused/grouped_mlp.py` also hardcodes `"use_dynamic_sched": True` on the dprob-carrying
kernel invocation. Under a determinism flag this should be `False`: the tile→CTA sequence
determines `acc_consumer_state.phase`, which flips the subtile traversal direction and hence
the summation order of each per-tile partial.
