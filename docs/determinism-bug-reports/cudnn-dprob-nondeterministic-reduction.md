# [BUG] Grouped-MLP dSReLU epilogue computes `dprob` with an order-dependent reduction

**Component:** `cudnn` Python package (CuTe DSL kernels) — `grouped_gemm/grouped_gemm_dsrelu/`
**Version:** cudnn 1.26.0 (as shipped in `nvcr.io/nvidian/nemo:26.08.rc3`)
**Caller:** Transformer Engine 2.18.0+e7c550c5, `pytorch/ops/fused/grouped_mlp.py`
**Hardware:** GB300 (sm_100), MXFP8 block-scaled grouped MLP, 256 GPUs

## Summary

`grouped_gemm_dsrelu_wrapper_sm100` computes the routing-probability gradient
`dprob` with **two order-dependent float32 accumulations**. Two identical runs of the
same model produce different `dprob`, so MoE training is not bit-reproducible even
with every documented determinism switch enabled.

## Reproduction

Nemotron-3-Ultra, MXFP8, 256×GB300, `tp=1 pp=1 cp=1 dp=256 ep=64`, 512 experts,
`moe_router_topk=22`, HybridEP dispatcher. Two identical launches, 3 steps, every ATen
op and collective fingerprinted per rank and diffed.

`dprob` is the **first divergent value in the backward pass on 256 of 256 ranks**,
relative `|Δsum| ≈ 1.5e-6`, with bit-identical inputs. It surfaces to the caller at
`transformer_engine/pytorch/ops/fused/grouped_mlp.py:1836`:

```python
grad_scales = fc2_dgrad_kernel_out["dprob_tensor"].view(-1)
```

## Root cause

`moe_blockscaled_grouped_gemm_dsrelu_quant.py`. The kernel's own docstring states the math:

```
dprob[m] += Σ_n( relu(C[m,n])² · alpha · acc[m,n] )
```

The reduction is over `N`, which is tiled across CTAs. Two distinct order dependencies:

### Level 3 — cross-CTA atomic (line 2377)

```python
_ = atomic_add_float32(
    ptr=real_dprob[(mPosition, None, None)].iterator.llvm_ptr,
    value=dProbVal,
)
```

~`ceil(N/256)` CTAs atomically add their partial into the same scalar; arrival order sets
the low bits. `atomic_add_float32` lowers to a hardware FADD
(`grouped_gemm/moe_kernel_helpers.py:325`, `nvvm.atomicrmw(op=AtomicOpKind.FADD, ...)`),
whose own docstring reads *"used for dprob gradient accumulation"*.

### Level 2 — subtile traversal order (lines 2116, 2186)

```python
reverse_subtile = cutlass.Boolean(True) if acc_consumer_state.phase == 0 else cutlass.Boolean(False)
...
dProbVal = dProbVal + <subtile contribution>     # accumulated in traversal order
```

`reverse_subtile` is `(this tile's position within THIS CTA's work sequence) mod 2` — the
producer mirrors it with a toggle at lines 1893-1896. With `use_dynamic_sched=True`, tiles
are assigned by a runtime work counter, so the same tile is summed forward in one run and
backward in another. **Each per-tile partial therefore varies before any atomic is involved.**

Both must be fixed; each alone was measured to have no effect.

## Impact

MoE models using the fused grouped MLP cannot be bit-reproducible. Beyond reproducibility
this blocks debugging techniques that require run-to-run equality (A/B bisection of numerical
regressions, deterministic checkpoint/resume equivalence).

`dprob` is not a diagnostic quantity — it is the gradient of the routing probabilities and
feeds router weight updates directly.

## No workaround exists at the API level

- The wrapper exposes only `use_dynamic_sched` and `use_dsrelu_reuse`; neither controls the
  atomic.
- `overlapping_accum` (which drives `reverse_subtile`) is derived, not a parameter:
  `self.num_acc_stage == 1 and self.mma_tiler[1] == 256` (line 405).
- Forcing `reverse_subtile = False` is **unsafe** — the reversal guards overlapped TMEM
  accumulator regions (line 414); it is a correctness mechanism, not a free choice.

## Suggested fix (verified working)

Make `dprob` single-writer, then reduce in fixed order. This is licensed by line 2099:

```python
mPosition = epi_work_tile_info.tile_m_idx * self.cta_tile_shape_mnk[0] + tidx
```

Each thread owns exactly one `m`, so there is exactly **one writer per `(m, tile_n)`** — the
atomic can become a plain store with no contribution lost.

```python
# kernel: store into a per-N-tile slot instead of atomically accumulating
if deterministic:
    real_dprob[mPosition, epi_work_tile_info.tile_n_idx, 0] = dProbVal
else:
    _ = atomic_add_float32(...)          # unchanged default
```

```python
# caller: allocate (M, ceil(N/256), 1), then reduce along dim 1 in fixed order
grad_scales = dprob.sum(dim=1).view(-1)
```

plus `use_dynamic_sched=False` on the dprob-carrying invocation so the tile→CTA sequence,
and hence `acc_consumer_state.phase`, is reproducible.

Two implementation notes:

1. **Slots must live in dim 1.** The MoE domain conversion preserves `shape[1]` but hardcodes
   dim 2 (`moe_sched_extension.py:164`, `c1 = cutlass.Int32(1)`), so an `(M, 1, NTILES)`
   layout is silently flattened.
2. `api.py:331` asserts `dprob` is `(tensor_m, 1, 1)` and must accept the wider tensor. Bound
   it against `ceil_div(n_out, mma_tiler_mn[1])` — an under-allocation would otherwise write
   out of bounds silently.

## Validation of the suggested fix

Applied to the reproduction above:

| | divergent records |
|---|---|
| stock | ~16,000 per rank |
| level 3 only | ~15,800 per rank (divergence moves to the new reduction) |
| **levels 2 + 3** | **0 across 256/256 ranks, 13,706,980 records compared** |

wandb metrics compared as raw IEEE-754 bytes: **15/15 bitwise identical**, including
`lm loss 10.965963363647461 (0x4025ee92c0000000)`.

Numerically equivalent, not merely reproducible: `lm loss` versus the stock atomic build
agrees to ~1e-5 relative — the expected float-reassociation difference from summing the same
terms in a fixed order.

## Related

Transformer Engine already treats this exact reduction as nondeterministic in its *Triton*
implementation and refuses to run it under `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`
(`transformer_engine/pytorch/triton/grouped_dbias_dscales.py:35`). The cuDNN path computing
the same quantity is unguarded. See the companion TE report.
