# Determinism root-cause analysis — Nemotron-3-Ultra MXFP8, 256 GPUs (gb300)

**Question:** two identical launches of the same recipe under `--deterministic` do not
produce bit-identical results. Where does the divergence originate?

**Method.** Run two arms (A/B) of the same config, fingerprint every ATen op and every
collective per rank into an ordered stream, and diff. A record is an **origin** only if
`inputs MATCH ∧ output DIFFERS`; anything with differing inputs is inherited.

**Versions.** Container `nvcr.io/nvidian/nemo:26.08.rc3`; Transformer Engine
**2.18.0+e7c550c5**; cudnn python package **1.26.0**; Megatron-Core submodule as pinned.
Config: `tp=1 pp=1 cp=1 dp=256 ep=64`, 512 experts, `moe_router_topk=22`,
`moe_router_dtype=fp32`, `seq_aux_loss`, MXFP8, Megatron-FSDP,
`num_distributed_optimizer_instances=4`, HybridEP flex dispatcher.

> Line numbers below were read from the **container's own files**, extracted with
> `srun ... shutil.copy(...)`. They differ from public checkouts — a local TE 2.19.0.dev0
> checkout had the same code at different lines. Always extract from the image you run.

---

## Summary of findings

| # | Source | Location | Status |
|---|---|---|---|
| 1 | MoE aux-loss `atomicAdd` | TE `common/fused_router/fused_moe_aux_loss.cu:69,173` | **Confirmed, fixed** |
| 2 | `dprob`/`dscales` order-dependent sums in the cuDNN grouped-MLP backward | cudnn `moe_blockscaled_grouped_gemm_dsrelu_quant.py` — subtile sum (2186) + `atomic_add_float32` (2377) | **Confirmed; FIX VERIFIED — 0/13.7M divergent records, 256/256 ranks** |
| 3 | ~~bf16 gradient reduce-scatter~~ | `megatron_fsdp/param_and_grad_buffer.py:4092` | **RETRACTED — carrier, not source** (single-rank test invalid for collectives) |

---

## Source 1 — fused MoE aux loss (forward)

`switch_load_balancing_loss_func(..., fused=True)` dispatches to TE's
`fused_moe_aux_loss`, which accumulates with `atomicAdd`:

```
transformer_engine/common/fused_router/fused_moe_aux_loss.cu
  69:  atomicAdd(&Coeff_buf[1], static_cast<float>(block_sum * coeff));
 173:  atomicAdd(&Coeff_buf[1], static_cast<float>(block_sum * C_coeff));
```

**Evidence.** With fusion on, **222 of 256 ranks** had their first divergence at
`moe/router.py` in the layer-1 router, and every `|Δ|` was an integer multiple of
**2⁻³⁷** — exactly one float32 ULP. With fusion off, **0 of 256** did, and the entire
forward pass became bit-identical. Reproduced on three independent pairs (`bwd`/`post`
records only, zero `fwd` records in all three).

**This is not a logging-only value.** `MoEAuxLossAutoScaler.apply(activation, aux_loss)`
(`moe/router.py:666,670`) puts it on the autograd graph at `moe_aux_loss_coeff=1e-4`, so a
1-ULP perturbation becomes a router-weight gradient.

**Fix applied** (`recipes/utils/determinism_utils.py`): `cfg.model.moe_router_fusion = False`.

**This fix is overbroad and should be replaced.** `moe_router_fusion` gates seven call
sites in `moe/router.py` — the aux-loss ones (467, 571, and the `seq_aux_loss` site) *and*
the top-k routing ones (413, 817, 999) plus `compute_routing_scores_for_aux_loss` (839).
Routing was never implicated: routing decisions matched bit-for-bit on every rank in every
trace. Megatron exposes no narrower flag (`transformer_config.py:898` is a single bool).

**Proper upstream fix** — gate only the aux-loss sites:

```python
fused=self.config.moe_router_fusion and not self.config.deterministic_mode
```

in `_apply_aux_loss` / `_apply_seq_aux_loss` / `_apply_global_aux_loss`. `deterministic_mode`
already propagates, so no new config field is needed. Once that lands, delete the Bridge line.

---

## Source 2 — `dprob` atomics in the cuDNN fused grouped MLP (backward)

With source 1 removed the forward is clean and the first mismatch moves to the backward,
at the same location on **256/256 ranks**, rel `|Δsum| ≈ 1.5e-6`:

```
transformer_engine/pytorch/ops/fused/grouped_mlp.py
 1710:  dscales_tensor = torch.zeros_like(scales_tensor)      # float32 (M,1,1)
 1747:  "dprob_tensor": dscales_tensor,                       # -> cuDNN kernel
 1836:  grad_scales = fc2_dgrad_kernel_out["dprob_tensor"].view(-1)   # FIRST DIVERGENT VALUE
```

The kernel is **Python (CuTe DSL)**, not a closed binary — `cudnn.grouped_gemm_dsrelu_wrapper_sm100`,
source at `cudnn/grouped_gemm/grouped_gemm_dsrelu/`. Its own docstring gives the math:

```
dprob[m] += Σ_n( relu(C[m,n])² · alpha · acc[m,n] )
```

The reduction is over **N**, and N is tiled across CTAs:

```python
# moe_blockscaled_grouped_gemm_dsrelu_quant.py:2353
real_dprob, _ = epi_ext.get_gmem_tensor("dprob", dprob, padded_offsets, epi_work_tile_info)
_ = atomic_add_float32(
    ptr=real_dprob[(mPosition, None, None)].iterator.llvm_ptr,
    value=dProbVal,
)
```

Float addition is not associative, so arrival order at that atomic decides the low bits.

`atomic_add_float32` bottoms out in a hardware FADD atomic — there is no deeper layer to
inspect, no `.so`, no C++ fallback:

```python
# cudnn_pkg/grouped_gemm/moe_kernel_helpers.py:325
def atomic_add_float32(ptr, value: Float32, ...) -> Float32:
    """Atomic FP32 addition in global memory (used for dprob gradient accumulation)."""
    old_value = nvvm.atomicrmw(op=AtomicOpKind.FADD, ptr=ptr, a=value.ir_value(...), ...)
```

cuDNN's own docstring names dprob as the reason this helper exists. The whole kernel is CuTe
DSL Python plus inline PTX (cf. the sibling `atomic_add_bf16x2`, which emits
`red.global.add.noftz.bf16x2` literally), so the package extracts to 187 `.py` files and zero
binaries — the code that runs is fully readable.

**Caveat for anyone patching this:** the package defines `atomic_add_float32` in FOUR places
(`grouped_gemm/utils.py:268`, `grouped_gemm/moe_kernel_helpers.py:325`,
`gemm_dsrelu/dense_blockscaled_gemm_persistent_dsrelu_quant.py:50`,
`discrete_grouped_gemm/discrete_kernel_utils.py:343`). The dsrelu kernel imports the
`moe_kernel_helpers` one. Patching a same-named sibling silently does nothing.

**Where to read it:** this code ships only in the container — it is NOT in the
TransformerEngine repo. Grepping a TE checkout for `atomic_add_float32`, `dProbVal`,
`reverse_subtile`, or `overlapping_accum` returns nothing; TE holds only the call boundary
(`from cudnn import grouped_gemm_dsrelu_wrapper_sm100`). Extracted copy lives at
`/lustre/.../te-patch/cudnn_pkg/`. Package version: `cudnn` 1.26.0.

### Three levels of reduce ordering

| Level | Where | Deterministic? |
|---|---|---|
| 1 | within a thread's register fragment (`tDprob.load().reduce(ADD)`) | yes |
| 2 | across N-subtiles inside one work tile (`dProbVal = dProbVal + …`) | **no** — traversal flips |
| 3 | across N-tiles / CTAs into `dprob[m]` | **no** — `atomic_add_float32` |

Level 2 (line 2106):

```python
if cutlass.const_expr(self.overlapping_accum):
    acc_stage_index = acc_consumer_state.phase
    reverse_subtile = cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
```

`overlapping_accum` is derived, not a user knob:
`self.overlapping_accum = self.num_acc_stage == 1 and self.mma_tiler[1] == 256` (line 405).
Forcing `reverse_subtile = False` is **unsafe** — the reversal guards overlapped TMEM
accumulator regions (line 414), so it is a correctness mechanism, not a free choice.

### No determinism guard exists on this path

`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` does **not** cover it. TE guards the *Triton*
implementation of the same reduction and refuses to run it:

```python
# transformer_engine/pytorch/triton/grouped_dbias_dscales.py:35
if _is_deterministic_mode():
    raise RuntimeError(
        "grouped_dbias Triton kernel uses non-deterministic atomic adds "
        "and cannot be used when deterministic execution is requested "
        "(NVTE_ALLOW_NONDETERMINISTIC_ALGO=0). ..."
    )
```

but the cuDNN implementation computing the same quantity runs unguarded — `grep` for
`NVTE_ALLOW_NONDETERMINISTIC_ALGO` in `ops/fused/grouped_mlp.py` returns nothing. **This
inconsistency is itself a bug**: the flag is documented as enforcing determinism.

### No config lever avoids the path

- `moe_apply_probs_on_input` — asserts `moe_router_topk == 1` (`moe/experts.py:729,1269`); this model uses 22.
- `_grouped_mlp_unit_activation_scale` — only ever *read* (`grouped_mlp.py:1060`), never set, and forced `False` when `num_groups != 1` (line 1062); this model has 8.
- `NVTE_CUTEDSL_FUSED_GROUPED_MLP=0` — TE's own default is `0` and the perf recipe opts in (`perf_recipes/nemotronh/gb300/nemotronh.py:243`), but disabling it hits
  `RuntimeError: ScaledSReLU(activation_recompute_in_mlp=True) requires the fused grouped MLP path`, since `activation_recompute_in_mlp` derives from `recompute_modules=[moe_act]` (`moe/experts.py:475`).
- `moe_grouped_gemm=false` — selects SequentialMLP; OOM-killed at 256 GPUs.

### Implemented fix — single-writer slots

The key enabling fact, from line 2099:

```python
mPosition = epi_work_tile_info.tile_m_idx * self.cta_tile_shape_mnk[0] + tidx
```

Each thread owns exactly **one** `m`, so there is exactly **one writer per `(m, tile_n)`**.
That licenses replacing the atomic with a plain store into a per-tile slot, then reducing
in fixed order:

1. **TE** allocates `(M, n_slots, 1)` instead of `(M,1,1)` and does `.sum(dim=1)` after the kernel.
2. **Kernel** stores `real_dprob[mPosition, epi_work_tile_info.tile_n_idx, 0] = dProbVal`.
3. **api.py:331** shape assertion relaxed (the rest of the pipeline already carries the real extent via `dprob_desc.shape[1:]`).

Two details that would silently break a naive attempt:

- **Slots must live in dim 1, not dim 2.** The MoE domain conversion preserves `shape[1]`
  but hardcodes dim 2 (`moe_sched_extension.py:164`, `c1 = cutlass.Int32(1)`; used at 189-193).
  A `(M,1,NTILES)` layout is silently flattened.
- **Slot count must be a safe upper bound.** `ceil(N/64)` covers tile widths 64/128/256;
  unused slots stay zero. Guessing 256 and being wrong means out-of-bounds writes.

This preserves the gradient exactly in exact arithmetic — same terms, same values, only the
summation order is pinned. It fixes **level 3**. Level 2 additionally requires
`use_dynamic_sched=False` on the same kwargs dict (`grouped_mlp.py:1755`) so the tile→CTA
sequence, and therefore `acc_consumer_state.phase`, is reproducible. The two are
complementary; **neither is sufficient alone** — static scheduling tested by itself produced
no change, because the level-3 atomics dominated.

---

### VERIFIED RESULT — jobs 341581 (A) / 341491 (B)

Both order dependencies fixed together, gated on `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`:

| | jobs | divergent records |
|---|---|---|
| before (stock kernel) | 334628/9 | ~16,000 per rank |
| L3 only (slots, atomic removed) | 337061/3 | ~15,800 per rank — first mismatch moved onto the new `sum(dim=1)`, inputs differing |
| **L2 + L3** | **341581/341491** | **0 across 256/256 ranks, 13,706,980 records compared** |

Runtime assertions on BOTH arms:
`[det-dprob-check] shape=(182016, 20, 1) expected_n_tiles=20 occupied_slots=20 -> OK`
(20 = 5120/256; every tile landed in its own slot, confirming the layout at production shapes).

Numerically sane, not merely reproducible — lm loss vs the STOCK ATOMIC baseline (334628):

| iter | stock atomic | deterministic |
|---|---|---|
| 1 | 1.220116E+01 | 1.220107E+01 |
| 2 | 1.220136E+01 | 1.220141E+01 |
| 3 | 1.096610E+01 | 1.096596E+01 |

Agreement to ~1e-5 relative: the expected float-reassociation difference from summing the same
terms in a fixed order. A dropped or double-counted contribution would be orders larger. (Loss
values are corroboration only — the zero-divergence trace is the evidence.)

**Neither lever works alone.** Static scheduling alone: no change (L3 atomics dominate). Slots
alone: no change in total divergence, the first mismatch simply moves to the slot reduction
because each slot's own value still varies with traversal direction. Both together: clean.

## Source 3 — bf16 gradient reduce-scatter — **RETRACTED, it is a carrier**

This was reported as a source. It is not. The error is worth recording because it is easy
to repeat.

`param_and_grad_buffer.py:4092` records appeared to satisfy the root-cause test — local
input matched between arms, output differed. **But the test is invalid for collectives.**
The tracer hashes only *this rank's* `input` tensor (`_wrap_out_in`), while a reduce-scatter
output is a function of **every group member's** input. The correct test is:

```
origin  <=>  ALL ranks' inputs match  AND  output differs
```

Checked directly on the traces: for the first collective dp0 flagged as an origin
(`window=1, grp_sz64_0, reduce_scatter_tensor, align_idx=1`), **16 of the 256
participating ranks had a differing local input**. dp0's own input matched, so the
single-rank test called it a source; the output differed because sixteen peers fed it
different gradients — inherited from the `dprob` divergence upstream.

This also explains the incoherent counts (dp0/dp93): 12/18, 2/1, 19/26, 25/5. They track
*which peers happened to be contaminated*, not a property of the collective.

A second, independent reason to distrust these records: `_coalescing_manager` fuses many
bucket collectives into one grouped NCCL op, so chunking depends on the group's composition
rather than the single collective.

Consequence: the `NCCL_PROTO=Simple` + fixed-channel pins in `apply_determinism_overrides()`
rest on **no measurement**. The chunk-ordering mechanism is real in general; there is no
evidence it affects this recipe. Treat as unvalidated.

Note this reproduces the pre-existing same-node-control finding (542/544 grad reduce-scatter
records had differing inputs -> carrier, not source). That conclusion was correct and was
wrongly overturned by applying a single-rank test to a multi-rank operation.

**The flaw is specific to collectives.** An ATen op's inputs are all local, so
`inputs match AND output differs` remains valid for op-level records — which is what
sources 1 and 2 rest on.

---

## Tracer defects found (fixed) — these produced false root causes

Anyone repeating this analysis will hit these.

1. **`_IllegalWork` skipped the wait.** `param_and_grad_buffer.py:4707` passes `async_op=True`
   but receives an `_IllegalWork` sentinel; the tracer returned early without ordering, and
   fingerprinted a bucket `fetch_bucket()` had just allocated. That produced a 240-rank
   "genuine origin" that was uninitialized memory (bf16 `absmax = 2⁶⁵`). Fixed with a device
   sync fallback.
2. **Coalesced collectives hashed before launch.** `_coalescing_manager` only *captures*;
   the grouped NCCL op fires at `__exit__`. All four Megatron-FSDP paths (4038/4132/4542/4561)
   use it, so every FSDP collective record was meaningless. Fixed by deferring the output hash
   to manager exit — and patching `_coalescing_manager` on the **captured** reference, since
   `param_and_grad_buffer.py:33` binds it at import and patching `torch.distributed` alone
   misses it.
3. **Digest-only signatures.** Added `sum`/`absmax` moments so `|Δ|` is rankable; without
   them a 1-ULP difference and a wholly wrong tensor are indistinguishable.

## Artifact classes — filter these or the analysis lands on the wrong op

1. **int64 pointer arrays** (`grouped_mlp.py:1256`, `absmax` 1e12–1e15) — CUDA device
   addresses, differ between any two processes. Unfiltered they masquerade as the global
   first mismatch on 204/256 ranks.
2. **Uninitialized output buffers** — view ops on a freshly allocated buffer read garbage.
   Proven: the same 29,040,640-element e8m0 buffer DIFFERS at seq 1359 and MATCHES at
   1363/1368. `aten.empty` is skipped by the tracer but its *views* are not. Mitigate with
   `train.fill_uninitialized_memory=true` (`training/config.py:443`, default `True`; perf
   recipes set it `False`).
3. **A view op can never introduce divergence.** If the first mismatch is
   `permute`/`slice`/`view`/`as_strided`, it is inherited or a stale read — a contradiction
   to investigate, not a finding.

**Root-cause test:** `inputs MATCH ∧ output DIFFERS`. `diff_streams.py` does **not** apply
it — it prints "ROOT CAUSE" on output divergence alone, even while reporting
`inputs DIFFER`. Apply the filter manually.

---

## CHANGES REQUIRED FOR DETERMINISM (complete list)

Everything below is needed together. Removing any one of items 1-3 reintroduces divergence.

### 1. Megatron-Bridge — `src/megatron/bridge/recipes/utils/determinism_utils.py`

```python
cfg.model.moe_router_fusion = False    # source 1: TE fused_moe_aux_loss atomicAdd
```

Overbroad (also unfuses top-k routing, which was never implicated). Replace with the
upstream Megatron gate once it lands:
`fused=self.config.moe_router_fusion and not self.config.deterministic_mode`
in `_apply_aux_loss` / `_apply_seq_aux_loss` / `_apply_global_aux_loss`.

Already present and retained: `deterministic_mode=True`, `cross_entropy_loss_fusion=False`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`,
`MAMBA_DETERMINISTIC=1`, `tp_comm_overlap=False`.
`NCCL_PROTO=Simple` + `NCCL_MIN/MAX_NCHANNELS=4` are also set but are **UNVALIDATED** —
source 3 was retracted, so nothing measured supports them.

### 2. cuDNN kernel — `moe_blockscaled_grouped_gemm_dsrelu_quant.py` (level 3)

Replace the cross-CTA atomic with a single-writer per-N-tile slot store:

```python
if cutlass.const_expr(_DET_DETERMINISTIC):
    real_dprob[mPosition, epi_work_tile_info.tile_n_idx, 0] = dProbVal
else:
    _ = atomic_add_float32(ptr=real_dprob[(mPosition, None, None)].iterator.llvm_ptr,
                           value=dProbVal)
```

Correct because `mPosition = tile_m_idx * cta_tile_m + tidx` (line 2099): one thread owns
one `m`, so each `(m, tile_n)` has exactly one writer and no contribution is lost.

### 3. TE — `transformer_engine/pytorch/ops/fused/grouped_mlp.py` (levels 2 + 3)

```python
# L3: allocate slots, reduce in fixed order
_det_nslots = (fc1_weight_shape[0] + 255) // 256          # kernel asserts mma_tiler N == 256
dscales_tensor = torch.zeros((scales_tensor.shape[0], _det_nslots, 1),
                             dtype=torch.float32, device=scales_tensor.device)
...
grad_scales = fc2_dgrad_kernel_out["dprob_tensor"].sum(dim=1).view(-1)

# L2: fixed tile->CTA sequence => reproducible acc phase => fixed subtile traversal
"use_dynamic_sched": (False if _DET_STATIC_SCHED else True),   # dprob-carrying dict ONLY
```

Slots MUST live in dim 1: the MoE domain conversion preserves `shape[1]` but hardcodes dim 2
(`moe_sched_extension.py:164`, `c1 = Int32(1)`), so `(M, 1, NTILES)` is silently flattened.

### 4. cuDNN `api.py` — accept the wider tensor, and bound it

```python
_n_tiles = ceil_div(n_out, self.mma_tiler_mn[1])
if self.dprob_desc.shape[1] < _n_tiles:
    raise ValueError(f"dprob deterministic mode needs >= {_n_tiles} slots in dim 1 ...")
```

The stock assertion hardcoded `(tensor_m, 1, 1)`. Checking `shape[1]` against itself would be
tautological and would let an UNDER-allocated dprob through, after which the kernel writes out
of bounds — silent corruption, not a crash.

### Single switch

All three patches gate on TE's existing flag, so no new knob is introduced:

```python
_DET_DETERMINISTIC = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1") == "0"
```

This also closes the documented inconsistency: TE already refuses its *Triton* dbias/dscales
kernel under that flag ("uses non-deterministic atomic adds") while the cuDNN kernel computing
the same quantity ran unguarded.

### Verification built in

`_det_dprob_selfcheck` logs once per process on the real tensor:
`[det-dprob-check] shape=(182016, 20, 1) expected_n_tiles=20 occupied_slots=20 -> OK`.
Slots are single-writer and `tile_n_idx` spans `ceil(n_out/256)` tiles, so occupancy must equal
that count. This catches a misconfiguration that a determinism diff cannot: a wrong slot layout
still yields a perfectly *reproducible* answer.

### Evidence of bitwise alignment

- op/collective streams: **0 divergent records, 256/256 ranks, 13,706,980 records**
- wandb summary metrics compared as raw IEEE-754 bytes: **15/15 bitwise identical**, incl.
  `lm loss 10.965963363647461 (0x4025ee92c0000000)`,
  `grad-norm 2.3894100189208984 (0x40031d8300000000)`,
  `seq_load_balancing_loss 0.9616295099258423 (0x3feec5ab40000000)` — the metric that was the
  original smoking gun (9.616297E-01 vs 9.616296E-01).

---

## Recommended actions

1. **Megatron-LM** — gate aux-loss fusion on `deterministic_mode` (3 call sites), so
   `moe_router_fusion=False` is no longer needed and top-k routing keeps its fused kernel.
2. **cuDNN** — make the `dprob` reduction deterministic. The single-writer slot scheme above
   is a working design; per-CTA partials plus a fixed-order second pass is the general form.
3. **Transformer Engine** — until then, gate the cuDNN grouped-MLP path under
   `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, matching the existing Triton behaviour, so the flag
   stops silently under-delivering.
4. **Megatron-Bridge** — keep `moe_router_fusion=False`. The NCCL pins rest on no
   measurement (source 3 retracted); either drop them or justify them independently.
5. **Determinism tooling** — the root-cause test for a COLLECTIVE must require group-wide
   input agreement, not this rank's. A single-rank test manufactures origins for any rank
   downstream of a contaminated peer, and that is how source 3 was wrongly promoted.
