# Root Cause: HybridEP Non-Determinism with Virtual Pipeline Parallelism

**Date**: 2026-03-24
**Config**: DeepSeek-V3, PP=2, VP=2, EP=8, 16×GB200

---

## Summary

Training with HybridEP (`moe_flex_dispatcher_backend=hybridep`) is non-deterministic when
Virtual Pipeline Parallelism (VP≥2) is enabled. The root cause is a **shared `_hybrid_ep_buffer`
global singleton whose internal state is mutated by backward-pass all-to-all ops while
forward-pass all-to-all ops for later microbatches are executing on the same buffer** — a
pattern that only arises in the interleaved 1F1B schedule.

---

## Isolation Experiments

### Factor isolation: HybridEP and VP=2 must both be present

The first set of experiments vary the dispatcher and VP independently:

| HybridEP? | VP=2? | Deterministic? |
|---|---|---|
| No (alltoall) | No | **Yes** |
| No (alltoall) | Yes | **Yes** |
| Yes | No | **Yes** |
| Yes | Yes | **No** — diverges by step 3–5 |

Neither factor alone causes non-determinism. VP=2 with alltoall is fully deterministic.
HybridEP with VP=None is fully deterministic. Only the combination breaks it.

This rules out VP=2's interleaved schedule itself as the cause, and rules out HybridEP's
all-to-all communication as the cause. The bug must be in the **interaction** between the
two — specifically, the way VP=2's schedule causes the shared `_hybrid_ep_buffer` to be
accessed by interleaved backward and forward calls simultaneously.

### Overlap isolation: GBS=16 removes backward/forward interleaving

The second experiment keeps HybridEP + VP=2 but eliminates the backward/forward overlap
by reducing GBS:

| Config | GBS | Microbatches/step | Deterministic? |
|---|---|---|---|
| PP=2, VP=2, HybridEP | 256 | 32 (steady-state 1F1B overlap) | **No** — diverges by step 3–5 |
| PP=2, VP=2, HybridEP | 16 | 2 (= warmup count, no overlap) | **Yes** — bit-identical × 50 steps |

With PP=2, VP=2 and DP=8:
- `num_microbatches = GBS / (MBS × DP) = 16 / 8 = 2`
- `warmup count = (PP − 1) × num_model_chunks = 1 × 2 = 2`
- All 2 microbatches are consumed in the warmup phase; the step has zero steady-state 1F1B
  iterations, so no backward ever overlaps a forward on the buffer.

This run is **bit-identical** across 50 training steps and all router weight gradients.
HybridEP is still active. VP=2 is still active. The only thing removed is the
backward/forward overlap on the buffer. The result is deterministic — confirming the
overlap is the sole trigger.

---

## Mechanism

### The global buffer singleton

`fused_a2a.py` holds a single module-level `_hybrid_ep_buffer`:

```python
_hybrid_ep_buffer = None          # module-level singleton

def init_hybrid_ep_buffer(...):
    global _hybrid_ep_buffer
    _hybrid_ep_buffer = HybridEPBuffer(...)
```

Every call to `HybridEPDispatch.forward` and `HybridEPCombine.forward` — and their
corresponding backward passes — all operate on this **same object**.

### VP=None: calls are strictly sequential

With VP=None, Megatron's pipeline schedules process all microbatches for a given layer
sequentially: complete all forward passes, then complete all backward passes. The call
sequence on `_hybrid_ep_buffer` for one training step looks like:

```
F_dispatch(mb=0) → F_combine(mb=0) → ... → F_dispatch(mb=N) → F_combine(mb=N)
→ B_combine_backward(mb=N) → B_dispatch_backward(mb=N) → ... → B_combine_backward(mb=0)
```

No backward call on `_hybrid_ep_buffer` ever interleaves with a forward call.

### VP=2: backward and forward calls interleave on the same buffer

With VP=2 and GBS=256 (32 microbatches), Megatron's interleaved 1F1B schedule enters
its **steady state** after 2 warmup microbatches. In steady state, for each new forward
microbatch there is also a backward microbatch running for an earlier microbatch. Since
both VP chunks share `_hybrid_ep_buffer`, the call sequence becomes:

```
F_dispatch(mb=0,chunk=0) → F_dispatch(mb=0,chunk=1) → [warmup end]
F_dispatch(mb=1,chunk=0) + B_combine_bwd(mb=0,chunk=1)   ← same buffer, concurrent
F_dispatch(mb=1,chunk=1) + B_dispatch_bwd(mb=0,chunk=1)  ← same buffer, concurrent
...
```

`HybridEPBuffer` maintains internal state across calls (token-count metadata, RDMA/NVLink
buffer bookkeeping, routing handles). When a backward `combine_with_unpermute` or
`dispatch_with_permute` modifies buffer state while a forward `dispatch_with_permute` is
executing, the forward sees a partially-updated state. The exact interleaving of CUDA
kernel launches from these Python-level calls is non-deterministic across runs, producing
different routing outcomes from the same input — starting from `tokens_per_expert` at step 1
and amplifying through `expert_bias` updates into visible loss divergence by step 3–5.

### Why it is subtle

- The non-determinism is sub-token (383 vs 384 tokens differ out of ~250k in step 1).
- Router *inputs* are bit-identical; the divergence is in the buffer-mediated all-to-all
  result, not in the gating logits.
- `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` do not cover
  `HybridEPBuffer` internals.
- Adding `torch.cuda.synchronize()` before dispatch does not help: the issue is buffer
  state mutation, not stream ordering of CUDA kernels visible to PyTorch.

---

## Fix

The fix is to **give each MoE layer its own `HybridEPBuffer` instance** by keying the
module-level buffer dict on the Python identity of the `_HybridEPManager` instance that
owns it (`id(manager)`).

### Why per-manager, not per-VP-rank

A first attempt keyed on `parallel_state.get_virtual_pipeline_model_parallel_rank()`.
This is the intuitive choice — VP chunk 0 gets buffer 0, chunk 1 gets buffer 1 — but it
does not work:

- `set_virtual_pipeline_model_parallel_rank` is **deprecated** in Megatron-LM (emits a
  `DeprecationWarning`). The pipeline schedule **never calls it**. As a result,
  `get_virtual_pipeline_model_parallel_rank()` always returns `None` during training,
  making the keyed dict reduce to a single entry (key 0) shared by every VP chunk —
  identical behaviour to the original broken singleton.

### Why per-manager works

Each MoE layer creates exactly one `_HybridEPManager` at model-construction time.
VP chunk 0's layers and VP chunk 1's layers are distinct Python objects with distinct
`id()` values. Storing one `HybridEPBuffer` per `id(manager)` therefore gives each VP
chunk's MoE layers an exclusive buffer with no cross-chunk aliasing, regardless of what
the pipeline schedule does or does not set in `parallel_state`.

### Implementation

**`fused_a2a.py`**: Replace the module-level `_hybrid_ep_buffer` singleton with a
`dict[int, HybridEPBuffer]`. Thread a `buffer_key: int` parameter through
`hybrid_ep_dispatch` → `HybridEPDispatch.apply/.forward`,
`hybrid_ep_combine` → `HybridEPCombine.apply/.forward`,
and `init_hybrid_ep_buffer`. The backward passes are unaffected: they already capture
`ctx.buffer` at forward time and never touch the global dict.

**`token_dispatcher.py`**: In `_HybridEPManager.__init__`, assign
`self._buffer_key = id(self)`. Pass `buffer_key=self._buffer_key` in `dispatch()` and
`combine()`.

Memory: one `HybridEPBuffer` per MoE layer (same as before per VP chunk). With VP=2 and
N MoE layers, there are 2N buffers total — one per layer per chunk — versus N with
VP=None. This is unavoidable; any sharing between chunks recreates the aliasing bug.

---

## Affected Configs

Any training run with **both** of the following:
- `moe_flex_dispatcher_backend=hybridep` (or any backend backed by `HybridEPBuffer`)
- Virtual Pipeline Parallelism enabled (`vp ≥ 2`)

PP=1 and VP=None are not affected. The alltoall dispatcher is not affected.
