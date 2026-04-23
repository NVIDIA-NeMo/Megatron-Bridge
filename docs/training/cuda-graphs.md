# CUDA Graphs

CUDA graphs capture a sequence of GPU operations once and replay them with
minimal host overhead, reducing repeated kernel-launch and driver costs on
every training step.

This page is the stable guide for what CUDA graphs are, when they help, and
what tradeoffs to expect. For exact enablement knobs, code anchors, and
verification commands, see
[skills/perf-techniques/cuda-graphs/SKILL.md](../skills/perf-techniques/cuda-graphs/SKILL.md).

## What It Is

CUDA graphs record a fixed sequence of GPU work during a capture phase and then
replay that sequence on later steps. The main benefit is lower host-side
launch overhead.

Megatron Bridge supports two capture implementations:

| `cuda_graph_impl` | Mechanism | Scope support |
|---|---|---|
| `"local"` | MCore `CudaGraphManager` / `FullCudaGraphWrapper` | `full_iteration` |
| `"transformer_engine"` | TE `make_graphed_callables()` per layer | `attn`, `mlp`, `moe`, `moe_router`, `moe_preprocess`, `mamba` |
| `"none"` (default) | Disabled | — |

`"local"` captures the whole forward-backward iteration. `"transformer_engine"`
captures selected submodules and is usually the more flexible default path.

## What Problem It Solves

CUDA graphs mainly solve launch-bound training steps where GPU compute is fast
enough that repeated host-driver submission overhead becomes noticeable.

This is most useful when:

- tensor shapes are static across steps
- the workload has high step frequency or relatively small kernels
- the run has enough memory headroom to keep graph buffers resident

It is less about changing the math and more about reducing runtime overhead.

## Impacted Training Dimensions

| Dimension | Effect | Confidence | Why |
|---|---|---|---|
| `speed` | ~10-30% faster step time | medium | Replays pre-captured GPU work and reduces launch overhead. The gain is biggest when the run is visibly launch-bound. |
| `memory` | near-neutral to several GB higher, depending on scope | high | Graph buffers stay allocated for replay. TE-scoped paths can be modest, while larger models or deeper PP can make memory noticeably tighter. |
| `scale` | neutral to slightly positive | low | Can help at scale if host overhead matters, but extra memory residency can also gate larger configs. |
| `convergence` | no change expected | medium | Intended to preserve training math when capture constraints are satisfied. |
| `stability` | adds operational constraints | medium | Requires static shapes, specific RNG or NaN settings, and compatible scope selections. Failure modes are well-defined but add surface area. |

## When to Use It

Enable CUDA graphs when all of the following are mostly true:

- sequence length and micro-batch size are static
- host overhead is a meaningful part of step time
- the run has spare memory budget
- you want throughput improvement without changing the training objective

As a rule of thumb:

- prefer `transformer_engine` scoped graphs for the safer first rollout
- use `local` `full_iteration` graphs only when you specifically want the
  largest launch-overhead reduction and can accept the stricter constraints

## When Not to Use It

Avoid CUDA graphs when any of these are true:

- sequence length or batch shapes vary step to step
- CPU offloading is enabled
- memory is already tight, especially with `PP > 1`
- you rely on runtime checks that conflict with `full_iteration` capture
- you need unsupported scope combinations for MoE or recompute paths
- SFT/LoRA with packed sequences (`packed_sequence=True`) — TE-scoped graphs
  cannot capture `packed_seq_params` (non-Tensor input)
- full activation recompute (`recompute_granularity=full`) with TE-scoped
  graphs — only `local` full-iteration graphs support full recompute

## Feature Interactions

The most important interactions are:

- `use_te_rng_tracker` and `rng.te_rng_tracker`: required when CUDA graphs are enabled
- `rerun_state_machine.check_for_nan_in_loss`: must be disabled for `local` + `full_iteration`
- MoE routing scopes: `moe` and `moe_router` are mutually exclusive
- `moe_preprocess`: requires `moe_router`
- `delay_wgrad_compute`: adds extra constraints when captured scopes include attention or MoE router
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: requires `NCCL_GRAPH_REGISTER=0` in the relevant path
- CPU offloading: incompatible

These interactions are stable enough to treat as design constraints, not just
debugging tips.

## Bridge Configuration

Configure CUDA graphs through:

- `model.cuda_graph_impl`
- `model.cuda_graph_scope`
- `model.cuda_graph_warmup_steps`
- `model.use_te_rng_tracker`
- `rng.te_rng_tracker`

If you choose `local` with `full_iteration`, disable the loss and gradient NaN
checks that conflict with full capture.

For exact config snippets and runnable commands, see
[skills/perf-techniques/cuda-graphs/SKILL.md](../skills/perf-techniques/cuda-graphs/SKILL.md).

## Minimal Runnable Example

For a minimal Bridge-facing example, start from the functional smoke test:

- `tests/functional_tests/recipes/test_llama_recipes_pretrain_cuda_graphs.py`

For a lightweight CLI-driven path, use the performance harness with scoped
capture and a small model recipe.

## Expected Metric Changes

| Metric | Expected Change | Conditions | Evidence |
|---|---|---|---|
| `step_time` | ~10-25% down | Static shapes, launch-bound training, especially TE-scoped MoE paths | measured |
| `tokens_per_sec` | ~10-30% up | Same as above | measured |
| `peak_memory` | flat to moderately higher | TE-scoped paths with headroom | measured |
| `OOM risk` | up | Tight memory budget or large MoE configs | measured |

Do not assume a fixed throughput gain across models. The improvement depends on
how launch-bound the workload is and how much scope is captured.

## Representative Validation Patterns

### Mid-sized MoE pretrain

On mid-sized MoE pretrain runs with TE-scoped graphs
(`attn + moe_router + moe_preprocess`), the common pattern is:

- low-teens to low-20s percent faster step time
- corresponding throughput gains when the eager baseline is launch-bound
- short-run loss behavior that stays close to baseline
- little or no obvious memory penalty in the friendliest TE-scoped cases

### Packed-sequence SFT and LoRA

Packed-sequence finetuning remains sensitive:

- TE-scoped graphs can fail if non-Tensor packed-sequence arguments reach the
  captured path
- some failures are environment or container blockers rather than graph-specific
- treat packed-sequence plus CUDA graphs as a separate validation target, not as
  something automatically inherited from pretrain success

### Larger MoE pretrain

Larger MoE runs can become memory-gated before graph replay pays off:

- a run that barely fits in eager mode may OOM once graph buffers stay resident
- this is especially common with large MoE models, deeper PP, or already-tight
  HBM headroom
- treat CUDA graphs as a throughput optimization for runs with margin, not as a
  fit-enabling technique

## Optimizer CUDA graph

The `OptimizerCudaGraphWrapper` is an experimental utility that enables CUDA graph capture of the ADAM optimizer step. By encapsulating optimizer updates into a replayable CUDA graph, it can further reduce host launch overhead, especially when optimizer has additional CPU overhead such as additional cast when using low-precision native parameters. This feature can be enabled by passing `optimizer_cuda_graph=True` to `OptimizerConfig`. Avoid using it if your optimizer step is includes conditional logic, or uses operations that are not graph-compatible, as these may cause capture or replay failures.

## Common Failure Modes

- Missing TE RNG tracker settings causes an assertion before training starts.
- Dynamic sequence or batch shapes break capture or replay assumptions.
- `local` `full_iteration` graphs fail when NaN-loss checking is still enabled.
- Illegal scope combinations such as `moe` with `moe_router` fail validation.
- Runs that fit in eager mode can OOM after enabling graphs because buffers stay pinned.
- Full activation recompute (`recompute_granularity=full`) with TE-scoped graphs
  asserts: `full recompute is only supported with full iteration CUDA graph`.
  Disable recompute or switch to `local` implementation.
- Packed-sequence SFT/LoRA asserts: `CUDA graph accepts only Tensor inputs.
  inference_context and packed_seq_params are excluded from input list.`
  TE-scoped graphs cannot capture non-Tensor forward arguments.
- Older TE/container builds can fail packed-sequence attention before graph
  capture begins (`Available backends = {FlashAttention=False,
  FusedAttention=False, UnfusedDotProductAttention=False}`). In that case the
  baseline and graph runs are both blocked, so fix the environment first.

## Related Docs

- [Performance Guide](../performance-guide.md)
- [Communication Overlap](communication-overlap.md)
- [skills/perf-techniques/cuda-graphs/SKILL.md](../skills/perf-techniques/cuda-graphs/SKILL.md)
