# CUDA Graphs

CUDA graphs capture a sequence of GPU operations once and replay them with
minimal host overhead, reducing repeated kernel-launch and driver costs on
every training step.

This page is the stable guide for what CUDA graphs are, when they help, and
what tradeoffs to expect. For exact enablement knobs, code anchors, and
verification commands, see `skills/perf-techniques/cuda-graphs/SKILL.md`.

## What It Is

CUDA graphs record a fixed sequence of GPU work during a capture phase and then
replay that sequence on later steps. The main benefit is lower host-side
launch overhead.

Megatron Bridge supports two capture implementations:

| `cuda_graph_impl` | Mechanism | Scope support |
|---|---|---|
| `"local"` | MCore `CudaGraphManager` / `FullCudaGraphWrapper` | `full_iteration` |
| `"transformer_engine"` | TE `make_graphed_callables()` per layer | `attn`, `mlp`, `moe`, `moe_router`, `moe_preprocess`, `mamba` |
| `"none"` | Disabled | — |

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
| `speed` | ~15-30% faster step time | medium | Replays pre-captured GPU work and reduces launch overhead. Measured 16-24% on GPT-OSS-20B and 22% on Qwen3-30B-A3B with TE-scoped graphs. Gain depends on how launch-bound the workload is. |
| `memory` | ~0-2 GB extra (TE scoped); 10 GB+ possible with `PP > 1` or large MoE | high | Graph buffers stay allocated for replay. TE-scoped showed no measurable increase on 20B/30B models but OOM'd on 120B at 70/79 GB. |
| `scale` | neutral to slightly positive | low | Can help at scale if launch overhead matters, but memory overhead can gate larger configs (e.g., GPT-OSS-120B OOM). |
| `convergence` | no change expected | medium | Intended to preserve training math when capture constraints are satisfied. Loss matched within 0.001 on Qwen3-30B-A3B over 20 iterations. |
| `stability` | adds operational constraints | medium | Requires static shapes, specific RNG/NaN settings, and compatible scope selections. Failure modes are well-defined but add surface area. |

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

Minimal high-level configuration:

```python
cfg.model.cuda_graph_impl = "transformer_engine"   # or "local"
cfg.model.cuda_graph_scope = ["attn"]              # or other valid scopes
cfg.model.cuda_graph_warmup_steps = 3
cfg.model.use_te_rng_tracker = True
cfg.rng.te_rng_tracker = True
```

If you use `local` + `full_iteration`, also disable:

```python
cfg.rerun_state_machine.check_for_nan_in_loss = False
cfg.ddp.check_for_nan_in_grad = False
```

## Minimal Runnable Example

For a minimal Bridge-facing example, start from the functional smoke test:

- `tests/functional_tests/recipes/test_llama_recipes_pretrain_cuda_graphs.py`

For a lightweight CLI-driven path, use the performance harness with scoped
capture and a small model recipe.

## Expected Metric Changes

| Metric | Expected Change | Conditions | Evidence |
|---|---|---|---|
| `step_time` | ~15-25% down | Static shapes, MoE, TE scoped (`attn+moe_router+moe_preprocess`) | measured: Qwen3-30B-A3B 623→484ms; GPT-OSS-20B 467-520→391-399ms |
| `tokens_per_sec` | ~20-33% up | Same as above | measured: Qwen3-30B-A3B 214→274 TFLOP/s/GPU; GPT-OSS-20B 37.9-42.2→49.4-50.4 |
| `peak_memory` | same pre-capture | TE scoped graphs on H100 80GB | measured: no increase in allocated memory on Qwen3-30B-A3B and GPT-OSS-20B |
| `OOM risk` | up | Tight memory budget or large MoE configs | measured: GPT-OSS-120B blocked at ~70/79 GB before capture |

Do not assume a fixed throughput gain across models. The improvement depends on
how launch-bound the workload is and how much scope is captured.

## Measured Results (Qwen3-30B-A3B MoE, H100, TP2 PP2 EP4, 2 nodes)

### Pretrain

TE-scoped CUDA graphs (`attn + moe_router + moe_preprocess`) on Qwen3-30B-A3B
with mock data, GBS=8, MBS=1:

- **~22% faster** iteration time (484ms vs 623ms steady-state)
- **~28% higher TFLOP/s** (274 vs 214 TFLOP/s/GPU)
- **Loss matches** baseline within 0.001 across all 20 iterations
- 24 graphable layers per pipeline rank, capture completes in ~5.6s
- No memory increase pre-capture, no NCCL errors

### SFT (packed sequences)

SFT with packed sequences (`packed_sequence=True`, SQuAD dataset) hits a
hard incompatibility:

```
AssertionError: CUDA graph accepts only Tensor inputs.
inference_context and packed_seq_params are excluded from input list.
```

TE-scoped CUDA graphs require all forward inputs to be Tensors. Packed
sequence SFT passes `packed_seq_params` (a dataclass), which is not captured.
The baseline SFT runs fine without graphs (~880ms/iter).

Workarounds: disable packing, or use `local` full-iteration graphs. Also make
sure the TE/container build actually supports the packed-sequence attention
backend your recipe needs.

## Additional Validation (GPT-OSS, H100, Mar 2026)

### GPT-OSS-20B pretrain

TE-scoped CUDA graphs on `gpt-oss-20b` with `TP2 PP4 EP4 CP1`, 2 nodes, and
mock data:

- capture succeeds with 6 graphable layers per pipeline rank; capture completes
  in ~0.95s
- steady-state iteration time improves by ~16-24% (467-520ms to 391-399ms)
- throughput improves by ~19-33% (37.9-42.2 to 49.4-50.4 TFLOP/s/GPU)
- the pre-capture memory report is unchanged and the 20-iteration run completes
  without NCCL or illegal-memory-access errors
- loss comparison is inconclusive: the first ~10 post-capture iterations are
  close, but the run used mock data, `GBS=4`, and a production LR, so later
  divergence is too noisy to treat as a correctness signal

A cleaner loss-match pass should lower LR and/or raise GBS before drawing
equivalence conclusions.

### GPT-OSS-20B SFT and LoRA

Both packed-sequence finetuning workloads were blocked in the
`mbridge-260128.sqsh` container before any CUDA-graph-specific behavior could
be isolated:

- baseline and graphed runs both fail with no TE attention backend available
  for the packed-sequence path
- treat this as an environment/container blocker first, not as proof that CUDA
  graphs are or are not the root cause
- after upgrading TE/container support, these workloads still need separate
  validation because packed-sequence plus TE-scoped graphs remains a sensitive
  combination

### GPT-OSS-120B pretrain

`gpt-oss-120b` pretrain at `TP2 PP4 EP8`, 4 nodes, hits OOM on iteration 2:

- iteration 1 already uses ~69-70 GB allocated and ~72-73 GB reserved on 79 GB
  H100s
- the failure is a `torch.OutOfMemoryError` on an additional 1.54 GiB
  allocation
- treat larger MoE rollouts as memory-gated even before capture benefits are
  realized; more PP or different memory settings may be needed

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
- `skills/perf-techniques/cuda-graphs/SKILL.md`
