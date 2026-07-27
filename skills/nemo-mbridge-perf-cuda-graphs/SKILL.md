---
name: nemo-mbridge-perf-cuda-graphs
description: Validate and use CUDA graph capture in Megatron Bridge, including local full-iteration graphs and Transformer Engine scoped graphs for attention, MLP, and MoE modules. Use for host-driver-overhead reduction, CUDA-graph crash or regression investigation, cuda_graph_impl, full-iteration graphs, TE scoped graphs, graphed callables, or CUDA graph capture.
license: Apache-2.0
---

# CUDA Graphs

Stable documentation: @docs/training/cuda-graphs.md
Card: @skills/nemo-mbridge-perf-cuda-graphs/card.yaml

## What It Is

CUDA graphs capture GPU operations once and replay them with minimal
host-driver overhead. Bridge supports two implementations:

| `cuda_graph_impl` | Mechanism | Scope support |
|---|---|---|
| `"local"` | MCore `FullCudaGraphWrapper` wrapping entire fwd+bwd | `full_iteration` |
| `"transformer_engine"` | TE `make_graphed_callables()` per layer | `attn`, `mlp`, `moe`, `moe_router`, `moe_preprocess`, `mamba` |

## Quick Decision

Start with TE-scoped graphs for most training workloads, then verify replay
timing against eager on the same dispatcher, layout, and container:

- dense models: `attn`, then optionally `mlp`
- dropless MoE: `attn moe_router moe_preprocess`
- VLMs: the same dropless-MoE scope, but only after the real-data path is stable

Use `local` + `full_iteration` only when you specifically want full-iteration
capture and can satisfy the tighter constraints.

For recompute-heavy workloads:

- TE-scoped graphs pair naturally with selective recompute
- full recompute usually pushes you toward `local` full-iteration graphs or away
  from graphs entirely

Related docs:

- @docs/training/cuda-graphs.md
- @docs/training/activation-recomputation.md

## Enablement

### Local full-iteration graph

```python
cfg.model.cuda_graph_impl = "local"
cfg.model.cuda_graph_scope = ["full_iteration"]
cfg.model.cuda_graph_warmup_steps = 3
cfg.model.use_te_rng_tracker = True
cfg.rng.te_rng_tracker = True
cfg.rerun_state_machine.check_for_nan_in_loss = False
cfg.ddp.check_for_nan_in_grad = False
```

### TE scoped graph (dense model)

```python
cfg.model.cuda_graph_impl = "transformer_engine"
cfg.model.cuda_graph_scope = ["attn"]           # or ["attn", "mlp"]
cfg.model.cuda_graph_warmup_steps = 3
cfg.model.use_te_rng_tracker = True
cfg.rng.te_rng_tracker = True
```

### TE scoped graph (MoE model)

```python
cfg.model.cuda_graph_impl = "transformer_engine"
cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
cfg.model.cuda_graph_warmup_steps = 3
cfg.model.use_te_rng_tracker = True
cfg.rng.te_rng_tracker = True
```

### Performance harness CLI

```bash
uv run python scripts/performance/run_script.py \
  -m qwen \
  -mr qwen3_30b_a3b \
  --task pretrain \
  -g h100 \
  -c bf16 \
  -ng 16 \
  --cuda_graph_impl transformer_engine \
  --cuda_graph_scope attn,moe_router,moe_preprocess \
  ...
```

Valid CLI values live in `scripts/performance/argument_parser.py`:
- `VALID_CUDA_GRAPH_IMPLS`: `["none", "local", "transformer_engine"]`
- `VALID_CUDA_GRAPH_SCOPES`: `["full_iteration", "attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba"]`

The performance harness uses a comma-separated `--cuda_graph_scope` value and
auto-enables `model.use_te_rng_tracker` plus `rng.te_rng_tracker` when
`--cuda_graph_impl` is not `none`.

### Required constraints

- `use_te_rng_tracker = True` (enforced in `gpt_provider.py`)
- `full_iteration` scope only with `cuda_graph_impl = "local"`
- `full_iteration` scope requires `check_for_nan_in_loss = False`
- Do not combine `moe` scope and `moe_router` scope
- Tensor shapes must be static (fixed seq_length, fixed micro_batch_size)
- Dropless MoE capture requires static/capturable layer inputs; scope labels
  may select an entire MoE TransformerLayer rather than only dense submodules
- With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set
  `NCCL_GRAPH_REGISTER=0` (MCore enforces for local impl on arch < sm_100;
  TE impl asserts unconditionally)
- CPU offloading is incompatible with CUDA graphs
- `moe_preprocess` scope requires `moe_router` scope to also be set

Dropless Hopper MoE still needs a runtime proof before attempting
`full_iteration`. On the measured Qwen3.5-35B-A3B H100 stack, dynamic
`tokens_per_expert` handling retained a host synchronization, while the static
rank-capacity route required an sm100-only Transformer Engine operation-fuser
GroupedMLP implementation on the public/native path. Do not bypass the
architecture guard; use TE-scoped capture only after confirming the selected
layer set and expert-shape contract on Hopper.

The TE scope names select graphable layer objects in current MCore; they do not
necessarily wrap only the named kernel. `_layer_is_graphable()` selects an
entire `TransformerLayer` for `moe`, `moe_router`, or `moe_preprocess` when
`layer.mlp` is a `MoELayer`. On an all-MoE model (`moe_layer_freq=1`), a
router/preprocess-only list can therefore graph every transformer layer. Print
the reported graphable-layer count and inspect the model's MoE frequency before
calling a shorter scope an isolated router A/B.

### Practical bring-up order

1. Stabilize the eager run first.
2. Fix sequence length and micro-batch size.
3. Finish non-graph JIT compilation first. Inductor, Triton, or TileLang
   compilation is distinct from CUDA-graph warmup/capture and can dominate the
   first iteration. Persist the relevant compiler caches on a mounted path.
4. Enable the narrowest useful graph scope.
5. Confirm replay is active and memory is still acceptable.
6. Compare eager against graph replay iterations after warmup and capture; do
   not include the capture step in steady-state timing.
7. Only then widen scope or combine with overlap features.

Treat benchmark startup as three separate phases:

```text
kernel JIT/cache warmup -> CUDA graph warmup and capture -> steady replay
```

Do not infer a graph regression from the first step of a cold kernel cache, and
do not count either compilation or capture in the steady-state window.

## Code Anchors

### Bridge config and validation

```1524:1531:src/megatron/bridge/training/config.py
        # CUDA graph scope validation: check_for_nan_in_loss must be disabled with full_iteration graph
        if self.model.cuda_graph_impl == "local" and CudaGraphScope.full_iteration in self.model.cuda_graph_scope:
            assert not self.rerun_state_machine.check_for_nan_in_loss, (
                "check_for_nan_in_loss must be disabled when using full_iteration CUDA graph. "
                "Set rerun_state_machine.check_for_nan_in_loss=False."
            )
        if self.model.cuda_graph_impl == "none":
            self.model.cuda_graph_scope = []
```

### TE RNG tracker requirement

```213:216:src/megatron/bridge/models/gpt_provider.py
        if self.cuda_graph_impl != "none":
            assert getattr(self, "use_te_rng_tracker", False), (
                "Transformer engine's RNG tracker is required for cudagraphs, it can be "
                "enabled with use_te_rng_tracker=True'."
```

### Graph creation and capture in training loop

```231:255:src/megatron/bridge/training/train.py
    # Capture CUDA Graphs.
    cuda_graph_helper = None
    if model_config.cuda_graph_impl == "transformer_engine":
        cuda_graph_helper = TECudaGraphHelper(...)
    # ...
    if config.model.cuda_graph_impl == "local" and CudaGraphScope.full_iteration in config.model.cuda_graph_scope:
        forward_backward_func = FullCudaGraphWrapper(
            forward_backward_func, cuda_graph_warmup_steps=config.model.cuda_graph_warmup_steps
        )
```

### TE graph capture after warmup

```338:350:src/megatron/bridge/training/train.py
        # Capture CUDA Graphs after warmup.
        if (
            model_config.cuda_graph_impl == "transformer_engine"
            and cuda_graph_helper is not None
            and not cuda_graph_helper.graphs_created()
            and global_state.train_state.step - start_iteration == model_config.cuda_graph_warmup_steps
        ):
            if model_config.cuda_graph_warmup_steps > 0 and should_toggle_forward_pre_hook:
                disable_forward_pre_hook(model, param_sync=False)
            cuda_graph_helper.create_cudagraphs()
            if model_config.cuda_graph_warmup_steps > 0 and should_toggle_forward_pre_hook:
                enable_forward_pre_hook(model)
                cuda_graph_helper.cuda_graph_set_manual_hooks()
```

### RNG initialization

```199:206:src/megatron/bridge/training/initialize.py
        _set_random_seed(
            rng_config.seed,
            rng_config.data_parallel_random_init,
            rng_config.te_rng_tracker,
            rng_config.inference_rng_tracker,
            use_cudagraphable_rng=(model_config.cuda_graph_impl != "none"),
            pg_collection=pg_collection,
        )
```

### Delayed wgrad + CUDA graph interaction

```522:555:src/megatron/bridge/training/comm_overlap.py
            cuda_graph_scope = getattr(model_cfg, "cuda_graph_scope", []) or []
            # ... scope parsing ...
            if wgrad_in_graph_scope:
                assert is_te_min_version("2.12.0"), ...
                assert model_cfg.gradient_accumulation_fusion, ...
                if attn_scope_enabled:
                    assert not model_cfg.add_bias_linear and not model_cfg.add_qkv_bias, ...
```

### Perf harness override helper

```102:124:scripts/performance/utils/overrides.py
def _set_cuda_graph_overrides(
    recipe, cuda_graph_impl=None, cuda_graph_scope=None
):
    # Sets impl, scope, and auto-enables te_rng_tracker
```

### Graph cleanup

```1414:1441:src/megatron/bridge/training/train.py
def _delete_cuda_graphs(cuda_graph_helper):
    # Deletes FullCudaGraphWrapper and TE graph objects to free NCCL buffers
```

### MCore classes (in 3rdparty/Megatron-LM)

- `CudaGraphManager`: `megatron/core/transformer/cuda_graphs.py`
- `TECudaGraphHelper`: `megatron/core/transformer/cuda_graphs.py`
- `FullCudaGraphWrapper`: `megatron/core/full_cuda_graph.py`
- `CudaGraphScope` enum: `megatron/core/transformer/enums.py`

### Positive recipe anchors

- `src/megatron/bridge/perf_recipes/deepseek/gb300/deepseek_v3.py`
- `src/megatron/bridge/perf_recipes/qwen/gb300/qwen3_moe.py`
- `src/megatron/bridge/perf_recipes/gpt_oss/gb300/gpt_oss.py`

### Tests

| File | Coverage |
|---|---|
| `tests/unit_tests/training/test_config.py` | `full_iteration` NaN-check constraint |
| `tests/unit_tests/training/test_comm_overlap.py` | `delay_wgrad` + CUDA graph interaction |
| `tests/unit_tests/models/test_gpt_full_te_layer_autocast_spec.py` | TE autocast with CUDA graphs |
| `tests/functional_tests/test_groups/recipes/test_llama_recipes_pretrain_cuda_graphs.py` | End-to-end local and TE graph smoke tests |
| `tests/unit_tests/recipes/kimi/test_kimi_k2.py` | TE + CUDA graph recipe config |
| `tests/unit_tests/recipes/gpt/test_gpt3_175b.py` | TE + CUDA graph recipe config |
| `tests/unit_tests/recipes/qwen_vl/test_qwen25_vl_recipes.py` | VLM CUDA graph settings |

## Pitfalls

1. **TE RNG tracker is mandatory**: Setting `cuda_graph_impl` without
   `use_te_rng_tracker=True` and `rng.te_rng_tracker=True` will assert
   in the provider.

2. **`full_iteration` requires NaN checks disabled**: The entire fwd+bwd is
   captured, so loss-NaN checking cannot inspect intermediate values.

3. **MoE scope restrictions**: `moe` scope and `moe_router` scope are
   mutually exclusive. Token-dropless MoE can only graph `moe_router` and
   `moe_preprocess`, not the full expert dispatch.

4. **Memory overhead**: CUDA graphs pin all intermediate buffers for the
   graph's lifetime (no memory reuse). TE scoped graphs add a few GB;
   full-iteration graphs can increase peak memory by 1.5–2×. `PP > 1`
   compounds overhead since each stage holds its own graph.

5. **Delayed wgrad interaction**: When `delay_wgrad_compute=True` and
   attention or MoE router is in `cuda_graph_scope`, additional constraints
   apply: TE >= 2.12.0, `gradient_accumulation_fusion=True`, and no
   attention bias.

6. **Variable-length sequences break graphs**: Sequence lengths must be
   constant across steps. Use padded packed sequences if packing is needed.

7. **Graph cleanup is required**: CUDA graph objects hold NCCL buffer
   references. Bridge handles this in `_delete_cuda_graphs()` at the end
   of training, but early exits must call it explicitly.

8. **Older GPU architectures**: On GPUs with compute capability < 10.0
   (pre-Blackwell), set `NCCL_GRAPH_REGISTER=0` when using
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Enforced in MCore
   `CudaGraphManager` (cuda_graphs.py:1428) and `TECudaGraphHelper`
   (cuda_graphs.py:1697). The TE impl asserts unconditionally regardless
   of arch.

9. **CPU offloading incompatible**: CUDA graphs cannot be used with CPU
   offloading. Enforced in MCore `transformer_config.py:1907`.

10. **MoE recompute + moe_router scope**: MoE recompute is not supported
    with `moe_router` CUDA graph scope when using `cuda_graph_impl =
    "transformer_engine"`. Enforced in MCore `transformer_config.py:1977`.

11. **Layer-level recompute requires `full_iteration` scope**: Using
    `recompute_granularity="full"` with `recompute_num_layers` (recompute N
    whole transformer layers) is incompatible with TE-scoped graphs. MCore
    calls this "full" granularity even though you're selecting how many
    layers — the name refers to recomputing the full layer, not full model.
    Any TE-scoped scope (`attn`, `mlp`, `moe_router`, etc.) will assert:
    `AssertionError: full recompute is only supported with full iteration CUDA graph.`
    This commonly hits FP8 configs that default to TE-scoped graphs (e.g.
    `LLAMA3_70B_SFT_CONFIG_H100_FP8_CS_V1` uses `cuda_graph_impl=
    "transformer_engine"`, `cuda_graph_scope="mlp"`). Fix: use submodule
    recompute (`recompute_granularity="selective"` + `recompute_modules`),
    disable CUDA graphs, or switch to `local` + `full_iteration`. Enforced
    in MCore `transformer_config.py:2001-2005`. See also
    @skills/nemo-mbridge-perf-activation-recompute/SKILL.md.

12. **Benchmark numbers are workload-specific**: graph wins are usually real
    when host overhead is visible, but the exact gain depends on batch shape,
    PP depth, recompute, dispatcher backend, and whether the eager baseline was
    already optimized.

13. **A successful capture is not a speedup guarantee**: On 2026-05-18,
    Qwen3 30B A3B H100 BF16 pretrain with the all-to-all dispatcher captured
    TE-scoped `attn,moe_router,moe_preprocess` graphs successfully (`48`
    graphable layers, about `6.9 s` capture time on rank 0), but replay
    iterations 5-8 averaged `42.00 s` versus `41.36 s` for eager. Treat
    scoped graphs as a bring-up candidate and validate on the target stack.

14. **Narrow MoE scopes can win on a tuned dispatcher, but verify that they
    are actually narrow**: benchmark `moe_router,moe_preprocess` first, then
    add `attn` as a separate A/B only when `_layer_is_graphable()` and the
    runtime count show different selected layer sets. Compare only steady
    replay.

15. **Optimizer graph capture is a separate acceptance gate**: successful TE
    module graphs do not prove optimizer-step graph support. Bound optimizer
    capture time independently and reject a no-progress capture before waiting
    for replay measurements.

16. **Rebaseline graphs after changing precision or expert kernels**: a neutral
    BF16 graph result does not predict the FP8 path. On the measured
    Qwen3-30B-A3B 26.08.rc2 H100 stack, blockwise FP8 without scoped graphs was
    about 29.18 seconds/step, while
    `attn,moe_router,moe_preprocess` replay averaged 19.99728 seconds over
    steps 5--10 and reached 301.60 model TFLOP/s/GPU. The same campaign found
    adding attention scope negative for BF16. Graph capture can remove
    precision-path-specific launch/scaling overhead, so repeat the eager/graph
    A/B after changing precision, scaling recipe, or expert implementation.
    The transfer can reverse sharply: on Qwen3.5-35B-A3B with global blockwise
    FP8 but BF16 Torch grouped experts, the full
    `attn,moe_router,moe_preprocess` scope produced two consistent replay
    steps of 44.8776 and 44.8274 seconds versus 24.21325 seconds for eager.
    The bounded job timed out after iteration 6, so it is not a valid positive
    benchmark, but the 85.24% replay regression is sufficient to reject that
    scope and audit scope equivalence before attempting another A/B.

17. **Scope labels can be layer selectors rather than submodule boundaries**:
    Qwen3.5-35B-A3B has `moe_layer_freq=1`; its full scoped run reported 41
    graphable layers. A proposed `moe_router,moe_preprocess` follow-up was
    cancelled before allocation because current MCore would still select all
    40 model layers plus the MTP layer. Verify `_layer_is_graphable()`, layer
    types, and the runtime count before spending an allocation on a supposedly
    narrower scope.

## Verification

### Unit tests

```bash
uv run python -m pytest \
  tests/unit_tests/training/test_config.py -k "cuda_graph" \
  tests/unit_tests/training/test_comm_overlap.py -k "cuda_graph" \
  tests/unit_tests/models/test_gpt_full_te_layer_autocast_spec.py -k "cuda_graph" -q
```

### Functional smoke test (requires GPU)

```bash
uv run python -m pytest \
  tests/functional_tests/test_groups/recipes/test_llama_recipes_pretrain_cuda_graphs.py -q
```

### Success criteria

- Unit tests pass, covering config validation for both `local` and
  `transformer_engine` implementations.
- Functional test completes training steps with both CUDA graph
  implementations.
- No NCCL errors or illegal memory access in logs.
