# Bridge Training vs. Megatron-LM Training Loop: Detailed Comparison

This document captures all differences found between `src/megatron/bridge/training/` (Bridge)
and `3rdparty/Megatron-LM/megatron/training/` (Megatron-LM upstream) in the training loop,
dataloader setup, and related infrastructure.

---

## 1. DataLoader Construction Differences

### 1.1 `num_workers` default

| | Bridge | Megatron-LM |
|---|---|---|
| Default | **8** | **2** |
| Source | `DataloaderConfig.num_workers` in `bridge/training/config.py` | `--num-workers` in `megatron/training/arguments.py` |

Bridge uses 4x more dataloader workers by default.

### 1.2 `persistent_workers`

| | Bridge | Megatron-LM |
|---|---|---|
| Default | **`False`** | **`True if num_workers > 0 else False`** (effectively `True`) |

Bridge (`bridge/training/config.py`):
```python
persistent_workers: bool = False
```

Megatron-LM (`megatron/training/datasets/data_samplers.py`):
```python
return torch.utils.data.DataLoader(
    dataset,
    # ...
    persistent_workers=True if args.num_workers > 0 else False,
)
```

Megatron-LM keeps dataloader workers alive between batches; Bridge tears them down.

### 1.3 `pin_memory`

| | Bridge | Megatron-LM |
|---|---|---|
| Default | `True` (configurable via `DataloaderConfig.pin_memory`) | `True` (**hardcoded**, not configurable) |

Same effective default, but Bridge allows overriding.

### 1.4 `collate_fn`

| | Bridge | Megatron-LM |
|---|---|---|
| Behavior | Always checked: passes `dataset.collate_fn` if the attribute exists | Only for `hybrid_context_parallel`: `lambda x: x` |

Bridge (`bridge/data/loaders.py`):
```python
collate_fn=train_ds.collate_fn if hasattr(train_ds, "collate_fn") else None,
```

Megatron-LM (`megatron/training/datasets/data_samplers.py`):
```python
if args.hybrid_context_parallel:
    extra_kwargs = {"collate_fn": lambda x: x,}
else:
    extra_kwargs = {}
```

### 1.5 `dataloader_type` choices and default

| | Bridge | Megatron-LM |
|---|---|---|
| Supported types | `"single"`, `"cyclic"`, **`"batch"`**, `"external"` | `"single"`, `"cyclic"`, `"external"` |
| Default | `None` (user must set) | `None` → auto-set to **`"single"`** during argument validation |

Bridge adds the `"batch"` dataloader type with `MegatronPretrainingBatchSampler` for
fine-tuning (SFT). This sampler supports `pad_samples_to_global_batch_size`.

### 1.6 `worker_init_fn` gating

| | Bridge | Megatron-LM |
|---|---|---|
| Condition | `exit_signal_handler_for_dataloader` (dedicated boolean flag, default `False`) | `exit_signal_handler and num_workers > 0` |

Bridge uses a separate, explicit flag for dataloader worker signal handling.

### 1.7 Dataset build TP-rank gating

| | Bridge | Megatron-LM |
|---|---|---|
| Who builds datasets | **All ranks** | **TP rank 0 only** (unless `is_distributed`) |

Bridge (`bridge/data/utils.py`):
```python
# Build the dataset on all ranks for TP-replicated loading
train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(...).build()
```

Megatron-LM (`megatron/training/training.py`):
```python
if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:
    # Build datasets and dataloaders.
```

### 1.8 Validation dataloader

| | Bridge | Megatron-LM |
|---|---|---|
| Validation `dataloader_type` | Forces `"cyclic"` when using `GPTDatasetConfig` | Same as training type |
| Multiple validation sets | Not supported | Supported (`--multiple-validation-sets`) |
| Full validation mode | Not supported | Supported (`--full-validation`, consumes all samples) |

---

## 2. Sampler Differences

### 2.1 `MegatronPretrainingBatchSampler` (Bridge only)

Bridge has a batch sampler for SFT that supports:
- Global batch-level sampling (rather than microbatch-level)
- `pad_samples_to_global_batch_size` for handling incomplete batches

### 2.2 `HybridCPMegatronPretrainingSampler` (Megatron-LM only)

Megatron-LM has a sampler for `--hybrid-context-parallel` mode. Bridge does not support this.

---

## 3. `get_batch` / Data Fetching

### 3.1 TP-rank data distribution

| | Bridge | Megatron-LM |
|---|---|---|
| Mechanism | `get_batch_from_iterator` — reads directly from iterator on every rank | `get_batch_on_this_tp_rank` — broadcasts data from TP rank 0 to other TP ranks |

Bridge (`bridge/training/gpt_step.py`):
```python
def get_batch_from_iterator(data_iterator, use_mtp, ...):
    batch = next(data_iterator)
    # moves to cuda directly per-rank
```

Megatron-LM (`pretrain_gpt.py`):
```python
batch = get_batch_on_this_tp_rank(
    data_iterator,
    mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
)
```

This is coupled with the dataset build TP-gating difference (§1.7): Megatron-LM only
builds datasets on TP rank 0 and broadcasts; Bridge builds on all ranks and reads directly.

### 3.2 Context parallelism packed sequence handling

| | Bridge | Megatron-LM |
|---|---|---|
| Standard CP (no packing) | `get_batch_on_this_cp_rank` | `get_batch_on_this_cp_rank` |
| THD packed + CP | `_partition_packed_batch_for_cp` (via TE `thd_get_partitioned_indices`) | `get_thd_batch_on_this_cp_rank` |
| Hybrid CP | **Not supported** | `get_batch_on_this_hybrid_cp_rank` (via `local_cp_size`) |

Both Bridge and Megatron-LM support sequence packing (THD format) with Context Parallelism.
The only missing format in Bridge is **Hybrid CP** (`--hybrid-context-parallel`), which
combines different CP strategies using a `local_cp_size` field in the batch.

---

## 4. `train_step` Differences

### 4.1 `force_all_reduce` and gradient debug saving

| | Bridge | Megatron-LM |
|---|---|---|
| `force_all_reduce` | **Not passed** to `forward_backward_func` | Passed (tied to `save_wgrads_interval`) |
| `save_dgrads` / `save_wgrads` | **Not supported** | Supported for gradient debugging |

Megatron-LM (`megatron/training/training.py`):
```python
save_dgrads_in_this_iteration = (args.save_dgrads_interval is not None and ...)
save_wgrads_in_this_iteration = (args.save_wgrads_interval is not None and ...)
# ...
model_chunk.force_all_reduce = save_wgrads_in_this_iteration
# ...
losses_reduced = forward_backward_func(
    ...,
    force_all_reduce=save_wgrads_in_this_iteration,
)
```

Bridge does not have gradient saving infrastructure. However, since `force_all_reduce` defaults
to `False` in MCore's `forward_backward_func`, the normal training path is unaffected. This is
a missing debug feature, not a correctness issue.

### 4.2 `decoder_seq_length` handling

| | Bridge | Megatron-LM |
|---|---|---|
| `decoder_seq_length` | Always set to `seq_length` (same as encoder) | Uses `args.decoder_seq_length` (separately configurable) |

Bridge (`bridge/training/train.py`):
```python
decoder_seq_length=seq_length,
```

Megatron-LM (`megatron/training/training.py`):
```python
decoder_seq_length=args.decoder_seq_length,
```

This matters for encoder-decoder models where the decoder sequence length differs.

### 4.3 Optimizer step synchronization

| | Bridge | Megatron-LM |
|---|---|---|
| `logical_and_across_model_parallel_group` | **Conditional** on `train_config.check_optimizer_step_success` | **Always** called |
| `reduce_max_stat_across_model_parallel_group` (grad_norm) | **Conditional** on `not train_config.skip_sync_grad_norm_across_mp` | **Always** called |

Bridge (`bridge/training/train.py`):
```python
if train_config.check_optimizer_step_success:
    update_successful = logical_and_across_model_parallel_group(update_successful, mp_group=pg_collection.mp)

if not train_config.skip_sync_grad_norm_across_mp:
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm, mp_group=pg_collection.mp)
```

Megatron-LM (`megatron/training/training.py`):
```python
update_successful = logical_and_across_model_parallel_group(update_successful)
grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
```

Bridge makes these configurable, but the **defaults match Megatron-LM** (`check_optimizer_step_success=True`,
`skip_sync_grad_norm_across_mp=False`), so the effective behavior is identical out of the box.
The conditional flags exist for advanced users who want to skip syncs for performance.

### 4.4 `qk_clip` and `log_max_attention_logit`

| | Bridge | Megatron-LM |
|---|---|---|
| Default `log_max_attention_logit` | `None` | `0` |
| Condition | `hasattr(cfg.model, "qk_clip") and cfg.model.qk_clip` | `args.qk_clip or args.log_max_attention_logit` |

Megatron-LM supports logging max attention logit **without** clipping (via
`--log-max-attention-logit`). Bridge only logs it when clipping is enabled.

### 4.5 Finetuning variable-length sequence support (Bridge only)

Bridge has a dedicated finetuning path in `train_step` (`bridge/training/train.py`):
```python
if cfg.dataset.dataloader_type == "batch":
    from megatron.bridge.data.finetuning import prepare_finetuning_batch
    forward_backward_data_iterator, seq_length = prepare_finetuning_batch(
        data_iterator=data_iterator,
        num_microbatches=get_num_microbatches(),
        default_seq_length=model_config.seq_length,
        seq_key="tokens",
    )
```

Megatron-LM does not have this finetuning batch preparation.

### 4.6 Vision pretraining (DINO) support (Megatron-LM only)

Megatron-LM has DINO-specific code in `train_step` (`megatron/training/training.py`):
```python
if args.vision_pretraining and args.vision_pretraining_type == "dino":
    unwrapped_model = unwrap_model(model[0])
    unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)
```

Bridge does not have this.

### 4.7 ModelOpt integration in loss function (Megatron-LM only)

Megatron-LM (`pretrain_gpt.py`):
```python
if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):
    loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
```

Bridge does not integrate ModelOpt into the loss function.

---

## 5. `dummy_train_step` Differences

| | Bridge | Megatron-LM |
|---|---|---|
| Behavior | Distinguishes `"batch"` (consume 1 global batch) vs pretrain (consume N microbatches) | Always consumes N microbatches + materializes via `get_batch_on_this_tp_rank` / `get_batch_on_this_cp_rank` |

Bridge (`bridge/training/train.py`):
```python
if cfg.dataset.dataloader_type == "batch":
    _ = next(train_data_iterator)       # Finetuning: consume global batch once
else:
    for _ in range(num_microbatches):    # Pretrain: consume microbatches
        _ = next(train_data_iterator)
```

Megatron-LM (`megatron/training/training.py`):
```python
def dummy_train_step(data_iterator):
    for _ in range(num_microbatches):
        batch = get_batch_on_this_tp_rank(data_iterator)
        batch = get_batch_on_this_cp_rank(batch)
```

---

## 6. `data_parallel_random_init`

| | Bridge (config) | Bridge (model provider API) | Megatron-LM |
|---|---|---|---|
| Default | `False` (`RNGConfig`) | **`True`** (function signature) | `False` |
| Implementation | `seed + 10 * dp_rank` when enabled | same | same |

There is a **split default** in Bridge:

- `RNGConfig.data_parallel_random_init` defaults to `False` (`bridge/training/config.py`)
- But the model provider functions (`provide_distributed_model` in `model_provider.py`,
  `build_model_stages` in `unimodal.py`, `base.py`, `mamba_builder.py`, `mimo_provider.py`)
  all default to `data_parallel_random_init=True` in their **function signatures**

The training loop path (`setup.py`) explicitly passes `cfg.rng.data_parallel_random_init`
(i.e., `False`), so it matches Megatron-LM. However, anyone calling the model provider API
**directly** (tests, examples, standalone scripts) gets `True` by default, causing
`broadcast_params()` to be called, which **differs** from Megatron-LM's default of `False`
(no broadcast).

```python
# setup.py — passes config value (False by default), matches Megatron-LM
model = cfg.model.provide_distributed_model(
    data_parallel_random_init=cfg.rng.data_parallel_random_init,  # False
    ...
)

# model_provider.py — function signature defaults to True
def provide_distributed_model(
    self,
    data_parallel_random_init: bool = True,  # <-- differs from Megatron-LM!
    ...
)
```

When `data_parallel_random_init=True`, the model provider calls `broadcast_params()` from
DP rank 0, ensuring all DP ranks start with identical weights despite random initialization.
When `False` (Megatron-LM default), each DP rank keeps its own random init — but this only
makes sense when loading from a checkpoint.

---

## 7. Architectural / Code Organization Differences

| Area | Bridge | Megatron-LM |
|---|---|---|
| State management | `GlobalState` + `ConfigContainer` dataclasses | Global `args` via `get_args()` |
| Process groups | Explicit `pg_collection` passed everywhere | Global `mpu` module |
| Callbacks | `CallbackManager` support | Not present |
| Profiling | Extracted into `profiling.py` helpers | Inline in training loop |
| Loss function | Extracted into `losses.py` module | Inline in `pretrain_gpt.py` |
| Forward step | Takes `GlobalState` as first arg | Takes `data_iterator` as first arg |
| Timers | `state.timers` | `get_timers()` global |
| Straggler detection | `state.straggler_timer` | Global `stimer` |

---

## Summary: High-Impact Differences for Reproducibility

The following differences are most likely to cause behavioral divergence when running the
same model/data between Bridge and Megatron-LM:

1. **`num_workers`**: 8 vs 2 — affects data loading speed and memory
2. **`persistent_workers`**: False vs True — affects worker lifecycle and memory
3. **TP-rank data gating**: All ranks build datasets vs TP rank 0 only — affects data distribution
4. **TP-rank data broadcast**: Direct read vs `get_batch_on_this_tp_rank` broadcast — affects data flow
5. **`decoder_seq_length`**: Always equals `seq_length` vs separately configurable
6. **`data_parallel_random_init`**: Model provider API defaults to `True` (broadcasts params);
   Megatron-LM defaults to `False` — affects standalone scripts/tests using the provider directly
7. **Optimizer step sync**: Conditional in Bridge but defaults match Megatron-LM — no issue in practice
8. **`force_all_reduce`**: Not passed in Bridge, but MCore defaults to `False` — no issue for normal training
   (missing gradient-saving debug feature only)
9. **Hybrid CP**: Bridge supports standard CP and THD packed+CP, but not Hybrid CP
