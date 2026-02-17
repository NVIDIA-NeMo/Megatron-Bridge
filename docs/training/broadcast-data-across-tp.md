# Broadcast Data Across Tensor-Parallel Ranks

## Feature overview

The `broadcast_data_across_tp` flag controls how data is loaded across
tensor-parallel (TP) ranks during pretraining.

| Mode | `broadcast_data_across_tp` | Behaviour |
|---|---|---|
| Replicated (default) | `False` | Every TP rank independently builds a DataLoader and reads from storage. |
| Broadcast | `True` | Only TP-rank-0 reads from storage; the batch is broadcast to the remaining TP ranks via NCCL. |

**When to enable broadcast mode:**
Replicated loading works well on low-latency parallel file-systems such as
Lustre.  On high-latency, network-attached storage (e.g. VAST) the redundant
I/O from all TP ranks causes severe contention -- `open()` syscall latency
degrades to ~1 s per call with 64+ concurrent processes.  Enabling
`broadcast_data_across_tp` reduces VAST readers by a factor of TP and
eliminates the stalls.

### Configuration

```yaml
dataset:
  broadcast_data_across_tp: true
```

## Implementation details

When the flag is enabled two things change:

1. **Dataset building** (`data/utils.py`): `BlendedMegatronDatasetBuilder`
   receives `is_dataset_built_on_rank` instead of `lambda: True`, so the
   dataset is only constructed on TP-rank-0 of first/last PP stages.

2. **Batch loading** (`gpt_step.py` / `batch_utils.py`):
   `get_batch_on_this_tp_rank` is called instead of `get_batch_from_iterator`.
   The five standard fixed-shape tensors (tokens, labels, loss_mask,
   attention_mask, position_ids) are broadcast via direct NCCL `broadcast`
   calls.  Any additional keys (e.g. packed-sequence metadata) are forwarded
   via `broadcast_object_list`, guarded by a lightweight boolean flag so the
   heavy path is skipped when there are no extra keys.

## Performance benchmark

### Methodology

- **Hardware:** 8x NVIDIA B200 (single node, NVLink interconnect)
- **Software:** PyTorch with NCCL backend, `torchrun --nproc_per_node=8`
- **Measurement:** Median of 50 iterations after 10 warmup iterations per
  configuration.  Each iteration includes data loading on rank 0, NCCL
  broadcast, and `.cuda()` transfer on receivers.  `torch.cuda.synchronize()`
  bookends each iteration.

### How to reproduce

```bash
torchrun --nproc_per_node=8 tests/benchmarks/bench_broadcast_tp.py
```

### Results (8x B200)

| Config                   |   Data |  Direct (ms) |  ObjList (ms) |  Overhead |
|--------------------------|--------|--------------|---------------|-----------|
| mbs=1 seq=8K             | 0.2 MB |       0.14   |        0.62   |   +0.48   |
| mbs=1 seq=8K +extra      | 0.2 MB |       0.16   |        0.64   |   +0.49   |
| mbs=1 seq=32K            | 0.9 MB |       0.18   |        0.89   |   +0.70   |
| mbs=1 seq=32K +extra     | 0.9 MB |       0.21   |        0.90   |   +0.70   |
| mbs=1 seq=128K           | 3.7 MB |       0.36   |        1.86   |   +1.50   |
| mbs=1 seq=256K           | 7.3 MB |       0.57   |        3.58   |   +3.01   |

**Direct broadcast** is the path taken for the standard batch keys.
**ObjList** is the fallback used only for extra keys (packed-sequence
metadata); in the common pretraining case this path is skipped entirely.

At 256K sequence length the direct broadcast adds < 0.6 ms per step --
negligible compared to a typical 22 s compute-bound step.

## Test coverage

### Unit tests

```bash
pytest tests/unit_tests/training/utils/test_batch_utils.py -v
```

| Test | What it validates |
|---|---|
| `test_standard_keys_returned` | TP rank 0 loads data; all 5 standard keys present in result. |
| `test_extra_keys_broadcast` | Extra keys (e.g. `cu_seqlens`) forwarded via `broadcast_object_list`. |
| `test_no_extra_keys_skips_heavy_broadcast` | When only standard keys exist, the heavy `broadcast_object_list` is skipped (only the boolean flag is broadcast). |

All tests use mocked `torch.distributed` and run on CPU without GPUs.

### Regression checklist

Manual validation steps for multi-GPU environments:

- [ ] Standard pretraining (TP=8, CP=1) -- no deadlocks, loss matches baseline
- [ ] Pretraining with context parallelism (TP=4, CP=2) -- no deadlocks
- [ ] Pretraining with MTP enabled -- tokens/position_ids broadcast to last PP stage
- [ ] `broadcast_data_across_tp=False` (default) -- behaviour unchanged from main
- [ ] Single DataLoader worker per rank on VAST storage -- no I/O stalls
- [ ] Lustre-backed storage -- no regression in step time
