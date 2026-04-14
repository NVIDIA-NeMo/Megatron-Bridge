# Qwen3-Omni Local Optimization Plan

This file is local-only and should remain untracked.

## Goal

Use the already-working 4-node 32-GPU thinker training path as the optimization baseline, then improve throughput without losing stability.

Current target:

- multimodal input is active
- audio data is present in the current local training set
- training objective remains thinker-side SFT
- optimization focus is throughput first, memory second

## Dataset Sanity

Current local training set:

- `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_data/omni_bench_fix_simple/train/train.jsonl`

Current observed composition:

- total samples: `20`
- audio-like samples: `20`
- image-like samples: `20`
- video-like samples: `0`

Implication:

- current A/B experiments do exercise audio inputs
- current 100-200 step observe runs are useful for stability and rough trend checks
- current 100-200 step observe runs are not sufficient to judge real convergence quality because the dataset is tiny

## Baseline

Best-known working 32-GPU recipe today:

- launcher:
  - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp.sh`
- parallel shape:
  - `NNODES=4`
  - `NUM_GPUS=8`
  - `TP=2`
  - `PP=2`
  - `CP=1`
  - `EP=8`
  - `ETP=1`
  - `SP=True`
- model/data:
  - `SEQ_LENGTH=16384`
  - `GBS=16`
  - `MBS=1`
  - `freeze_language_model=False`
  - `freeze_vision_model=True`
  - `freeze_audio_model=True`
- runtime:
  - `attention_backend=flash`
  - `recompute_granularity=full`
  - `recompute_method=uniform`
  - `recompute_num_layers=24`
  - `recompute_modules=core_attn`
  - `optimizer_cpu_offload=True`
  - `optimizer_offload_fraction=1.0`
  - `use_precision_aware_optimizer=True`

Observed 5-step baseline:

- warmup step:
  - `85.38s`
  - `3.8 TFLOP/s/GPU`
- steady-state steps:
  - `19.18s` to `19.88s`
  - `16.2` to `16.8 TFLOP/s/GPU`
- representative steady-state point:
  - `19.22s`
  - `16.8 TFLOP/s/GPU`
- memory:
  - rank 0 `mem-max-allocated-gigabytes=48.46`
  - rank 0 `mem-max-reserved-gigabytes=50.275`
  - another shard example rank 16 `mem-max-allocated-gigabytes=49.042`

## Parallel Recipe Assessment

Current answer:

- `TP=2 / PP=2 / EP=8 / SP=True` is the best-known stable recipe
- it is not yet proven to be the optimal recipe

Why it is the best-known stable recipe:

- `TP=1 / PP=2 / EP=8 / SP=False / seq=16384` was closer to the business-side shape but OOMed in MoE token dispatcher combine/sort
- moving to `TP=2` reduced the hidden-side MoE temporary buffer pressure enough to run successfully
- `SP=True` is required with MoE when `TP>1`

Why it is not yet proven optimal:

- we have not done controlled throughput A/B across multiple parallel layouts
- the current recipe was chosen to solve a concrete OOM, not because it won a performance sweep

Current recommendation:

- treat the current recipe as the optimization baseline
- do not change parallel layout in the first performance round
- first exhaust lower-risk runtime knobs before reopening parallel search

Parallel search can be revisited later if runtime knobs plateau.

## Optimization Priorities

### Phase 1: Low-risk throughput A/B on the current parallel recipe

Run one change at a time on top of the current baseline.

1. Optimizer offload A/B

- baseline:
  - `optimizer_cpu_offload=True`
  - `optimizer_offload_fraction=1.0`
  - `use_precision_aware_optimizer=True`
- concrete local experiment shells:
  - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp_ab_offload_on.sh`
  - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp_ab_offload_off.sh`
- default run length:
  - `TRAIN_ITERS=30`
  - `LOG_INTERVAL=5`
- compare:
  - steady-state `Step Time`
  - `throughput/tflops/device`
  - estimated `throughput/mfu_percent/device`
  - peak memory
- probe:
  - disable optimizer CPU offload first
- reason:
  - current memory headroom is large enough that CPU offload may be costing throughput unnecessarily
- result:
  - `offload off` wins clearly on throughput
  - use `offload off` as the new baseline for the next A/B round

2. Recompute intensity A/B

- baseline:
  - `full/uniform/24`
  - `recompute_modules=core_attn`
- concrete local experiment shells:
  - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp_ab_recompute24.sh`
  - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp_ab_recompute12.sh`
- default run length:
  - `TRAIN_ITERS=30`
  - `LOG_INTERVAL=5`
- probes:
  - reduce `recompute_num_layers` from `24` to `12`
  - if safe, test lighter recompute settings
- reason:
  - current config is memory-conservative and may be overpaying in compute

3. DDP overlap A/B

- current recipe path still effectively runs without overlap tuning
- concrete local experiment shells:
  - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp_ab_overlap_off.sh`
  - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp_ab_overlap_on.sh`
- default run length:
  - `TRAIN_ITERS=30`
  - `LOG_INTERVAL=5`
- probes:
  - `overlap_grad_reduce=True`
  - `overlap_param_gather=True`
  - `align_param_gather=True`
- reason:
  - can improve throughput with moderate implementation risk

### Phase 2: Medium-risk runtime/system tuning

4. MoE routing/runtime sanity

- inspect whether explicit `moe_router_dtype='fp32'` should be enforced for this recipe
- main value is numerical stability, not raw speed

5. CUDA graph / capture tuning

- current path runs and deletes CUDA graphs, but graph strategy has not been deliberately optimized
- revisit only after runtime knobs settle

### Phase 3: Parallel recipe exploration

Only start this phase after Phases 1-2 stop giving meaningful gains.

Candidate families:

- `TP=2 / PP=2 / EP=8 / SP=True`
  - current baseline
- `TP=4 / PP=2 / EP=4 / SP=True`
  - plausible exploration candidate
  - may trade expert parallelism for smaller TP-local hidden work
- other shapes should be considered only if they preserve workable MoE behavior and avoid reintroducing the old dispatcher OOM

Important note:

- any parallel search must be done as controlled A/B against the current baseline
- do not mix parallel-layout changes with optimizer/recompute changes in the same run

## Metrics To Track For Every A/B

For each run, record:

- steady-state step time
- steady-state `throughput/tflops/device`
- `mem-max-allocated-gigabytes`
- `mem-max-reserved-gigabytes`
- whether the run is stable for at least `100` steps
- rough loss trend only as a sanity check

## Immediate Next Steps

1. Run the `100` or `200` step observe shell on the current baseline recipe.
2. Create the first performance A/B pair on the same recipe:
   - baseline
   - optimizer offload disabled
3. Compare only:
   - step time
   - TFLOP/s/GPU
   - memory
4. If offload removal helps and remains stable, continue to recompute A/B.
