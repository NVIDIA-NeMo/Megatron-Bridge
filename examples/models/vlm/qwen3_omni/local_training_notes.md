# Qwen3-Omni Thinker Local Training Notes

This file is local-only and should remain untracked.

## Worktree

- Repo: `/home/luban/liuwei/omni/megatron-bridge/Megatron-Bridge-omni3-train`
- Branch: `omni3-train`

## Local Runtime Assumptions

- Saved env: `/nfs/ml-training-ssd/users/liuwei/condaenv/omni3_megatron`
- Smoke HF checkpoint: `/nfs/ml-training-ssd/users/liuwei/qwen3_omni_examples/hf/Qwen3-Omni-30B-A3B-Instruct-smoke`
- Full HF checkpoint: `/nfs/volume-1615-2/models/Qwen3-Omni-30B-A3B-Instruct`
- Converted local data root: `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_data/omni_bench_fix_simple`
- Local training script: `examples/models/vlm/qwen3_omni/local_train_thinker_full.sh`

## Local-Only Runtime Fixes

- `typing.override` shim for Python 3.11 in `sitecustomize.py`
- path cleanup to avoid leaking `/home/luban/anaconda3` and user site-packages into training jobs
- local thinker-only full-model mirror generated under:
  - `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/hf_thinker_only/Qwen3-Omni-30B-A3B-Instruct-thinker-only`
- local env patching for:
  - `transformer-engine`
  - `librosa`
  - `audioread`

## Validated Results

### Smoke thinker training

- Model: smoke HF checkpoint
- GPUs: 1
- Data: local preloaded `jsonl`
- Status: passed
- Command shape:
  - `HF_MODEL_PATH=/nfs/ml-training-ssd/users/liuwei/qwen3_omni_examples/hf/Qwen3-Omni-30B-A3B-Instruct-smoke`
  - `NUM_GPUS=1 TRAIN_ITERS=1 GLOBAL_BATCH_SIZE=1 MICRO_BATCH_SIZE=1`

### Smoke metrics

- `elapsed time per iteration (ms)`: `62475.2`
- `throughput per GPU (TFLOP/s/GPU)`: `0.2`
- `lm loss`: `1.235484E+01`
- `load_balancing_loss`: `3.891017E+00`
- `grad norm`: `58.951`
- `mem-allocated-gigabytes`: `37.141`

### Full model thinker training

- Model: full 30B thinker-only mirrored config
- GPUs: 8 x 80GB
- Current status: model construction reaches MoE TE grouped MLP allocation, then OOMs
- Current config was still:
  - `TP=1`
  - `PP=1`
  - `EP=1`
  - `ETP=1`
  - `SP=False`
- Conclusion:
  - the full training path is now logically wired through
  - current blocker is memory scaling, not missing thinker training support

### Parallel bring-up notes

- `EP=8`:
  - reaches training forward, but still OOMs in MoE token dispatch/combine
- `TP=2 + EP=4`:
  - blocked by Megatron requirement that MoE + TP also enable `sequence_parallel`
- `PP=2 + EP=4`:
  - reaches training loop
  - exposed a Qwen3-Omni step bug where pipeline-parallel batch padding assumed a 4D attention mask
  - fixed locally by making attention-mask padding handle both 2D and 4D masks
  - after the mask fix, reaches real thinker forward and then OOMs in the HF vision tower rotary-attention path
  - conclusion:
    - `PP + EP` is functionally alive
    - remaining blocker is memory pressure, not pipeline wiring
    - `TP` still cannot be pursued further until `SP` support is implemented for multimodal vision inputs

## 2026-04-14: MFU logging support

- Clarified that the existing `MODEL_TFLOP/s/GPU` logger is throughput, not hardware utilization.
- Added `logger.peak_theoretical_tflops_per_gpu` to training logger config.
- Added MFU logging in `src/megatron/bridge/training/utils/train_utils.py`:
  - console: `MFU: xx.x%`
  - TensorBoard/WandB/MLFlow/Comet:
    - `throughput/mfu/device`
    - `throughput/mfu_percent/device`
- Wired local Qwen3-Omni TP2/PP2/EP8/SP scripts to default `GPU_PEAK_TFLOPS_PER_DEVICE=312`.
  - This assumes A100 bf16/fp16 tensor-core peak for local experiments.

## 2026-04-14: Current baseline MFU estimate

- Current stable 32-GPU baseline is about:
  - `17.5` to `18.5 MODEL_TFLOP/s/GPU`
- Using local A100 peak assumption `312 TFLOP/s/GPU`, estimated MFU is:
  - `5.6%` to `5.9%`
- Interpretation:
  - the run is stable, but hardware utilization still has room
  - first optimization target is throughput, not memory

## 2026-04-14: First performance A/B defaults

- For optimizer-offload A/B, shortened the default run length from `100` to `30` steps.
- Reason:
  - convergence has already been sanity-checked
  - A/B now targets steady-state throughput, not training quality
  - `30` steps is enough to pass warmup and observe stable throughput while saving cluster time
- Current default for both A/B shells:
  - `TRAIN_ITERS=30`
  - `LOG_INTERVAL=5`

### SP bring-up notes

- first-version SP bring-up is based on:
  - `Qwen2.5-Omni` thinker SP padding for `combined_embeddings` and `position_ids`
  - `Qwen3-VL` splitting of `visual_pos_masks` and `deepstack_visual_embeds`
- goal of this first pass:
  - remove the local `vision + SP` hard stop in `Qwen3-Omni thinker`
  - make `image/text` path reach decoder under `SP=True`
  - defer deeper cleanup/restructure until after the initial bring-up is validated
- implementation note:
  - this pass keeps the current multimodal embedding flow intact and only adds SP scatter/padding plus deepstack split handling
- local validation:
  - `tests/unit_tests/models/qwen_omni/modeling_qwen3_omni/test_omni_model.py -k image_forward`
  - status: passed
- multi-gpu status:
  - `TP=2 + EP=4 + SP=True` now enters real training forward and no longer fails on the old vision/SP guard
  - current blocker moved to MoE token dispatcher memory pressure
  - next trial config: `TP=2 + PP=2 + EP=2 + SP=True`
  - `TP=2 + PP=2 + EP=2 + SP=True` also enters real training forward
  - no new PP/SP wiring errors were observed
  - current blocker remains MoE token dispatcher/combine memory pressure
  - next pragmatic trial is to keep the same parallel layout and reduce `seq_length` before scaling out to more GPUs

### ms-swift comparison notes

- local reference repo:
  - `/nfs/ofs-llab-hdd/users/liuwei/omni/msswift`
- key findings from the `Qwen3-Omni` path:
  - `ms-swift` also uses HF vision/audio towers for `Qwen3-Omni`, not a fully Megatron-native tower
  - multimodal tower memory savings come from runtime settings, not from replacing the tower architecture
  - Megatron-SWIFT docs explicitly state:
    - `vit_gradient_checkpointing=True` by default for multimodal training
    - `attn_impl='flash_attn'` by default for multimodal tower attention
  - trainer code enables `gradient_checkpointing_enable()` and `enable_input_require_grads()` on each multimodal tower
  - example scripts also constrain visual token budget with env vars such as:
    - `IMAGE_MAX_TOKEN_NUM`
    - `VIDEO_MAX_TOKEN_NUM`
    - `FPS_MAX_FRAMES`

### Tower runtime switches

- added tracked config switches for local `Qwen3-Omni thinker` training:
  - `model.vit_gradient_checkpointing`
  - `model.multimodal_attn_impl`
- implementation behavior:
  - when `vit_gradient_checkpointing=True`, the HF `visual` and `audio_model` towers now try to enable:
    - `gradient_checkpointing_enable()`
    - `enable_input_require_grads()`
  - when `multimodal_attn_impl` is not `auto`, the requested attention implementation is propagated into the HF tower configs
  - follow-up compatibility fix:
    - HF `Qwen3OmniMoeAudioEncoder.enable_input_require_grads()` assumes `get_input_embeddings()` returns a valid module
    - the default audio implementation points at a missing `conv1`
    - local thinker code now patches tower input embeddings to match the working ms-swift entrypoints:
      - `visual.get_input_embeddings() -> patch_embed`
      - `audio_model.get_input_embeddings() -> conv_out`
- current default behavior remains unchanged:
  - `vit_gradient_checkpointing=False`
  - `multimodal_attn_impl='auto'`

### Baseline plan

- before further memory tuning, keep one explicit baseline and one optimized profile:
  - baseline:
    - current parallel layout
    - `vit_gradient_checkpointing=False`
    - `multimodal_attn_impl='auto'`
    - no visual token-budget env limits beyond dataset defaults
  - optimized:
    - same parallel layout
    - `vit_gradient_checkpointing=True`
    - `multimodal_attn_impl='flash_attention_2'` (or `flash_attn` if required by runtime)
    - explicit visual token-budget env vars in the launcher
- reason:
  - this keeps the comparison interpretable and lets us separate:
    - language-side parallel gains
    - multimodal tower memory savings
- latest optimized 8x80G result with:
  - `TP=2`
  - `PP=2`
  - `EP=2`
  - `SP=True`
  - `SEQ_LENGTH=2048`
  - `vit_gradient_checkpointing=True`
  - `multimodal_attn_impl='flash_attention_2'`
  - visual token-budget limits
  still failed due pure OOM; NCCL could not `CUDA calloc` ~6 MB during TP communication
- next clean follow-up:
  - keep the same optimized profile
  - reduce only `SEQ_LENGTH` to `1024`
- current CP note:
  - `CP` may help for attention-heavy long-context memory, but the observed failures so far are not specifically pointing at context-parallel bottlenecks
  - keep `CP` behind the current seq-length check instead of expanding the search space immediately
- next alignment pass against public `ms-swift` examples:
  - keep `TP=2`, `PP=2`, `EP=2`, `SP=True`
  - set `GLOBAL_BATCH_SIZE=4`
  - set `ATTENTION_BACKEND=flash`
  - set `RECOMPUTE_GRANULARITY=full`
  - set `RECOMPUTE_METHOD=uniform`
  - set `RECOMPUTE_NUM_LAYERS=1`
  - set `MAX_PIXELS=1003520`
  - set `VIDEO_MAX_PIXELS=50176`
  - keep multimodal tower options already enabled
- 16-GPU trial script updated in-place for a larger bring-up attempt:
  - corrected from single-node `NUM_GPUS=16` to multi-node launch:
    - `NUM_GPUS=8`
    - `NNODES=2`
    - rely on scheduler-provided `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` when available
  - `TP=1`
  - `PP=2`
  - `EP=8`
  - `ETP=1`
  - `SP=True`
  - `SEQ_LENGTH=1024`
  - `GLOBAL_BATCH_SIZE=8`
  - keep `vit_gradient_checkpointing`, `flash` attention backend, `recompute_*`, and visual token-budget limits enabled
  - intent: approximate the public 16-card ms-swift omni parallel shape as closely as possible in the local launcher
- follow-up on the 2-node launcher after a rendezvous hang:
  - stop assuming `VC_TASK_INDEX` alone is the node rank
  - align node-rank derivation with the user's reference script:
    - `master -> NODE_RANK=VC_TASK_INDEX`
    - `worker -> NODE_RANK=VC_TASK_INDEX+1`
  - continue resolving `VC_MASTER_HOSTS` to a plain IPv4 master address
  - explicitly export normalized `MASTER_ADDR` / `MASTER_PORT` and `TORCH_MASTER_ADDR` / `TORCH_MASTER_PORT`
  - add `TASK_ROLE` / `TASK_INDEX` to launcher logs so future hangs can be diagnosed from one log header

## Known Follow-ups

- update affected unit tests after the `run_recipe` lazy-import / recipe refactor
- begin full-model parallelism bring-up:
  - first candidate knobs: `TP`, `EP`, `SP`
  - keep talker/code2wav out of scope
- later clean up local-only env shims before upstreaming training support

- 2026-04-02 2-node launch now reaches training forward; next blocker moved to Qwen3-Omni RoPE indexing.
- fixed `modeling_qwen3_omni/rope.py` to normalize multi-dimensional attention masks to `[batch, seq]` and use boolean indexing like Qwen2.5/Qwen3-VL.
- also hardened the 2-node thinker-only config mirror against concurrent writes by using pid-scoped temp files.
- 2026-04-02 follow-up: when `seq_length` truncates multimodal samples, align Qwen3-Omni audio RoPE/audio embedding lengths to the actual surviving `audio_token` placeholders in `input_ids`.
- 2026-04-02 follow-up: make Qwen3-Omni RoPE truncation-safe by clipping generated multimodal positions to the number of surviving tokens after `seq_length` truncation.
- 2026-04-02 follow-up: trim visual feature streams to the surviving image/video placeholder count so sequence truncation cannot leave HF vision features longer than `input_ids`.
- 2026-04-02 follow-up: expose `pre_process`, `post_process`, and `share_embeddings_and_output_weights` on the top-level Qwen3-Omni wrapper so Megatron finalize-model-grads can find embedding-sharing metadata during PP runs.
- 2026-04-03: switched the visible 8-card launcher to a non-truncated freeze experiment: single-node 8 GPUs, `TP=1`, `PP=2`, `EP=4`, `SEQ_LENGTH=4096`, `freeze_vision_model=True`, `freeze_audio_model=True`, `freeze_language_model=False`, keeping flash/recompute knobs enabled.
- 2026-04-03: aligned the visible 8-card launcher to the business-side ms-swift SFT shape as closely as single-node 8 GPUs allow:
  - confirmed from the shared args dump that the business run uses `max_length=16384`
  - keep `freeze_llm=False`, `freeze_vit=True`, `freeze_aligner=True` semantics approximated locally by:
    - `freeze_language_model=False`
    - `freeze_vision_model=True`
    - `freeze_audio_model=True`
  - keep `TP=1`, `PP=2`, `SP=False`, `micro_batch_size=1`, `attention_backend=flash`
  - on 8 GPUs, `EP=8` is not possible together with `PP=2`, so the nearest local substitute is `EP=4`
  - updated the visible 8-card launcher to:
    - `SEQ_LENGTH=16384`
    - `GLOBAL_BATCH_SIZE=16`
    - `RECOMPUTE_GRANULARITY=full`
    - `RECOMPUTE_METHOD=uniform`
    - `RECOMPUTE_NUM_LAYERS=24`
    - `RECOMPUTE_MODULES=core_attn`
    - `OPTIMIZER_CPU_OFFLOAD=True`
    - `OPTIMIZER_OFFLOAD_FRACTION=1.0`
    - `USE_PRECISION_AWARE_OPTIMIZER=True`
    - `VIT_GRADIENT_CHECKPOINTING=False`
  - launcher support was added for:
    - `model.recompute_modules`
    - `optimizer.optimizer_cpu_offload`
    - `optimizer.optimizer_offload_fraction`
    - `optimizer.use_precision_aware_optimizer`
- 2026-04-03 follow-up: corrected the business-side scale assumption from 8 GPUs to 4 nodes x 8 GPUs = 32 GPUs.
  - this makes the shared ms-swift parallel shape internally consistent:
    - `TP=1`
    - `PP=2`
    - `EP=8`
    - `SP=False`
  - the visible launcher was updated accordingly to:
    - `NNODES=4`
    - `NUM_GPUS=8`
    - `EP=8`
    - `SEQ_LENGTH=16384`
    - `GLOBAL_BATCH_SIZE=16`
- 2026-04-07: added a separate 2-node x 8-GPU = 16-GPU business-aligned probe launcher:
  - `examples/models/vlm/qwen3_omni/local_train_thinker_2node_ep8.sh`
  - keeps the same business-style knobs:
    - `TP=1`
    - `PP=2`
    - `EP=8`
    - `SP=False`
    - `SEQ_LENGTH=16384`
    - `GLOBAL_BATCH_SIZE=16`
    - `MICRO_BATCH_SIZE=1`
    - frozen vision/audio, trainable language model
    - `RECOMPUTE_GRANULARITY=full`
    - `RECOMPUTE_METHOD=uniform`
    - `RECOMPUTE_NUM_LAYERS=24`
    - `RECOMPUTE_MODULES=core_attn`
    - optimizer CPU offload enabled
  - TensorBoard is now explicitly routed through `logger.tensorboard_dir`
  - default 16-GPU probe TensorBoard path:
    - `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/results/qwen3_omni_sft16_business_align_tp1_pp2_ep8_seq16384/tb_logs`
- 2026-04-08: 4-node 32-GPU business-aligned `TP=1/PP=2/EP=8/SP=False/SEQ=16384` run reached backward recompute, then OOMed in MoE token dispatcher combine/sort:
  - failing allocation was about 518 MiB in `transformer_engine/pytorch/triton/permutation.py::sort_chunks_by_idx`
  - this matches the expected `[seq_length * topk, hidden]` style MoE combine buffer pressure for `seq=16k`, `topk=8`, `TP=1`
  - added a follow-up 32-GPU probe:
    - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp.sh`
    - `TP=2`
    - `PP=2`
    - `EP=8`
    - `SP=True`
  - rationale:
    - this preserves `EP=8` and `SEQ=16384`
    - TP should reduce the hidden-dimension side of the MoE temporary buffers
    - SP is required for MoE when `TP>1`
  - default TensorBoard path:
    - `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/results/qwen3_omni_sft32_tp2_pp2_ep8_sp_seq16384/tb_logs`
- 2026-04-13: 4-node 32-GPU `TP=2/PP=2/EP=8/SP=True` baseline run completed successfully with a short 5-step probe.
  - note: this was intentionally a functionality/performance baseline, not a convergence run
  - the small number of steps came from the baseline launcher default:
    - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp_baseline.sh`
    - `TRAIN_ITERS=5`
  - TensorBoard baseline path:
    - `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/results/qwen3_omni_sft32_tp2_pp2_ep8_sp_seq16384_baseline/tb_logs`
  - stdout log:
    - `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/logs/qwen3_omni_sft32_tp2_pp2_ep8_sp_seq16384_baseline_full.log`
  - stable-step performance after the first warmup/cuda-graph iteration:
    - step time about `19.18s` to `19.88s`
    - throughput about `16.2` to `16.8 TFLOP/s/GPU`
    - representative steady-state point:
      - `19.22s`
      - `16.8 TFLOP/s/GPU`
  - warmup first step:
    - `85.38s`
    - `3.8 TFLOP/s/GPU`
  - memory snapshot after the first iteration:
    - rank 0 `mem-max-allocated-gigabytes=48.46`
    - rank 0 `mem-max-reserved-gigabytes=50.275`
    - another shard example rank 16 `mem-max-allocated-gigabytes=49.042`
  - last-step scalar values from TensorBoard:
    - `lm loss=12.1757`
    - `load_balancing_loss=3.5354`
    - `grad_norm=126.27`
  - interpretation:
    - this run is sufficient as a throughput/memory baseline
    - it is not long enough to judge optimization stability or loss convergence
- 2026-04-13: 4-node 32-GPU `TP=2/PP=2/EP=8/SP=True` observe run completed successfully for 100 steps.
  - launcher:
    - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp_observe.sh`
  - TensorBoard path:
    - `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/results/qwen3_omni_sft32_tp2_pp2_ep8_sp_seq16384_observe/tb_logs`
  - stdout log:
    - `/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_train/logs/qwen3_omni_sft32_tp2_pp2_ep8_sp_seq16384_observe_full.log`
  - dataset note:
    - current local train set has only `20` samples
    - all `20/20` samples include image plus audio
    - no video samples were observed in this local set
  - throughput / step-time summary:
    - early warmup point at step 5:
      - `30.30s`
      - `10.6 TFLOP/s/GPU`
    - stable region after warmup:
      - roughly `17.5s` to `18.5s`
      - roughly `17.5` to `18.5 TFLOP/s/GPU`
    - best observed point:
      - `17.48s`
      - `18.5 TFLOP/s/GPU`
  - memory snapshot:
    - rank 0 `mem-max-allocated-gigabytes=48.46`
    - rank 0 `mem-max-reserved-gigabytes=51.932`
    - rank 16 `mem-max-allocated-gigabytes=49.099`
  - last-step TensorBoard scalars:
    - `lm loss=0.0002349`
    - `load_balancing_loss=1.3555`
    - `grad_norm=0.5459`
    - `learning_rate=4.9693e-06`
  - interpretation:
    - the recipe is stable for at least `100` steps
    - throughput baseline for optimization should use the warmup-free region, not the early step-5 number
    - this run is still not a meaningful generalization/convergence benchmark because the local dataset is tiny and heavily repeated
- 2026-04-14: added MFU logging support on top of throughput.
  - clarified that existing `MODEL_TFLOP/s/GPU` is throughput, not hardware utilization
  - added `logger.peak_theoretical_tflops_per_gpu`
  - added logging:
    - `throughput/mfu/device`
    - `throughput/mfu_percent/device`
  - local Qwen3-Omni TP2/PP2/EP8/SP scripts default:
    - `GPU_PEAK_TFLOPS_PER_DEVICE=312`
  - current stable baseline MFU estimate from the observe run:
    - throughput about `17.5` to `18.5 MODEL_TFLOP/s/GPU`
    - estimated MFU about `5.6%` to `5.9%`
- 2026-04-14: optimizer-offload A/B established a new performance baseline.
  - A/B shells:
    - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp_ab_offload_on.sh`
    - `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp_ab_offload_off.sh`
  - both default to:
    - `TRAIN_ITERS=30`
    - `LOG_INTERVAL=5`
  - `offload on`:
    - steady-state observed around step 10:
      - `19.46s / step`
      - `16.6 MODEL_TFLOP/s/GPU`
      - estimated `MFU 5.3%`
    - rank 0 peak memory:
      - `mem-max-allocated-gigabytes = 48.46`
      - `mem-max-reserved-gigabytes = 52.121`
  - `offload off`:
    - steady-state across steps 10-30:
      - `14.87s` to `15.16s / step`
      - `21.3` to `21.7 MODEL_TFLOP/s/GPU`
      - estimated `MFU 6.8%` to `7.0%`
    - stable average:
      - `21.56 MODEL_TFLOP/s/GPU`
      - estimated `MFU 6.91%`
    - rank 0 peak memory:
      - `mem-max-allocated-gigabytes = 59.848`
      - `mem-max-reserved-gigabytes = 63.168`
  - interpretation:
    - disabling optimizer offload improves throughput by about `30%`
    - memory rises by about `11 GB`, but still leaves useful headroom on 80GB GPUs
    - new optimization baseline should use:
      - `optimizer_cpu_offload=False`
      - `optimizer_offload_fraction=0.0`
      - `use_precision_aware_optimizer=False`
