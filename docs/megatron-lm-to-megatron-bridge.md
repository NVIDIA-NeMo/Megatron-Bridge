# Megatron-LM to Megatron Bridge Guide

Megatron Bridge is Python-first: configure models, data, and training via typed Python APIs. All configuration lives in a structured `ConfigContainer` (see [Configuration overview](training/config-container-overview.md)). Any field can be overridden from the command line using Hydra/OmegaConf syntax in the example training scripts.

## Quick start

Run your example training entrypoint and override config keys directly:

```bash
python examples/recipes/llama/pretrain_llama3_8b.py \
  train.micro_batch_size=2 \
  train.global_batch_size=128 \
  model.num_layers=32 model.hidden_size=4096 model.num_attention_heads=32 \
  model.max_position_embeddings=4096 \
  dataset.sequence_length=4096 \
  checkpoint.save=/workspace/ckpts checkpoint.save_interval=1000 \
  logger.wandb_project=my_proj logger.wandb_exp_name=exp1
```

Notes:
- Config groups are nested: `rng`, `train`, `model`, `optimizer`, `ddp`, `scheduler`, `dataset`, `logger`, `tokenizer`, `checkpoint`, `dist`, `profiling`, `peft`, `comm_overlap`, `mixed_precision`, `inprocess_restart`.
- After overrides are applied, runtime validation computes any dependent fields (e.g., data-parallel size, scheduler steps) and checks consistency.

## Mapping Megatron-LM arguments to Megatron Bridge config

Below is a concise mapping from common `megatron-lm/megatron/training/arguments.py` flags to the new dataclass fields. If a field is not listed here (e.g., highly model-specific knobs), it typically lives under `model.*`, `optimizer.*`, `dataset.*`, or `tokenizer.*` with similar names.


### Model topology and parallelisms

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--tensor-model-parallel-size` | `model.tensor_model_parallel_size` | TP degree. |
| `--pipeline-model-parallel-size` | `model.pipeline_model_parallel_size` | PP degree. |
| `--context-parallel-size` | `model.context_parallel_size` | CP degree. |
| `--expert-model-parallel-size` | `model.expert_model_parallel_size` | EP degree. |
| `--expert-tensor-parallel-size` | `model.expert_tensor_parallel_size` | Expert TP degree. |
| `--sequence-parallel` | `model.sequence_parallel` | Enable sequence parallelism. |
| `--account-for-embedding-in-pipeline-split` | `model.account_for_embedding_in_pipeline_split` | Asymmetric PP: embeddings. |
| `--account-for-loss-in-pipeline-split` | `model.account_for_loss_in_pipeline_split` | Asymmetric PP: loss. |

### Model architecture knobs

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--untie-embeddings-and-output-weights` | `model.share_embeddings_and_output_weights=false` | Untie embeddings/output. |
| `--position-embedding-type` | `model.position_embedding_type` | `learned_absolute` or `rope`. |
| `--rotary-percent` | `model.rotary_percent` | Fraction of rotary dims. |
| `--rotary-base` | `model.rotary_base` | RoPE base. |
| `--rotary-seq-len-interpolation-factor` | `model.seq_len_interpolation_factor` | RoPE interpolation factor. |
| `--normalization` | `model.normalization` | LayerNorm/RMSNorm, etc. |
| `--swiglu` | `model.gated_linear_unit=true` | Enable SwiGLU MLP. |
| `--norm-epsilon` | `model.layernorm_epsilon` | Epsilon for norm layers. |
| `--num-layers` | `model.num_layers` | Number of transformer layers. |
| `--hidden-size` | `model.hidden_size` | Model hidden size. |
| `--ffn-hidden-size` | `model.ffn_hidden_size` | MLP expansion size. |
| `--num-attention-heads` | `model.num_attention_heads` | Attention heads. |
| `--kv-channels` | `model.kv_channels` | Key/Value channels per head. |
| `--group-query-attention` | `model.num_query_groups` | Set groups (enable GQA). |
| `--num-query-groups` | `model.num_query_groups` | Number of query groups. |
| `--qk-layernorm` | `model.qk_layernorm` | Enable QK LayerNorm. |
| `--seq-length` | `model.seq_length` | Max model sequence length. |
| `--max-position-embeddings` | `model.seq_length` | Alias used by HF conversions. |
| `--make-vocab-size-divisible-by` | `model.make_vocab_size_divisible_by` | TP padding multiple. |
| `--disable-bias-linear` | `model.add_bias_linear=false` | Disable linear bias. |
| `--use-flash-attn` | `model.attention_backend=flash` | Use FlashAttention backend. |
| `--init-method-std` | `model.init_method_std` | Weight init standard deviation. |
| `--attention-dropout` | `model.attention_dropout` | Attention dropout. |
| `--hidden-dropout` | `model.hidden_dropout` | Hidden dropout. |

### MoE

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--num-experts` | `model.num_moe_experts` | Experts per MoE layer. |
| `--moe-ffn-hidden-size` | `model.moe_ffn_hidden_size` | Expert MLP hidden size. |
| `--moe-router-load-balancing-type` | `model.moe_router_load_balancing_type` | e.g., aux_loss or seq_aux_loss. |
| `--moe-router-topk` | `model.moe_router_topk` | Top-k experts per token. |
| `--moe-router-pre-softmax` | `model.moe_router_pre_softmax` | Pre-softmax routing. |
| `--moe-grouped-gemm` | `model.moe_grouped_gemm` | Grouped GEMM for MoE. |
| `--moe-aux-loss-coeff` | `model.moe_aux_loss_coeff` | Aux loss coefficient. |
| `--moe-token-dispatcher-type` | `model.moe_token_dispatcher_type` | Token dispatcher: alltoall or flex. |
| `--moe-enable-deepep` | `model.moe_enable_deepep` | Enable DeepEP optimizations. |
| `--moe-permute-fusion` | `model.moe_permute_fusion` | Enable MoE permute fusion. |
| `--moe-router-fusion` | `model.moe_router_fusion` | Enable MoE router fusion. |
| `--moe-router-dtype` | `model.moe_router_dtype` | Router dtype (e.g., fp32). |

### Mixed precision

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--bf16` | `mixed_precision` preset (e.g., "bf16_mixed") | Select a mixed-precision recipe; sets `model.bf16`/`optimizer.bf16`. |

Mixed precision is selected via the `mixed_precision` config key (e.g., preset names like `bf16_mixed`, `bf16`, or `fp16`, depending on your codebase) and is applied to `model`, `optimizer`, and `ddp` during `runtime_config_update`.

### Training

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--micro-batch-size` | `train.micro_batch_size` | Per-rank batch size before gradient accumulation. |
| `--global-batch-size` | `train.global_batch_size` | Total batch across DP and micro-batches. |
| `--train-samples` | `train.train_samples` | Total training samples (sample-based mode). |
| `--rampup-batch-size` | `train.rampup_batch_size` | Start size, increment, and sample count for linear batch ramp-up. |
| `--decrease-batch-size-if-needed` | `train.decrease_batch_size_if_needed` | Adjust GBS to remain divisible when DP changes. |
| `--empty-unused-memory-level` | `train.empty_unused_memory_level` | PyTorch CUDA empty_cache cadence (0, 1, or 2). |
| `--check-weight-hash-across-dp-replicas-interval` | `train.check_weight_hash_across_dp_replicas_interval` | Interval to validate DP weight consistency. |
| `--train-iters` | `train.train_iters` | Number of training iterations. |
| `--exit-interval` | `train.exit_interval` | Exit when iteration % interval == 0. |
| `--exit-duration-in-mins` | `train.exit_duration_in_mins` | Exit after N minutes. |
| `--exit-signal-handler` | `train.exit_signal_handler` | Save and shut down on SIGTERM. |
| `--manual-gc` | `train.manual_gc` | Enable manual Python GC scheduling. |
| `--manual-gc-interval` | `train.manual_gc_interval` | Steps between manual GC runs. |
| `--no-manual-gc-eval` | `train.manual_gc_eval=false` | Disable GC at eval boundaries. |
| `--eval-iters` | `train.eval_iters` | Eval iterations per validation run. |
| `--eval-interval` | `train.eval_interval` | Steps between validations. |
| `--skip-train` | `train.skip_train` | Skip training loop (eval-only). |

### Scheduler / Regularization

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--lr-decay-style` | `scheduler.lr_decay_style` | LR schedule: constant/linear/cosine/ISR/WSD. |
| `--lr-decay-iters` | `scheduler.lr_decay_iters` | Iterations over which to decay LR. |
| `--lr-wsd-decay-style` | `scheduler.lr_wsd_decay_style` | WSD anneal style. |
| `--lr-wsd-decay-iters` | `scheduler.lr_wsd_decay_iters` | Iterations for WSD anneal phase. |
| `--lr-warmup-fraction` | `scheduler.lr_warmup_fraction` | Warmup as fraction of decay span. |
| `--lr-warmup-iters` | `scheduler.lr_warmup_iters` | Warmup iterations (absolute). |
| `--lr-warmup-init` | `scheduler.lr_warmup_init` | Initial LR at start of warmup. |
| `--lr-decay-samples` | `scheduler.lr_decay_samples` | Samples over which to decay LR (sample-based training). |
| `--lr-warmup-samples` | `scheduler.lr_warmup_samples` | Warmup samples (sample-based training). |
| `--lr` | `optimizer.lr` | Base learning rate. |
| `--min-lr` | `optimizer.min_lr` | Minimum learning rate. |
| `--clip-grad` | `optimizer.clip_grad` | Gradient clipping value. |
| `--weight-decay` | `optimizer.weight_decay` | Weight decay. |
| `--adam-beta1` | `optimizer.adam_beta1` | Adam beta1. |
| `--adam-beta2` | `optimizer.adam_beta2` | Adam beta2. |
| `--override-opt_param-scheduler` | `scheduler.override_opt_param_scheduler` | Ignore ckpt scheduler and use config. |
| `--use-checkpoint-opt_param-scheduler` | `scheduler.use_checkpoint_opt_param_scheduler` | Load scheduler from checkpoint. |
| `--start-weight-decay` | `scheduler.start_weight_decay` | WD at start (non-constant modes). |
| `--end-weight-decay` | `scheduler.end_weight_decay` | WD at end (non-constant modes). |
| `--weight-decay-incr-style` | `scheduler.weight_decay_incr_style` | WD schedule: constant/linear/cosine. |

### Checkpointing

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--save` | `checkpoint.save` | Directory to write checkpoints. |
| `--save-interval` | `checkpoint.save_interval` | Iterations between persistent saves. |
| `--no-save-optim` | `checkpoint.save_optim=false` | Do not save optimizer state. |
| `--no-save-rng` | `checkpoint.save_rng=false` | Do not save RNG state. |
| `--load` | `checkpoint.load` | Directory to load from. |
| `--no-load-optim` | `checkpoint.load_optim=false` | Do not load optimizer state. |
| `--load-main-params-from-ckpt` | `checkpoint.load_main_params_from_ckpt` | Load FP32 main params directly. |
| `--no-load-rng` | `checkpoint.load_rng=false` | Do not load RNG state. |
| `--non-persistent-save-interval` | `checkpoint.non_persistent_save_interval` | Frequency for ephemeral saves. |
| `--non-persistent-ckpt-type` | `checkpoint.non_persistent_ckpt_type` | Kind of ephemeral checkpoint (global/local/memory). |
| `--non-persistent-global-ckpt-dir` | `checkpoint.non_persistent_global_ckpt_dir` | Dir for global ephemeral saves. |
| `--non-persistent-local-ckpt-dir` | `checkpoint.non_persistent_local_ckpt_dir` | Dir for local-per-rank ephemeral saves. |
| `--non-persistent-local-ckpt-algo` | `checkpoint.non_persistent_local_ckpt_algo` | Local save algorithm selection. |
| `--finetune` | `checkpoint.finetune` | Load weights, reset iters, no optim/rng. |
| `--pretrained-checkpoint` | `checkpoint.pretrained_checkpoint` | Path to pretrained weights for finetune/SFT. |
| `--ckpt-step` | `checkpoint.ckpt_step` | Explicit step to load. |
| `--use-checkpoint-args` | `checkpoint.use_checkpoint_args` | Override model args from checkpoint metadata. |
| `--exit-on-missing-checkpoint` | `checkpoint.exit_on_missing_checkpoint` | Exit if `load` not found. |
| `--ckpt-format` | `checkpoint.ckpt_format` | Format: torch_dist/zarr/fsdp_dtensor. |
| `--ckpt-convert-format` | `checkpoint.ckpt_convert_format` | Conversion target format. |
| `--ckpt-convert-save` | `checkpoint.ckpt_convert_save` | Output dir for converted ckpt. |
| `--no-ckpt-fully-parallel-save` | `checkpoint.fully_parallel_save=false` | Disable DP-parallel save. |
| `--async-save` | `checkpoint.async_save` | Enable async saves (torch_dist only). |
| `--use-persistent-ckpt-worker` | `checkpoint.use_persistent_ckpt_worker` | Background worker for async saves. |
| `--ckpt-fully-parallel-load` | `checkpoint.fully_parallel_load` | Enable DP-parallel load. |
| `--ckpt-assume-constant-structure` | `checkpoint.ckpt_assume_constant_structure` | Optimize for fixed structure. |
| `--dist-ckpt-strictness` | `checkpoint.dist_ckpt_strictness` | Handling of key mismatches on load. |
| `--auto-detect-ckpt-format` | `checkpoint.auto_detect_ckpt_format` | Auto-detect checkpoint format on load. |
| `--replication` | `checkpoint.replication` | Enable replication of local checkpoints. |
| `--replication-jump` | `checkpoint.replication_jump` | Spacing between replica ranks. |
| `--replication-factor` | `checkpoint.replication_factor` | Number of replicas. |
| `--no-strict-fsdp-dtensor-load` | `checkpoint.strict_fsdp_dtensor_load=false` | Relax FSDP-DTensor strict load. |

### Logging

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--log-interval` | `logger.log_interval` | Steps between console logs. |
| `--log-params-norm` | `logger.log_params_norm` | Compute and log parameter L2 norm. |
| `--log-throughput` | `logger.log_throughput` | Log tokens/sec per GPU. |
| `--log-progress` | `logger.log_progress` | Write progress.txt with tokens and FLOPs. |
| `--timing-log-level` | `logger.timing_log_level` | 0=min; 1=coarse ops; 2=many ops. |
| `--timing-log-option` | `logger.timing_log_option` | max/minmax/all across ranks. |
| `--tensorboard-dir` | `logger.tensorboard_dir` | TensorBoard log directory. |
| `--tensorboard-log-interval` | `logger.tensorboard_log_interval` | Steps between TB events. |
| `--tensorboard-queue-size` | `logger.tensorboard_queue_size` | Pending TB event queue size. |
| `--log-timers-to-tensorboard` | `logger.log_timers_to_tensorboard` | Write timers to TB. |
| `--no-log-loss-scale-to-tensorboard` | `logger.log_loss_scale_to_tensorboard=false` | Disable loss-scale TB logs. |
| `--log-validation-ppl-to-tensorboard` | `logger.log_validation_ppl_to_tensorboard` | Write validation perplexity (ppl) to TB. |
| `--log-memory-to-tensorboard` | `logger.log_memory_to_tensorboard` | Enable memory stats in TB. |
| `--log-world-size-to-tensorboard` | `logger.log_world_size_to_tensorboard` | Log world size in TB. |
| `--wandb-project` | `logger.wandb_project` | Weights & Biases project. |
| `--wandb-entity` | `logger.wandb_entity` | Weights & Biases entity/team. |
| `--wandb-exp-name` | `logger.wandb_exp_name` | Run name in W&B. |
| `--wandb-save-dir` | `logger.wandb_save_dir` | Local directory for W&B artifacts. |
| `--logging-level` | `logger.logging_level` | Python logging level (e.g., 20=INFO). |
| `--log-energy` | `logger.log_energy` | Log energy in Joules (if available). |

### RNG / Initialization

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--seed` | `rng.seed` | Global random seed. |
| `--data-parallel-random-init` | `rng.data_parallel_random_init` | Enable per-DP-rank random init. |
| `--te-rng-tracker` | `rng.te_rng_tracker` | Use TE RNG (needed for CUDA graphs). |
| `--inference-rng-tracker` | `rng.inference_rng_tracker` | RNG tuned for inference stability. |

### Distributed init and topology

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--distributed-backend` | `dist.distributed_backend` | Process group backend (nccl/gloo). |
| `--distributed-timeout-minutes` | `dist.distributed_timeout_minutes` | PG init and collective timeout. |
| `--no-align-grad-reduce` | `dist.align_grad_reduce=false` | Launch DP reduces independently per PP stage. |
| `--disable-gloo-process-groups` | `dist.use_gloo_process_groups=false` | Disable auxiliary Gloo PG creation. |
| `--use-sharp` | `dist.use_sharp` | Enable SHARP collectives for DP PG. |
| `--sharp-enabled-group` | `dist.sharp_enabled_group` | Which DP group enables SHARP. |
| `--high-priority-stream-groups` | `dist.high_priority_stream_groups` | Use high-priority comm streams for groups. |
| `--use-tp-pp-dp-mapping` | `dist.use_tp_pp_dp_mapping` | Use TP-PP-DP rank ordering at init. |

Additional distributed/optimizer overlap settings:

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--use-distributed-optimizer` | `ddp.use_distributed_optimizer` and `optimizer.use_distributed_optimizer` | Enable distributed optimizer; settings are synchronized. |
| `--overlap-grad-reduce` | `ddp.overlap_grad_reduce` | Overlap DP gradient reduce-scatter. |
| `--overlap-param-gather` | `ddp.overlap_param_gather` | Overlap parameter all-gather with fprop. |

### Profiling

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--profile` | `profiling.use_nsys_profiler` | Enable nsys profiling (capture is controlled via external CLI). |
| `--use-pytorch-profiler` | `profiling.use_pytorch_profiler` | Enable PyTorch profiler (TB-friendly). |
| `--profile-step-start` | `profiling.profile_step_start` | Global step to start profiling. |
| `--profile-step-end` | `profiling.profile_step_end` | Global step to stop profiling. |
| `--profile-ranks` | `profiling.profile_ranks` | Global ranks to profile. |
| `--record-memory-history` | `profiling.record_memory_history` | Track memory history. |
| `--memory-snapshot-path` | `profiling.memory_snapshot_path` | Output path for memory snapshot. |
| (shapes) | `profiling.record_shapes` | Record tensor shapes (overhead). |

### In-process restart

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--inprocess-restart` | `inprocess_restart.enabled` | Enable nvrx in-process restart. |
| `--inprocess-max-iterations` | `inprocess_restart.max_iterations` | Max restart attempts. |
| `--inprocess-monitor-thread-interval` | `inprocess_restart.monitor_thread_interval` | Monitor thread polling interval. |
| `--inprocess-monitor-process-interval` | `inprocess_restart.monitor_process_interval` | Monitor process polling interval. |
| `--inprocess-progress-watchdog-interval` | `inprocess_restart.progress_watchdog_interval` | Auto progress timestamp update cadence. |
| `--inprocess-heartbeat-interval` | `inprocess_restart.heartbeat_interval` | Unresponsive-rank heartbeat cadence. |
| `--inprocess-soft-timeout` | `inprocess_restart.soft_timeout` | Soft progress timeout. |
| `--inprocess-hard-timeout` | `inprocess_restart.hard_timeout` | Hard timeout until kill. |
| `--inprocess-heartbeat-timeout` | `inprocess_restart.heartbeat_timeout` | Missing heartbeat timeout. |
| `--inprocess-barrier-timeout` | `inprocess_restart.barrier_timeout` | Timeout for internal barriers. |
| `--inprocess-completion-timeout` | `inprocess_restart.completion_timeout` | Timeout for completion barrier. |
| `--inprocess-last-call-wait` | `inprocess_restart.last_call_wait` | Delay to collect terminal failures. |
| `--inprocess-termination-grace-time` | `inprocess_restart.termination_grace_time` | SIGTERM→SIGKILL grace period. |
| `--inprocess-granularity` | `inprocess_restart.granularity` | Restart granularity (node/rank). |
| `--inprocess-active-world-size` | `inprocess_restart.active_world_size` | Active ranks count; rest are reserve. |
| `--inprocess-empty-cuda-cache` | `inprocess_restart.empty_cuda_cache` | Empty CUDA cache on restart finalize. |

### Straggler detection

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--log-straggler` | `straggler.log_straggler` | Track and log straggler GPUs. |
| `--disable-straggler-on-startup` | `straggler.disable_straggler_on_startup` | Start with straggler detector disabled. |
| `--straggler-ctrlr-port` | `straggler.straggler_ctrlr_port` | Controller port for toggling. |
| `--straggler-minmax-count` | `straggler.straggler_minmax_count` | Num ranks to report for min/max throughput. |

### Rerun state machine

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--error-injection-rate` | `rerun_state_machine.error_injection_rate` | Frequency of injected validation perturbations. |
| `--error-injection-type` | `rerun_state_machine.error_injection_type` | Kind of injection (correct/transient/persistent). |
| `--rerun-mode` | `rerun_state_machine.rerun_mode` | Disabled/validate_results/report_stats. |

### Data / Tokenizer args

| megatron-lm arguments | Megatron Bridge config | Description |
| --- | --- | --- |
| `--tokenizer-type` | `tokenizer.tokenizer_type` | Tokenizer implementation (e.g., HuggingFaceTokenizer). |
| `--tokenizer-model` | `tokenizer.tokenizer_model` | Model name/path for tokenizer. |
| `--num-workers` | `dataset.num_workers` | DataLoader workers. |
| `--no-create-attention-mask-in-dataloader` | `dataset.skip_getting_attention_mask_from_dataset=true` | Use backend-generated masks. |
