# Reduced-scale Qwen3-235B GB300 recipes

These recipes derive from
`qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config` and retain its BF16
precision, sequence length, global batch size, HybridEP transport, kernels,
and 50-step benchmark schedule.

`qwen3_235b_a22b_pretrain_32gpu_gb300_bf16_config` uses TP=1, PP=2,
CP=1, EP=16, and MBS=1. MBS=2 has already been observed to run out of
memory on this topology.

`qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_config` uses TP=1, PP=1,
CP=1, EP=16, and MBS=1. It enables Megatron-FSDP
`optim_grads_params` sharding for dense and expert data-parallel groups,
full activation recompute, meta-device initialization, and eager execution.
It does not enable CPU offload, FP8, or reduced-precision optimizer states.

`qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_zero2_config` keeps the same
validated topology and memory policy but uses `optim_grads` ZeRO-2 sharding.
It is the performance candidate when the additional replicated-parameter
memory fits.

`qwen3_235b_a22b_pretrain_16gpu_gb300_bf16_common_contract_config` matches
the GB/VR comparison contract: distributed optimizer, BF16 Adam moments,
no recompute, and TE router/preprocess graphs.

Three diagnostic variants isolate the remaining activation-memory choices:

- `..._zero2_selective_config` uses eager execution and recomputes only
  `moe_act`.
- `..._zero2_no_recompute_config` is the eager no-recompute control.
- `..._zero2_selective_graph_config` adds TE scoped graphs to the selective
  candidate.

TP/PP/EP mapping candidates are also provided for matched short-run tuning:

- 16-GPU `..._zero2_tp2_config` and `..._zero2_tp2_selective_config` test
  whether TP2+SP creates enough activation headroom to reduce recompute.
- 32-GPU `..._pp1_config`, `..._tp2_pp1_config`, and
  `..._tp2_pp1_mbs2_config` remove PP bubbles and compare TP/MBS trade-offs.
- 32-GPU `..._pp1_mbs2_config` tests the winning TP1/PP1/EP32 mapping at
  the next power-of-two micro batch size.
- 16-GPU TP2 selective/full-recompute graph variants probe whether additional
  TP headroom can support TE-scoped or Megatron-FSDP full-iteration graphs.
- 32-GPU SM16/24/40/48 variants tune HybridEP communication SM allocation
  against the default 32-SM winner; chunk64/256 variants bracket the default
  128-token combine chunk.

All reduced-scale recipes retain GBS=2048 from the 64-GPU template. Override
`train.global_batch_size` when running the GBS 16, 256, or 1024 experiments.
