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

All reduced-scale recipes retain GBS=2048 from the 64-GPU template. Override
`train.global_batch_size` when running the GBS 16, 256, or 1024 experiments.
