# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepSeek-V4 unit policy until Core's wrapping entry exposes custom FSDP units."""

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_model_config


def wrap_deepseek_v4_fsdp(
    *,
    model: list[MegatronModule],
    data_parallel_random_init: bool,
    ddp_config: DistributedDataParallelConfig,
    overlap_param_gather_with_optimizer_step: bool,
    use_torch_fsdp2: bool,
    pg_collection: ProcessGroupCollection,
    use_layer_wise_distributed_optimizer: bool,
) -> list[MegatronModule]:
    """Wrap DeepSeek-V4 with mHC-aware Megatron FSDP v2 units.

    Split mHC layers call inner Transformer helpers directly, bypassing their
    module hooks. The called mHC wrapper must therefore own their dense weights.
    Selective mHC recomputation can replay several layers before their backward
    hooks run, so that case uses the enclosing HybridStack as the dense unit.
    Core still wraps routed experts separately; remaining MTP projections and
    reused embedding/output weights stay root-owned.

    Args:
        model: DeepSeek-V4 mHC model chunks selected by the common dispatcher.
        data_parallel_random_init: Whether independent DP initialization was requested.
        ddp_config: Configuration selecting Megatron FSDP v2.
        overlap_param_gather_with_optimizer_step: Legacy optimizer-step gather overlap.
        use_torch_fsdp2: Whether the incompatible PyTorch FSDP path was requested.
        pg_collection: Model communication groups.
        use_layer_wise_distributed_optimizer: Whether layer-wise optimization was requested.

    Returns:
        Wrapped chunks with model-specific dense units and native expert units.

    Raises:
        ValueError: An incompatible wrapping or synchronization mode was requested.
    """
    if ddp_config.megatron_fsdp_version != 2:
        raise ValueError("DeepSeek-V4 custom units require Megatron FSDP v2")
    if use_torch_fsdp2 or use_layer_wise_distributed_optimizer or overlap_param_gather_with_optimizer_step:
        raise ValueError("DeepSeek-V4 MFSDP v2 requires its native optimizer and parameter-gather scheduling")
    if data_parallel_random_init:
        raise ValueError("MFSDP v2 does not support data-parallel random initialization")
    if any(get_model_config(chunk).overlap_moe_expert_parallel_comm for chunk in model):
        raise ValueError("DeepSeek-V4 MFSDP v2 requires automatic hooks; EP communication overlap is unsupported")

    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer

    wrapped_model = []
    for chunk in model:
        config = get_model_config(chunk)
        recompute_mhc = config.recompute_granularity == "selective" and "mhc" in (config.recompute_modules or [])
        wrapped_model.append(
            FullyShardedDataParallelV2(
                config=config,
                ddp_config=ddp_config,
                module=chunk,
                fsdp_unit_modules=[HybridStack if recompute_mhc else HyperConnectionHybridLayer],
                disable_bucketing=False,
                pg_collection=pg_collection,
            )
        )
    return wrapped_model
