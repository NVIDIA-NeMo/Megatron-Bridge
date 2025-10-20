import logging

from megatron.bridge.recipes.llama.llama3 import llama3_8b_pretrain_config
from megatron.bridge.training.config import ConfigContainer

from scripts.performance.utils.helpers import get_precision_config, set_megatron_fsdp_overrides, set_basic_perf_overrides

logger = logging.getLogger(__name__)


def llama3_8b_h100_bf16_config(fp8_recipe = None) -> ConfigContainer:
    """H100, 8xGPU, BF16 baseline config."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    return cfg


def llama3_8b_h100_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """H100, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    if fp8_recipe == "cs":
        set_megatron_fsdp_overrides(cfg, perf_overrides={"use_megatron_fsdp": True})
        cfg.ddp.keep_fp8_transpose_cache = True
        cfg.ddp.nccl_ub = True
        cfg.model.gradient_accumulation_fusion = False

    return cfg
