import logging

from megatron.bridge.recipes.llama import llama3_8b_pretrain_config
from megatron.bridge.training.config import ConfigContainer

try:
    from utils.helpers import get_precision_config, set_megatron_fsdp_overrides, set_basic_perf_overrides, set_cuda_graph_overrides
except (ImportError, ModuleNotFoundError):
    from ..utils.helpers import get_precision_config, set_megatron_fsdp_overrides, set_basic_perf_overrides, set_cuda_graph_overrides

logger = logging.getLogger(__name__)


def llama3_8b_gb200_8gpus_bf16_config(fp8_recipe = None) -> ConfigContainer:
    """GB200, 8xGPU, BF16 baseline config."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 2
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": True})

    return cfg

def llama3_8b_gb200_8gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """GB200, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
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
    cfg.train.micro_batch_size = 2
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": True})

    return cfg


def llama3_8b_b200_8gpus_bf16_config(**kwargs) -> ConfigContainer:
    """B200, 8xGPU, BF16 baseline config."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = kwargs.get("tensor_model_parallel_size", 1)
    cfg.model.pipeline_model_parallel_size = kwargs.get("pipeline_model_parallel_size", 1)
    cfg.model.context_parallel_size = kwargs.get("context_parallel_size", 1)
    cfg.model.virtual_pipeline_model_parallel_size = kwargs.get("virtual_pipeline_model_parallel_size", None)
    cfg.model.expert_model_parallel_size = kwargs.get("expert_model_parallel_size", 1)
    cfg.model.expert_tensor_parallel_size = kwargs.get("expert_tensor_parallel_size", None)
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = kwargs.get("global_batch_size", 128)
    cfg.train.micro_batch_size = kwargs.get("micro_batch_size", 2)
    cfg.model.seq_length = kwargs.get("seq_length", 8192)
    cfg.dataset.sequence_length = kwargs.get("seq_length", 8192)

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    
    enable_cuda_graphs = kwargs.get("enable_cuda_graphs", True)
    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": enable_cuda_graphs})

    return cfg

def llama3_8b_b200_8gpus_fp8_config(**kwargs) -> ConfigContainer:
    """B200, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    fp8_recipe = kwargs.get("fp8_recipe", "cs")
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = kwargs.get("tensor_model_parallel_size", 1)
    cfg.model.pipeline_model_parallel_size = kwargs.get("pipeline_model_parallel_size", 1)
    cfg.model.context_parallel_size = kwargs.get("context_parallel_size", 1)
    cfg.model.virtual_pipeline_model_parallel_size = kwargs.get("virtual_pipeline_model_parallel_size", None)
    cfg.model.expert_model_parallel_size = kwargs.get("expert_model_parallel_size", 1)
    cfg.model.expert_tensor_parallel_size = kwargs.get("expert_tensor_parallel_size", None)
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = kwargs.get("global_batch_size", 128)
    cfg.train.micro_batch_size = kwargs.get("micro_batch_size", 2)
    cfg.model.seq_length = kwargs.get("seq_length", 8192)
    cfg.dataset.sequence_length = kwargs.get("seq_length", 8192)

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    enable_cuda_graphs = kwargs.get("enable_cuda_graphs", True)
    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": enable_cuda_graphs})

    return cfg


def llama3_8b_h100_8gpus_bf16_config(fp8_recipe = None) -> ConfigContainer:
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


def llama3_8b_h100_8gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
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
