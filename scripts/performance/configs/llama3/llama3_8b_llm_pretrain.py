import logging

from megatron.bridge.recipes.llama import llama3_8b_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.comm_overlap import CommOverlapConfig

from utils.helpers import (
    get_precision_config, 
    set_megatron_fsdp_overrides, 
    set_basic_perf_overrides, 
    set_cuda_graph_overrides,
    get_user_parallelism_and_batch_size_configs,
)


logger = logging.getLogger(__name__)


def llama3_8b_gb200_8gpus_bf16_config(**kwargs) -> ConfigContainer:
    """GB200, 8xGPU, BF16 baseline config."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 1 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp
    cfg.model.expert_model_parallel_size = 1 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = None if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128 if gbs is None else gbs
    cfg.train.micro_batch_size = 2 if mbs is None else mbs
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": True})

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1),
    )
    cfg.tokenizer.vocab_size = 128256
    cfg.model.cuda_graph_scope = "full_iteration"
    cfg.model.should_pad_vocab = True

    return cfg

def llama3_8b_gb200_8gpus_fp8_config(**kwargs) -> ConfigContainer:
    """GB200, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 1 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp
    cfg.model.expert_model_parallel_size = 1 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = None if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128 if gbs is None else gbs
    cfg.train.micro_batch_size = 2 if mbs is None else mbs
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": True})

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1),
    )
    cfg.tokenizer.vocab_size = 128256
    cfg.model.cuda_graph_scope = "full_iteration"
    cfg.model.should_pad_vocab = True

    return cfg


def llama3_8b_b200_8gpus_bf16_config(**kwargs) -> ConfigContainer:
    """B200, 8xGPU, BF16 baseline config."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 1 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp
    cfg.model.expert_model_parallel_size = 1 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = None if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128 if gbs is None else gbs
    cfg.train.micro_batch_size = 2 if mbs is None else mbs
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": True})

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1),
    )
    cfg.tokenizer.vocab_size = 128256
    cfg.model.cuda_graph_scope = "full_iteration"
    cfg.model.should_pad_vocab = True

    return cfg

def llama3_8b_b200_8gpus_fp8_config(**kwargs) -> ConfigContainer:
    """B200, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 1 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp
    cfg.model.expert_model_parallel_size = 1 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = None if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128 if gbs is None else gbs
    cfg.train.micro_batch_size = 2 if mbs is None else mbs
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": True})

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1),
    )
    cfg.tokenizer.vocab_size = 128256
    cfg.model.cuda_graph_scope = "full_iteration"
    cfg.model.should_pad_vocab = True

    return cfg


def llama3_8b_h100_8gpus_bf16_config(**kwargs) -> ConfigContainer:
    """H100, 8xGPU, BF16 baseline config."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 1 if pp is None else pp
    cfg.model.context_parallel_size = 2 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp
    cfg.model.expert_model_parallel_size = 1 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = None if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1),
    )
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    return cfg


def llama3_8b_h100_8gpus_fp8_config(**kwargs) -> ConfigContainer:
    """H100, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 1 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp
    cfg.model.expert_model_parallel_size = 1 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = None if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    if fp8_recipe == "cs":
        use_megatron_fsdp = True if kwargs.get("use_megatron_fsdp") is None else kwargs.get("use_megatron_fsdp")
        set_megatron_fsdp_overrides(cfg, perf_overrides={"use_megatron_fsdp": use_megatron_fsdp})
        cfg.ddp.nccl_ub = True
        cfg.model.gradient_accumulation_fusion = False
    
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1),
    )
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    return cfg
