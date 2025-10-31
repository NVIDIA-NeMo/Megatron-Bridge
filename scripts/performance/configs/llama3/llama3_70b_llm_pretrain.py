# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from utils.helpers import (
    get_precision_config,
    get_user_parallelism_and_batch_size_configs,
    set_basic_perf_overrides,
    set_cuda_graph_overrides,
    set_megatron_fsdp_overrides,
    set_recompute_overrides,
)

from megatron.bridge.recipes.llama import llama3_70b_pretrain_config
from megatron.bridge.training.comm_overlap import (
    userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def llama3_70b_gb200_64gpus_bf16_config(**kwargs) -> ConfigContainer:
    """GB200, 64xGPU, BF16 baseline config."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))

    cuda_graph_impl = None if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = None if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
        set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

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

    use_megatron_fsdp = True if kwargs.get("use_megatron_fsdp") is None else kwargs.get("use_megatron_fsdp")
    set_megatron_fsdp_overrides(cfg, perf_overrides={"use_megatron_fsdp": use_megatron_fsdp})
    cfg.ddp.fsdp_double_buffer = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.ddp.suggested_communication_unit_size = 800000000
    set_recompute_overrides(cfg, cpu_offloading_num_layers=20)

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_gb200_64gpus_fp8_config(**kwargs) -> ConfigContainer:
    """GB200, 64xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))

    cuda_graph_impl = None if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = None if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
        set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

    if fp8_recipe == "cs":
        cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
        cfg.model.pipeline_model_parallel_size = 1 if pp is None else pp
        cfg.model.context_parallel_size = 1 if cp is None else cp
        cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp

        cfg.train.micro_batch_size = 2 if mbs is None else mbs

        use_megatron_fsdp = True if kwargs.get("use_megatron_fsdp") is None else kwargs.get("use_megatron_fsdp")
        set_megatron_fsdp_overrides(cfg, perf_overrides={"use_megatron_fsdp": use_megatron_fsdp})
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False
        cfg.ddp.suggested_communication_unit_size = 800000000
        set_recompute_overrides(cfg, cpu_offloading_num_layers=40)

    if fp8_recipe == "mx":
        cfg.model.tensor_model_parallel_size = 2 if tp is None else tp
        cfg.model.pipeline_model_parallel_size = 4 if pp is None else pp
        cfg.model.context_parallel_size = 1 if cp is None else cp
        cfg.model.virtual_pipeline_model_parallel_size = 5 if vp is None else vp

        cfg.train.micro_batch_size = 1 if mbs is None else mbs

    cfg.model.expert_model_parallel_size = 1 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = None if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128 if gbs is None else gbs
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_b200_64gpus_bf16_config(**kwargs) -> ConfigContainer:
    """B200, 64xGPU, BF16 baseline config."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))

    cuda_graph_impl = "local" if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = "full_iteration" if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
        set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

    cfg.model.tensor_model_parallel_size = 2 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 4 if pp is None else pp
    cfg.model.context_parallel_size = 2 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = 5 if vp is None else vp
    cfg.model.expert_model_parallel_size = 1 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = None if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_b200_64gpus_fp8_config(**kwargs) -> ConfigContainer:
    """B200, 64xGPU, FP8 cs preset."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))

    cuda_graph_impl = None if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = None if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
        set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

    if fp8_recipe == "cs":
        cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
        cfg.model.pipeline_model_parallel_size = 1 if pp is None else pp
        cfg.model.context_parallel_size = 1 if cp is None else cp
        cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp

        use_megatron_fsdp = True if kwargs.get("use_megatron_fsdp") is None else kwargs.get("use_megatron_fsdp")
        set_megatron_fsdp_overrides(cfg, perf_overrides={"use_megatron_fsdp": use_megatron_fsdp})
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False
        cfg.ddp.suggested_communication_unit_size = 800000000
        set_recompute_overrides(cfg, recompute_num_layers=5)

    if fp8_recipe == "mx":
        cfg.model.tensor_model_parallel_size = 2 if tp is None else tp
        cfg.model.pipeline_model_parallel_size = 4 if pp is None else pp
        cfg.model.context_parallel_size = 1 if cp is None else cp
        cfg.model.virtual_pipeline_model_parallel_size = 5 if vp is None else vp

    cfg.model.expert_model_parallel_size = 1 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = None if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_h100_64gpus_bf16_config(**kwargs) -> ConfigContainer:
    """H100, 64xGPU, BF16 baseline config."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))

    cuda_graph_impl = None if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = None if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
        set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

    cfg.model.tensor_model_parallel_size = 4 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 4 if pp is None else pp
    cfg.model.context_parallel_size = 2 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = 5 if vp is None else vp
    cfg.model.expert_model_parallel_size = 1 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = None if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192

    return cfg


def llama3_70b_h100_64gpus_fp8_config(**kwargs) -> ConfigContainer:
    """H100, 64xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))
    cuda_graph_impl = None if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = None if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
        set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

    cfg.model.tensor_model_parallel_size = 4 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 8 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = 5 if vp is None else vp
    cfg.model.expert_model_parallel_size = 1 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = None if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192

    return cfg
