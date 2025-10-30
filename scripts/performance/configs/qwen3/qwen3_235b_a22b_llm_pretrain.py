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

from megatron.bridge.recipes.qwen3.qwen3_moe import qwen3_235b_a22b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop

from utils.helpers import (
  get_precision_config, 
  set_basic_perf_overrides, 
  set_cuda_graph_overrides,
  get_user_parallelism_and_batch_size_configs,
  moe_a2a_1f1b_overrides,
)


logger = logging.getLogger(__name__)


def set_qwen3_235b_a22b_specific_overrides(cfg: ConfigContainer) -> ConfigContainer:
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True


def qwen3_235b_a22b_gb200_64gpus_bf16_config(**kwargs) -> ConfigContainer:
    """GB200, 64xGPU, BF16 baseline config."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    cfg = qwen3_235b_a22b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("bf16"),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 8 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp
    cfg.model.expert_model_parallel_size = 8 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 1024 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_qwen3_235b_a22b_specific_overrides(cfg)

    cuda_graph_impl = "local" if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = "full_iteration" if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
      set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

    use_tokendrop = True if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    if use_tokendrop:
      cfg.model = apply_moe_token_drop(cfg.model)

    A2A_1F1B = False if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")
    if A2A_1F1B:
      moe_a2a_1f1b_overrides(cfg)

    return cfg

def qwen3_235b_a22b_gb200_64gpus_fp8_config(**kwargs) -> ConfigContainer:
    """GB200, 64xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")
    cfg = qwen3_235b_a22b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("fp8", fp8_recipe),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 8 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp
    cfg.model.expert_model_parallel_size = 8 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 1024 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_qwen3_235b_a22b_specific_overrides(cfg)

    cuda_graph_impl = "local" if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = "full_iteration" if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
      set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": True})

    use_tokendrop = True if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    if use_tokendrop:
      cfg.model = apply_moe_token_drop(cfg.model)

    A2A_1F1B = False if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")
    if A2A_1F1B:
      moe_a2a_1f1b_overrides(cfg)

    return cfg


def qwen3_235b_a22b_b200_64gpus_bf16_config(**kwargs) -> ConfigContainer:
    """B200, 64xGPU, BF16 baseline config."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    cfg = qwen3_235b_a22b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("bf16"),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 8 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = 2 if vp is None else vp
    cfg.model.expert_model_parallel_size = 8 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 1024 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_qwen3_235b_a22b_specific_overrides(cfg)

    cuda_graph_impl = None if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = None if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
      set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

    use_tokendrop = True if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    if use_tokendrop:
      cfg.model = apply_moe_token_drop(cfg.model)

    A2A_1F1B = False if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")
    if A2A_1F1B:
      moe_a2a_1f1b_overrides(cfg)

    return cfg

def qwen3_235b_a22b_b200_64gpus_fp8_config(**kwargs) -> ConfigContainer:
    """B200, 64xGPU, FP8 cs preset."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")
    cfg = qwen3_235b_a22b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("fp8", fp8_recipe),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 8 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = 2 if vp is None else vp
    cfg.model.expert_model_parallel_size = 8 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 1024 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_qwen3_235b_a22b_specific_overrides(cfg)

    cuda_graph_impl = None if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = None if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
      set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

    use_tokendrop = True if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    if use_tokendrop:
      cfg.model = apply_moe_token_drop(cfg.model)

    A2A_1F1B = False if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")
    if A2A_1F1B:
      moe_a2a_1f1b_overrides(cfg)

    return cfg

def qwen3_235b_a22b_h100_256gpus_bf16_config(**kwargs) -> ConfigContainer:
    """H100, 256xGPU, BF16 baseline config."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    cfg = qwen3_235b_a22b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("bf16"),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 2 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 8 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = 4 if vp is None else vp
    cfg.model.expert_model_parallel_size = 32 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_qwen3_235b_a22b_specific_overrides(cfg)

    cuda_graph_impl = None if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = None if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
      set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

    use_tokendrop = True if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    if use_tokendrop:
      cfg.model = apply_moe_token_drop(cfg.model)

    A2A_1F1B = False if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")
    if A2A_1F1B:
      moe_a2a_1f1b_overrides(cfg)

    return cfg


def qwen3_235b_a22b_h100_256gpus_fp8_config(**kwargs) -> ConfigContainer:
    """H100, 256xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")
    cfg = qwen3_235b_a22b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("fp8", fp8_recipe),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 2 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 8 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = 4 if vp is None else vp
    cfg.model.expert_model_parallel_size = 32 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_qwen3_235b_a22b_specific_overrides(cfg)

    cuda_graph_impl = None if kwargs.get("cuda_graph_impl") is None else kwargs.get("cuda_graph_impl")
    cuda_graph_scope = None if kwargs.get("cuda_graph_scope") is None else kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None:
      set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

    use_tokendrop = True if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    if use_tokendrop:
      cfg.model = apply_moe_token_drop(cfg.model)

    A2A_1F1B = False if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")
    if A2A_1F1B:
      moe_a2a_1f1b_overrides(cfg)

    return cfg
