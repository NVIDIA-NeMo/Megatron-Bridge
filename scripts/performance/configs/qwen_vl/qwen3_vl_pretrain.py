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

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.qwen_vl.qwen3_vl import (
    qwen3_vl_30b_a3b_pretrain_mock_config,
    qwen3_vl_235b_a22b_pretrain_mock_config,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def set_qwen3_vl_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Qwen3-VL configs."""
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.apply_rope_fusion = False

    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True  # required for token dropless

    # CUDA graphs for moe_router and moe_preprocess only — "attn" scope is not compatible with Qwen3VLModel
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    # qwen_vl does not support overlap_grad_reduce=True and overlap_param_gather=True in current implementation
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False
    cfg.comm_overlap.overlap_param_gather = False
    cfg.comm_overlap.overlap_grad_reduce = False


def _qwen3_vl_pretrain_config(
    model_recipe_name: str,
    gpu: str,
    recipe_fn,
    precision: str = "bf16",
    mock: bool = True,
    config_variant: str = "v1",
    tp_comm_overlap: bool = True,
) -> ConfigContainer:
    """Build a Qwen3-VL pretrain config for a given model size and GPU."""
    base_cfg = get_workload_base_config(
        model_family_name="qwen_vl",
        model_recipe_name=model_recipe_name,
        gpu=gpu,
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = recipe_fn(
        mock=mock,
        precision_config=precision_config,
        comm_overlap_config=CommOverlapConfig(tp_comm_overlap=tp_comm_overlap),
        moe_flex_dispatcher_backend=base_cfg.moe_flex_dispatcher_backend,
    )
    set_workload_base_configs(cfg, base_cfg)
    set_qwen3_vl_common_configs(cfg)

    return cfg


# =============================================================================
# Qwen3-VL 235B-A22B pretrain configs
# =============================================================================


def qwen3_vl_235b_a22b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    return _qwen3_vl_pretrain_config(
        "qwen3_vl_235b_a22b", "gb300", qwen3_vl_235b_a22b_pretrain_mock_config,
        precision=precision, mock=mock, config_variant=config_variant,
    )


def qwen3_vl_235b_a22b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    cfg = _qwen3_vl_pretrain_config(
        "qwen3_vl_235b_a22b", "gb200", qwen3_vl_235b_a22b_pretrain_mock_config,
        precision=precision, mock=mock, config_variant=config_variant,
    )
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
    cfg.model.num_layers_in_first_pipeline_stage = 10
    cfg.model.num_layers_in_last_pipeline_stage = 12
    return cfg


def qwen3_vl_235b_a22b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    cfg = _qwen3_vl_pretrain_config(
        "qwen3_vl_235b_a22b", "b200", qwen3_vl_235b_a22b_pretrain_mock_config,
        precision=precision, mock=mock, config_variant=config_variant,
    )

    if precision == "fp8_mx":  # keeping this enabled causes NaN grad norm
        cfg.comm_overlap.overlap_param_gather = False
        cfg.ddp.overlap_param_gather = False
        cfg.optimizer.overlap_param_gather = False

    return cfg


def qwen3_vl_235b_a22b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    return _qwen3_vl_pretrain_config(
        "qwen3_vl_235b_a22b", "h100", qwen3_vl_235b_a22b_pretrain_mock_config,
        precision=precision, mock=mock, config_variant=config_variant,
        tp_comm_overlap=False,
    )


# =============================================================================
# Qwen3-VL 30B-A3B pretrain configs
# =============================================================================


def qwen3_vl_30b_a3b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    return _qwen3_vl_pretrain_config(
        "qwen3_vl_30b_a3b", "gb300", qwen3_vl_30b_a3b_pretrain_mock_config,
        precision=precision, mock=mock, config_variant=config_variant,
    )


def qwen3_vl_30b_a3b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    cfg = _qwen3_vl_pretrain_config(
        "qwen3_vl_30b_a3b", "gb200", qwen3_vl_30b_a3b_pretrain_mock_config,
        precision=precision, mock=mock, config_variant=config_variant,
    )
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
    return cfg


def qwen3_vl_30b_a3b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    return _qwen3_vl_pretrain_config(
        "qwen3_vl_30b_a3b", "b200", qwen3_vl_30b_a3b_pretrain_mock_config,
        precision=precision, mock=mock, config_variant=config_variant,
    )


def qwen3_vl_30b_a3b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    return _qwen3_vl_pretrain_config(
        "qwen3_vl_30b_a3b", "h100", qwen3_vl_30b_a3b_pretrain_mock_config,
        precision=precision, mock=mock, config_variant=config_variant,
        tp_comm_overlap=True,
    )
