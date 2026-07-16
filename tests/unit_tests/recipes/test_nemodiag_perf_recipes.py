# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Tests for the standalone NeMoDiag V0 performance recipes."""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.bridge.perf_recipes.nemodiag.common import NEMODIAG_V0_PIPELINE_LAYOUT
from megatron.bridge.perf_recipes.nemodiag.gb300.nemodiag_v0 import (
    nemodiag_v0_pretrain_72gpu_gb300_bf16_perf72_e144_config,
    nemodiag_v0_pretrain_72gpu_gb300_fp8mx_perf72_e144_config,
    nemodiag_v0_pretrain_72gpu_gb300_nvfp4_perf72_e144_config,
    nemodiag_v0_pretrain_144gpu_gb300_bf16_perf72_e144_config,
    nemodiag_v0_pretrain_144gpu_gb300_fp8mx_perf72_e144_config,
    nemodiag_v0_pretrain_144gpu_gb300_nvfp4_perf72_e144_config,
    nemodiag_v0_pretrain_288gpu_gb300_bf16_perf72_e144_config,
    nemodiag_v0_pretrain_288gpu_gb300_fp8mx_perf72_e144_config,
    nemodiag_v0_pretrain_288gpu_gb300_nvfp4_perf72_e144_config,
)
from megatron.bridge.training.config import ConfigContainer


_RECIPE_CASES = [
    (nemodiag_v0_pretrain_72gpu_gb300_bf16_perf72_e144_config, 72, "bf16", 1152, 1, ["moe_act"]),
    (nemodiag_v0_pretrain_144gpu_gb300_bf16_perf72_e144_config, 144, "bf16", 2304, 1, ["moe_act"]),
    (nemodiag_v0_pretrain_288gpu_gb300_bf16_perf72_e144_config, 288, "bf16", 4608, 1, ["moe_act"]),
    (nemodiag_v0_pretrain_72gpu_gb300_fp8mx_perf72_e144_config, 72, "fp8_mx", 1152, 1, []),
    (nemodiag_v0_pretrain_144gpu_gb300_fp8mx_perf72_e144_config, 144, "fp8_mx", 2304, 1, []),
    (nemodiag_v0_pretrain_288gpu_gb300_fp8mx_perf72_e144_config, 288, "fp8_mx", 4608, 1, []),
    (nemodiag_v0_pretrain_72gpu_gb300_nvfp4_perf72_e144_config, 72, "nvfp4", 1152, 2, ["mla_up_proj"]),
    (nemodiag_v0_pretrain_144gpu_gb300_nvfp4_perf72_e144_config, 144, "nvfp4", 2304, 2, ["mla_up_proj"]),
    (nemodiag_v0_pretrain_288gpu_gb300_nvfp4_perf72_e144_config, 288, "nvfp4", 4608, 2, ["mla_up_proj"]),
]


@pytest.mark.parametrize("recipe_fn,num_gpus,precision,gbs,mbs,recompute_modules", _RECIPE_CASES)
def test_nemodiag_v0_recipe_is_standalone_and_fixed(
    recipe_fn,
    num_gpus: int,
    precision: str,
    gbs: int,
    mbs: int,
    recompute_modules: list[str],
):
    assert not inspect.signature(recipe_fn).parameters

    cfg = recipe_fn()

    assert isinstance(cfg, ConfigContainer)
    assert isinstance(cfg.model, MLAModelProvider)
    assert cfg.model.hf_model_id is None
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.tokenizer.tokenizer_model is None
    assert cfg.train.global_batch_size == gbs
    assert cfg.train.global_batch_size / num_gpus == 16
    assert cfg.train.micro_batch_size == mbs
    assert cfg.model.recompute_modules == recompute_modules

    assert cfg.model.num_layers == 31
    assert cfg.model.num_moe_experts == 144
    assert cfg.model.moe_layer_freq == [0, 0, 0] + [1] * 28
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.virtual_pipeline_model_parallel_size == 4
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 36
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_layout == NEMODIAG_V0_PIPELINE_LAYOUT

    if precision == "bf16":
        assert cfg.mixed_precision.fp8 is None
        assert cfg.model.cuda_graph_impl == "transformer_engine"
    elif precision == "fp8_mx":
        assert cfg.mixed_precision.fp8_recipe == "mxfp8"
        assert cfg.model.cuda_graph_impl == "full_iteration"
        assert cfg.mixed_precision.fp8_dot_product_attention is True
        assert cfg.model.fp8_output_proj is True
    else:
        assert cfg.mixed_precision.fp4 == "e2m1"
        assert cfg.mixed_precision.fp4_recipe == "nvfp4"
        assert cfg.model.cuda_graph_impl == "none"


@pytest.mark.parametrize("recipe_fn,num_gpus,precision,unused_gbs,unused_mbs,unused_recompute", _RECIPE_CASES)
def test_nemodiag_v0_recipe_is_discoverable(
    recipe_fn,
    num_gpus: int,
    precision: str,
    unused_gbs: int,
    unused_mbs: int,
    unused_recompute: list[str],
):
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts" / "performance"
    sys.path.insert(0, str(scripts_dir))
    try:
        from utils.utils import get_perf_recipe_by_name

        discovered = get_perf_recipe_by_name(
            model_recipe_name="nemodiag_v0",
            task="pretrain",
            num_gpus=num_gpus,
            gpu="gb300",
            precision=precision,
            config_variant="perf72_e144",
        )
    finally:
        sys.path.remove(str(scripts_dir))

    assert isinstance(discovered, ConfigContainer)
    assert type(discovered.model) is MLAModelProvider
    assert discovered.train.global_batch_size == recipe_fn().train.global_batch_size


@pytest.mark.parametrize(
    "recipe_fn",
    [
        nemodiag_v0_pretrain_72gpu_gb300_bf16_perf72_e144_config,
        nemodiag_v0_pretrain_72gpu_gb300_fp8mx_perf72_e144_config,
        nemodiag_v0_pretrain_72gpu_gb300_nvfp4_perf72_e144_config,
    ],
)
def test_nemodiag_v0_recipe_validates_for_gb300(recipe_fn, monkeypatch: pytest.MonkeyPatch):
    import megatron.bridge.training.config as config_module
    import megatron.bridge.training.flex_dispatcher_backend as flex_backend_module

    monkeypatch.setattr(config_module, "get_world_size_safe", lambda: 72)
    monkeypatch.setattr(
        flex_backend_module.torch.cuda,
        "get_device_properties",
        lambda unused_device: SimpleNamespace(major=10, name="NVIDIA GB300"),
    )

    config_module.runtime_config_update(recipe_fn())
