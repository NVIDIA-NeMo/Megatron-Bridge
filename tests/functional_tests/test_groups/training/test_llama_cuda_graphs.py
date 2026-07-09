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

"""Focused functional tests for Llama CUDA Graph training modes."""

import pytest

from megatron.bridge.recipes.llama.h100 import (
    llama32_1b_pretrain_1gpu_h100_bf16_config,
)
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_feature_test


LLAMA_CUDA_GRAPH_CASES = [
    # (config_func, name, config_overrides)
    (
        llama32_1b_pretrain_1gpu_h100_bf16_config,
        "llama32_1b_local_full_iteration_cuda_graph",
        {
            "model": {
                "num_layers": 2,
                "cuda_graph_impl": "local",
                "cuda_graph_scope": ["full_iteration"],
                "use_te_rng_tracker": True,
            },
            "rerun_state_machine": {"check_for_nan_in_loss": False},
            "ddp": {"check_for_nan_in_grad": False},
        },
    ),
    (
        llama32_1b_pretrain_1gpu_h100_bf16_config,
        "llama32_1b_transformer_engine_attention_cuda_graph",
        {
            "model": {
                "num_layers": 2,
                "cuda_graph_impl": "transformer_engine",
                "cuda_graph_scope": ["attn"],
                "use_te_rng_tracker": True,
            },
            "rerun_state_machine": {"check_for_nan_in_loss": False},
            "ddp": {"check_for_nan_in_grad": False},
        },
    ),
]


class TestLlamaCudaGraphs:
    """Exercise CUDA Graph modes that are not standalone performance recipes."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,test_name,config_overrides", LLAMA_CUDA_GRAPH_CASES)
    def test_llama_cuda_graph_training(self, config_func, test_name, config_overrides):
        """Run short training with each explicitly selected CUDA Graph implementation."""
        run_pretrain_feature_test(
            config_func,
            test_name,
            config_overrides=config_overrides,
        )
