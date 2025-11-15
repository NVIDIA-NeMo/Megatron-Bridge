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

"""Unit tests for Qwen3VL TransformerBlock implementation."""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLTransformerBlock
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig


@pytest.fixture(scope="module")
def transformer_config():
    """Create a basic transformer config for testing."""
    return Qwen3VLTransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=64,
        ffn_hidden_size=512,
        vocab_size=1000,
        language_max_sequence_length=512,
        normalization="RMSNorm",
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=True,
        layernorm_epsilon=1e-6,
        bf16=False,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )


class TestQwen3VLTransformerBlock:
    """Test suite for Qwen3VL TransformerBlock."""

    @classmethod
    def setup_class(cls):
        """Setup distributed process group."""
        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29501"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(0)

            dist.init_process_group(
                backend="nccl" if device_count > 0 else "gloo",
                world_size=1,
                rank=0,
                timeout=datetime.timedelta(minutes=30),
            )

    @classmethod
    def teardown_class(cls):
        """Teardown distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _setup_parallel_state(self):
        """Setup Megatron parallel state."""
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        """Teardown Megatron parallel state."""
        parallel_state.destroy_model_parallel()

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("recompute_method", ["uniform", "block"])
    def test_checkpointed_forward(self, transformer_config, recompute_method):
        """Test checkpointed forward pass."""
        self._setup_parallel_state()

        # Update config for specific recompute method
        transformer_config.recompute_method = recompute_method
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            fp8=False,
        )

        block = Qwen3VLTransformerBlock(
            config=transformer_config,
            spec=layer_spec,
            pre_process=True,
            post_process=True,
        )
        block.train()

        if torch.cuda.is_available():
            block = block.cuda()

        # Create dummy inputs
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(seq_len, batch_size, transformer_config.hidden_size)

        if torch.cuda.is_available():
            hidden_states = hidden_states.cuda()

        hidden_states.requires_grad = True

        # Forward pass (should use checkpointed forward)
        output = block(
            hidden_states=hidden_states,
            attention_mask=None,
        )

        assert output is not None
        assert output.shape == hidden_states.shape

        # Backward to ensure checkpointing works
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
