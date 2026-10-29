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

"""
Unit tests for Qwen2.5 Omni Model implementation.

Uses toy-sized configs to keep tests fast (<5s each).
"""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniVisionEncoderConfig,
)

from megatron.bridge.models.qwen_omni.modeling_qwen25_omni.model import Qwen25OmniModel
from megatron.bridge.models.qwen_omni.modeling_qwen25_omni.transformer_config import Qwen25OmniTransformerConfig


HIDDEN_SIZE = 128


def _make_toy_thinker_config():
    """Build a tiny Qwen2.5-Omni thinker config for fast unit tests."""
    vision_config = Qwen2_5OmniVisionEncoderConfig(
        depth=2,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        window_size=112,
        out_hidden_size=HIDDEN_SIZE,
        fullatt_block_indexes=[1],
    )
    audio_config = Qwen2_5OmniAudioEncoderConfig(
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        d_model=64,
        num_mel_bins=32,
        output_dim=HIDDEN_SIZE,
        max_source_positions=100,
        n_window=10,
    )
    return Qwen2_5OmniThinkerConfig(
        vision_config=vision_config,
        audio_config=audio_config,
        text_config={
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "rope_scaling": {"mrope_section": [4, 6, 6], "type": "mrope"},
        },
    )


@pytest.fixture(scope="module")
def thinker_config():
    """Toy thinker config shared across all tests in this module."""
    return _make_toy_thinker_config()


@pytest.mark.timeout(30)
class TestQwen25OmniModel:
    """Test suite for Qwen2.5 Omni Model."""

    @classmethod
    def setup_class(cls):
        """Setup distributed process group once for all tests in this class."""
        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
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
        """Teardown distributed process group once after all tests in this class."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _setup_parallel_state(self, tp_size=1, pp_size=1, cp_size=1):
        """Setup Megatron parallel state."""
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=cp_size,
        )

        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        """Teardown Megatron parallel state after each test method."""
        parallel_state.destroy_model_parallel()

    @staticmethod
    def _make_language_config(thinker_config):
        """Create a Qwen25OmniTransformerConfig from the toy thinker config."""
        text_cfg = thinker_config.text_config
        vision_cfg = thinker_config.vision_config

        rope_scaling = getattr(text_cfg, "rope_scaling", None)
        mrope_section = (rope_scaling or {}).get("mrope_section", [4, 6, 6])

        return Qwen25OmniTransformerConfig(
            num_layers=2,
            hidden_size=text_cfg.hidden_size,
            num_attention_heads=text_cfg.num_attention_heads,
            num_query_groups=text_cfg.num_key_value_heads,
            kv_channels=text_cfg.hidden_size // text_cfg.num_attention_heads,
            ffn_hidden_size=text_cfg.intermediate_size,
            vocab_size=text_cfg.vocab_size,
            language_max_sequence_length=text_cfg.max_position_embeddings,
            patch_size=vision_cfg.patch_size,
            temporal_patch_size=vision_cfg.temporal_patch_size,
            in_channels=vision_cfg.in_channels,
            spatial_merge_size=vision_cfg.spatial_merge_size,
            rotary_base=getattr(text_cfg, "rope_theta", 1000000),
            rotary_percent=1.0,
            mrope_section=mrope_section,
            normalization="RMSNorm",
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=True,
            qk_layernorm=False,
            layernorm_epsilon=text_cfg.rms_norm_eps,
            bf16=False,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=text_cfg.attention_dropout,
        )

    @staticmethod
    def _make_layer_spec():
        """Create a GPT layer spec for the language model."""
        return get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            fp8=False,
        )

    def _build_model(self, thinker_config, pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
        """Helper to build a Qwen25OmniModel with the given flags."""
        self._setup_parallel_state(tp_size=1, pp_size=1)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        return Qwen25OmniModel(
            language_transformer_config=self._make_language_config(thinker_config),
            language_transformer_layer_spec=self._make_layer_spec(),
            thinker_transformer_config=thinker_config,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=add_decoder,
            pg_collection=pg_collection,
        )

    @pytest.mark.parametrize("freeze_all", [True, False])
    def test_model_freeze_api(self, freeze_all, thinker_config):
        """Test model freeze API."""
        model = self._build_model(thinker_config)

        if torch.cuda.is_available():
            model.to("cuda")

        model.freeze(
            freeze_language_model=freeze_all,
            freeze_vision_model=freeze_all,
            freeze_audio_model=freeze_all,
        )

        for name, param in model.named_parameters():
            assert param.requires_grad != freeze_all, f"{name=}"

    def test_shared_embedding_or_output_weight(self, thinker_config):
        """Test shared_embedding_or_output_weight method."""
        model = self._build_model(thinker_config, add_decoder=True)
        weight = model.shared_embedding_or_output_weight()
        assert weight is not None

        model = self._build_model(thinker_config, add_decoder=False)
        weight_no_decoder = model.shared_embedding_or_output_weight()
        assert weight_no_decoder is None

    def test_set_input_tensor(self, thinker_config):
        """Test set_input_tensor method."""
        model = self._build_model(thinker_config, pre_process=True)
        test_tensor = torch.randn(2, 4, HIDDEN_SIZE)

        model.set_input_tensor([test_tensor])
        assert model.thinker.encoder_hidden_state is not None

        model = self._build_model(thinker_config, pre_process=False)
        model.set_input_tensor([test_tensor])
