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

import datetime
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)

from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniModel
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.rope import get_rope_index
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.transformer_config import (
    Qwen3OmniTransformerConfig,
)


HIDDEN_SIZE = 128
IMAGE_TOKEN_ID = 900
VIDEO_TOKEN_ID = 901
AUDIO_TOKEN_ID = 902
VISION_START_TOKEN_ID = 903
AUDIO_START_TOKEN_ID = 904


def _make_toy_thinker_config():
    return Qwen3OmniMoeThinkerConfig(
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": HIDDEN_SIZE,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        audio_config={
            "num_mel_bins": 8,
            "d_model": 32,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
            "encoder_layers": 2,
            "output_dim": HIDDEN_SIZE,
            "downsample_hidden_size": 16,
        },
        text_config={
            "num_hidden_layers": 2,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": 256,
            "moe_intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "rope_theta": 1000000.0,
            "rope_scaling": {"rope_type": "default", "mrope_section": [4, 6, 6]},
        },
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
    )


@pytest.fixture(scope="module")
def thinker_config():
    return _make_toy_thinker_config()


class TestQwen3OmniModel:
    _original_env: dict[str, str | None] = {}

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            cls._original_env = {
                key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")
            }

            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = cls._find_free_port()
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
        if dist.is_initialized():
            dist.destroy_process_group()
        for key, value in cls._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @staticmethod
    def _find_free_port() -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return str(sock.getsockname()[1])

    def _setup_parallel_state(self, tp_size=1, pp_size=1):
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
        )
        if torch.cuda.is_available():
            model_parallel_cuda_manual_seed(123)
        else:
            torch.manual_seed(123)

    def teardown_method(self):
        parallel_state.destroy_model_parallel()

    @staticmethod
    def _make_language_config():
        return Qwen3OmniTransformerConfig(
            num_layers=2,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=HIDDEN_SIZE // 4,
            ffn_hidden_size=256,
            moe_ffn_hidden_size=64,
            num_moe_experts=8,
            moe_router_topk=2,
            vocab_size=1000,
            language_max_sequence_length=128,
            normalization="RMSNorm",
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=False,
            qk_layernorm=True,
            layernorm_epsilon=1e-6,
            bf16=False,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            mrope_section=[4, 6, 6],
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            audio_token_id=AUDIO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            audio_start_token_id=AUDIO_START_TOKEN_ID,
            position_id_per_seconds=25,
            seconds_per_chunk=2,
        )

    @staticmethod
    def _make_layer_spec():
        if not torch.cuda.is_available():
            return get_gpt_layer_local_spec(
                num_experts=8,
                moe_grouped_gemm=True,
                qk_layernorm=True,
                normalization="RMSNorm",
            )
        return get_gpt_layer_with_transformer_engine_spec(
            num_experts=8,
            moe_grouped_gemm=True,
            qk_layernorm=True,
            fp8=False,
        )

    def _build_model(self, thinker_config):
        self._setup_parallel_state(tp_size=1, pp_size=1)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        return Qwen3OmniModel(
            language_transformer_config=self._make_language_config(),
            language_transformer_layer_spec=self._make_layer_spec(),
            thinker_transformer_config=thinker_config,
            parallel_output=True,
            pre_process=True,
            post_process=True,
            pg_collection=pg_collection,
        )

    def test_model_freeze_api(self, thinker_config):
        model = self._build_model(thinker_config)
        model.freeze(freeze_language_model=True)

        for name, param in model.named_parameters():
            if name.startswith("thinker.language_model"):
                assert param.requires_grad is False

    def test_set_input_tensor(self, thinker_config):
        model = self._build_model(thinker_config)
        test_tensor = torch.randn(2, 4, HIDDEN_SIZE)
        model.set_input_tensor([test_tensor])
        assert model.thinker.encoder_hidden_state is not None

    def test_text_only_forward(self, thinker_config):
        model = self._build_model(thinker_config)
        if torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"
        else:
            device = "cpu"

        input_ids = torch.randint(0, 1000, (1, 16), device=device)
        labels = torch.randint(0, 1000, (1, 16), device=device)
        output = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=None,
        )
        assert output is not None

    def test_image_forward(self, thinker_config):
        model = self._build_model(thinker_config)
        if torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"
        else:
            device = "cpu"

        input_ids = torch.tensor(
            [[VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12, 13, 14]],
            device=device,
        )
        labels = torch.randint(0, 1000, input_ids.shape, device=device)
        pixel_values = torch.randn(4, 3 * 1 * 2 * 2, device=device)
        image_grid_thw = torch.tensor([[1, 2, 2]], device=device)

        output = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        assert output is not None

    def test_audio_forward(self, thinker_config):
        model = self._build_model(thinker_config)
        if torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"
        else:
            device = "cpu"

        input_ids = torch.tensor(
            [[AUDIO_START_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 21, 22, 23]],
            device=device,
        )
        labels = torch.randint(0, 1000, input_ids.shape, device=device)
        input_features = torch.randn(1, 8, 10, device=device)
        feature_attention_mask = torch.ones(1, 10, dtype=torch.long, device=device)

        output = model(
            input_ids=input_ids,
            labels=labels,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        assert output is not None

    def test_audio_only_rope_index(self):
        input_ids = torch.tensor([[AUDIO_START_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 17, 18]])
        audio_seqlens = torch.tensor([16])

        position_ids, mrope_position_deltas = get_rope_index(
            spatial_merge_size=1,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            audio_token_id=AUDIO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            audio_start_token_id=AUDIO_START_TOKEN_ID,
            position_id_per_seconds=25,
            input_ids=input_ids,
            audio_seqlens=audio_seqlens,
        )

        assert position_ids.shape == (3, 1, input_ids.shape[1])
        assert mrope_position_deltas.shape == (1, 1)

    def test_video_rope_index_requires_second_per_grid(self):
        input_ids = torch.tensor([[VISION_START_TOKEN_ID, VIDEO_TOKEN_ID, VIDEO_TOKEN_ID, 17, 18]])
        video_grid_thw = torch.tensor([[1, 1, 2]])
        video_second_per_grid = torch.tensor([2.0])

        position_ids, mrope_position_deltas = get_rope_index(
            spatial_merge_size=1,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            audio_token_id=AUDIO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            audio_start_token_id=AUDIO_START_TOKEN_ID,
            position_id_per_seconds=25,
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            second_per_grids=video_second_per_grid,
        )

        assert position_ids.shape == (3, 1, input_ids.shape[1])
        assert mrope_position_deltas.shape == (1, 1)
