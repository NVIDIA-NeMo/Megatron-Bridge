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


@pytest.fixture(scope="module", autouse=True)
def _distributed_env():
    original_env = {
        key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")
    }

    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = _find_free_port()
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(0)

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=1,
            rank=0,
            timeout=datetime.timedelta(minutes=30),
        )

    yield

    if dist.is_initialized():
        dist.destroy_process_group()
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _setup_parallel_state():
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
    )
    if torch.cuda.is_available():
        model_parallel_cuda_manual_seed(123)
    else:
        torch.manual_seed(123)


def _build_model():
    _setup_parallel_state()
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    return Qwen3OmniModel(
        language_transformer_config=_make_language_config(),
        language_transformer_layer_spec=_make_layer_spec(),
        thinker_transformer_config=_make_toy_thinker_config(),
        parallel_output=True,
        pre_process=True,
        post_process=True,
        pg_collection=pg_collection,
    )


def test_qwen3_omni_toy_training_step():
    model = _build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    input_ids = torch.tensor(
        [[
            VISION_START_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            AUDIO_START_TOKEN_ID,
            AUDIO_TOKEN_ID,
            AUDIO_TOKEN_ID,
            31,
            32,
            33,
        ]],
        device=device,
    )
    labels = torch.randint(0, 1000, input_ids.shape, device=device)
    pixel_values = torch.randn(4, 3 * 1 * 2 * 2, device=device)
    image_grid_thw = torch.tensor([[1, 2, 2]], device=device)
    input_features = torch.randn(1, 8, 10, device=device)
    feature_attention_mask = torch.ones(1, 10, dtype=torch.long, device=device)

    optimizer.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids,
        labels=labels,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
    )

    loss = output.float().mean() if output.ndim > 0 else output.float()
    assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
    loss.backward()

    grads = [
        param.grad for name, param in model.named_parameters() if param.requires_grad and param.grad is not None
    ]
    assert grads, "Expected at least one trainable parameter to receive gradients"
    assert any(torch.isfinite(grad).all() for grad in grads), "Expected finite gradients after backward"

    optimizer.step()

    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
