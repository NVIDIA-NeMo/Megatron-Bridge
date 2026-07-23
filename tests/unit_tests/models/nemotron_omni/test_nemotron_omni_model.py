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

import copy
import datetime
import os
import socket
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.activations import squared_relu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from torch import nn

from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import (
    NemotronOmniModel,
    _pixel_shuffle_dynamic_resolution,
)
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT,
    NEMOTRON_OMNI_LLAVA_CONTRACT,
    NemotronOmniLlavaModelProvider,
    NemotronOmniModelProvider,
)


class _FakeLanguageModel(nn.Module):
    def embedding(self, input_ids, position_ids):
        del position_ids
        values = input_ids.transpose(0, 1).unsqueeze(-1).to(torch.float32)
        return values.repeat(1, 1, 3)

    def forward(self, *, decoder_input, **kwargs):
        del kwargs
        return decoder_input


class _BoundaryModel(NemotronOmniModel):
    """CPU-only shell that exercises the real expanded-sequence forward."""

    def __init__(self, image_features):
        nn.Module.__init__(self)
        self.pre_process = True
        self.image_token_index = 18
        self.context_parallel_lm = 1
        self.sequence_parallel_lm = False
        self.config = SimpleNamespace(mtp_num_layers=None)
        self.language_model = _FakeLanguageModel()
        self.image_features = image_features

    def _encode_images(self, images, imgs_sizes, vision_packed_seq_params, num_frames):
        del images, imgs_sizes, vision_packed_seq_params, num_frames
        return self.image_features


@dataclass
class _TinyOmniProvider(NemotronOmniModelProvider):
    """One-layer image model for the real RADIO/NemotronH Stage 1 smoke."""

    has_sound: bool = False
    language_model_type: str = "nemotron6-moe"
    hidden_size: int = 128
    ffn_hidden_size: int = 256
    num_attention_heads: int = 4
    num_query_groups: int = 2
    kv_channels: int = 32
    mamba_num_heads: int = 4
    mamba_head_dim: int = 32
    mamba_num_groups: int = 1
    mamba_state_dim: int = 16
    hybrid_layer_pattern: str = "M"
    vocab_size: int = 128
    seq_length: int = 32
    image_token_index: int = 18
    tokenizer_type: str = "nemotron6-moe"
    dynamic_resolution: bool = True
    temporal_patch_dim: int = 2
    separate_video_embedder: bool = True
    use_vision_backbone_fp8_arch: bool = False
    vision_proj_ffn_hidden_size: int = 256
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
    use_cpu_initialization: bool = True
    gradient_accumulation_fusion: bool = False
    nemotron_omni_contract: str = NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT

    def _build_vision_config(self, language_cfg):
        vision_cfg = copy.deepcopy(language_cfg)
        vision_cfg.sequence_parallel = False
        vision_cfg.context_parallel_size = 1
        vision_cfg.tp_comm_overlap = False
        vision_cfg.recompute_granularity = None
        vision_cfg.recompute_method = None
        vision_cfg.recompute_num_layers = None
        vision_cfg.mtp_num_layers = None
        vision_cfg.num_layers = 1
        vision_cfg.pipeline_model_parallel_size = 1
        vision_cfg.num_attention_heads = 4
        vision_cfg.add_bias_linear = True
        vision_cfg.add_qkv_bias = True
        vision_cfg.hidden_size = 128
        vision_cfg.ffn_hidden_size = 256
        vision_cfg.gated_linear_unit = False
        vision_cfg.kv_channels = 32
        vision_cfg.num_query_groups = 4
        vision_cfg.normalization = "LayerNorm"
        vision_cfg.qk_layernorm = False
        vision_cfg.layernorm_epsilon = 1e-6
        vision_cfg.class_token_len = 10
        return vision_cfg


def _find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


@pytest.fixture
def single_rank_model_parallel():
    original_env = {
        key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")
    }
    if not dist.is_initialized():
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=_find_free_port(),
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl",
            world_size=1,
            rank=0,
            timeout=datetime.timedelta(minutes=5),
        )
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(123)

    yield

    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_llava_provider_is_explicit_and_does_not_replace_canonical_provider():
    assert issubclass(NemotronOmniLlavaModelProvider, NemotronOmniModelProvider)
    assert NemotronOmniLlavaModelProvider.provide is not NemotronOmniModelProvider.provide


def test_canonical_provider_rejects_ambiguous_legacy_class_name():
    provider = _TinyOmniProvider(nemotron_omni_contract=None)

    with pytest.raises(RuntimeError, match="Refusing to guess which checkpoint layout"):
        provider.validate_model_contract()


def test_canonical_provider_rejects_explicit_llava_contract():
    provider = _TinyOmniProvider(nemotron_omni_contract=NEMOTRON_OMNI_LLAVA_CONTRACT)

    with pytest.raises(RuntimeError, match="requires 'expanded_sequence_v1'"):
        provider.validate_model_contract()


def test_llava_provider_requires_explicit_legacy_contract():
    provider = NemotronOmniLlavaModelProvider(nemotron_omni_contract=NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT)

    with pytest.raises(RuntimeError, match="requires 'llava_collapse_expand_v1'"):
        provider.validate_model_contract()


def test_vision_projection_matches_hf_and_vllm_activation():
    provider = _TinyOmniProvider()

    vision_projection_config = provider._build_vision_projection_config(provider)
    values = torch.tensor([-2.0, 0.0, 3.0])

    assert vision_projection_config.activation_func is squared_relu
    assert torch.equal(vision_projection_config.activation_func(values), torch.tensor([0.0, 0.0, 9.0]))


def test_radio_cpe_uses_square_interpolate_then_crop_by_default():
    provider = _TinyOmniProvider()

    assert provider.radio_interpolate_only_cpe is False


def test_llava_provider_preserves_existing_radio_cpe_default():
    provider = NemotronOmniLlavaModelProvider(nemotron_omni_contract=NEMOTRON_OMNI_LLAVA_CONTRACT)

    assert provider.radio_interpolate_only_cpe is True


def test_dynamic_resolution_pixel_shuffle_groups_spatial_2x2_blocks():
    features = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(1, 8, 2)

    shuffled = _pixel_shuffle_dynamic_resolution(features, height=2, width=4)

    assert torch.equal(
        shuffled,
        torch.tensor(
            [
                [
                    [0, 1, 2, 3, 8, 9, 10, 11],
                    [4, 5, 6, 7, 12, 13, 14, 15],
                ]
            ],
            dtype=torch.float32,
        ),
    )


def test_image_forward_replaces_expanded_placeholders_without_changing_length():
    image_features = torch.tensor([[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]])
    model = _BoundaryModel(image_features)
    input_ids = torch.tensor([[7, 18, 18, 9]])

    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        images=torch.ones(1),
    )

    assert output.shape == (4, 1, 3)
    assert torch.equal(output[0, 0], torch.tensor([7.0, 7.0, 7.0]))
    assert torch.equal(output[1, 0], image_features[0])
    assert torch.equal(output[2, 0], image_features[1])
    assert torch.equal(output[3, 0], torch.tensor([9.0, 9.0, 9.0]))


def test_text_only_control_preserves_language_embeddings():
    model = _BoundaryModel(torch.empty(0, 3))
    input_ids = torch.tensor([[7, 8, 9]])

    output = model(input_ids=input_ids)

    expected = torch.tensor([[[7.0, 7.0, 7.0]], [[8.0, 8.0, 8.0]], [[9.0, 9.0, 9.0]]])
    assert torch.equal(output, expected)


def test_media_alignment_mismatch_fails_loudly():
    language_embeddings = torch.zeros(4, 1, 3)
    input_ids = torch.tensor([[7, 18, 18, 9]])

    with pytest.raises(ValueError, match="2 valid placeholders for 1 projected features"):
        NemotronOmniModel._merge_projected_media(
            language_embeddings,
            input_ids,
            torch.ones(1, 3),
            media_token_id=18,
            attention_mask=None,
        )


def test_collapsed_llava_input_reports_contract_mismatch():
    language_embeddings = torch.zeros(3, 1, 3)
    input_ids = torch.tensor([[7, 18, 9]])

    with pytest.raises(ValueError, match="legacy LLaVAModel collapse/expand path"):
        NemotronOmniModel._merge_projected_media(
            language_embeddings,
            input_ids,
            torch.ones(2, 3),
            media_token_id=18,
            attention_mask=None,
        )


def test_padded_placeholder_is_not_treated_as_media():
    language_embeddings = torch.zeros(4, 1, 3)
    input_ids = torch.tensor([[7, 18, 9, 18]])
    attention_mask = torch.tensor([[True, True, True, False]])
    media_embedding = torch.tensor([[1.0, 2.0, 3.0]])

    output = NemotronOmniModel._merge_projected_media(
        language_embeddings,
        input_ids,
        media_embedding,
        media_token_id=18,
        attention_mask=attention_mask,
    )

    assert torch.equal(output[1, 0], media_embedding[0])
    assert torch.equal(output[3, 0], torch.zeros(3))


def test_media_merge_supports_backward_for_batch_size_one():
    language_embeddings = torch.randn(4, 1, 3, requires_grad=True)
    media_embeddings = torch.randn(2, 3, requires_grad=True)
    input_ids = torch.tensor([[7, 18, 18, 9]])

    output = NemotronOmniModel._merge_projected_media(
        language_embeddings,
        input_ids,
        media_embeddings,
        media_token_id=18,
        attention_mask=None,
    )
    output.sum().backward()

    assert language_embeddings.grad is not None
    assert media_embeddings.grad is not None


@pytest.mark.run_only_on("GPU")
def test_real_radio_image_forward_with_model_owned_cp1_packing(
    single_rank_model_parallel,
):
    del single_rank_model_parallel
    provider = _TinyOmniProvider()
    provider.finalize()
    model = provider.provide().cuda().eval()
    input_ids = torch.tensor([[7, 18, 9, 10]], device="cuda")
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    caller_packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            packed_seq_params=caller_packed_seq_params,
            pixel_values=torch.randn(1, 3, 32, 32, device="cuda"),
            imgs_sizes=torch.tensor([[32, 32]], dtype=torch.int32, device="cuda"),
            num_frames=torch.tensor([1], dtype=torch.int32, device="cuda"),
        )

    assert output.shape == (1, 4, 128)
    assert torch.isfinite(output).all()


@pytest.mark.run_only_on("GPU")
def test_packed_mamba_resets_state_between_samples(single_rank_model_parallel):
    del single_rank_model_parallel
    provider = _TinyOmniProvider()
    provider.finalize()
    model = provider.provide().cuda().eval()

    def forward(input_ids, attention_mask):
        caller_packed_seq_params = PackedSeqParams(qkv_format="thd")
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            packed_seq_params=caller_packed_seq_params,
        )

    input_ids = torch.tensor(
        [
            [7, 8, 9, 0],
            [11, 12, 0, 0],
        ],
        device="cuda",
    )
    attention_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ],
        device="cuda",
    )

    with torch.no_grad():
        packed_output = forward(input_ids, attention_mask)
        first_output = forward(input_ids[:1, :3], attention_mask[:1, :3])
        second_output = forward(input_ids[1:2, :2], attention_mask[1:2, :2])

    expected = torch.cat((first_output, second_output), dim=1)
    torch.testing.assert_close(packed_output, expected)
