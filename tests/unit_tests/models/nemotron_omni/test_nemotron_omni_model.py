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
    _pad_patch_grid_to_even,
    _pixel_shuffle_dynamic_resolution,
    _project_multimodal_embeddings,
)
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni_llava import NemotronOmniLlavaModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT,
    NEMOTRON_OMNI_LLAVA_CONTRACT,
    NemotronOmniLlavaModelProvider,
    NemotronOmniModelProvider,
)
from megatron.bridge.models.nemotron_vl.modeling_nemotron_vl import NemotronVLModel


class _FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def embedding(self, input_ids, position_ids):
        del position_ids
        values = input_ids.transpose(0, 1).unsqueeze(-1).to(torch.float32)
        return values.repeat(1, 1, 3)

    def forward(self, *, decoder_input, **kwargs):
        self.last_kwargs = kwargs
        return decoder_input


class _BoundaryModel(NemotronOmniModel):
    """CPU-only shell that exercises the real expanded-sequence forward."""

    def __init__(self, image_features, sound_features=None):
        nn.Module.__init__(self)
        self.pre_process = True
        self.image_token_index = 18
        self.sound_token_index = 19
        self.context_parallel_lm = 1
        self.sequence_parallel_lm = False
        self.config = SimpleNamespace(mtp_num_layers=None)
        self.language_model = _FakeLanguageModel()
        self.image_features = image_features
        self.sound_features = torch.empty(0, 3) if sound_features is None else sound_features

    def _encode_images(self, images, imgs_sizes, vision_packed_seq_params, num_frames):
        del images, imgs_sizes, vision_packed_seq_params, num_frames
        return self.image_features

    def _encode_sound(self, sound_clips, sound_length):
        del sound_clips, sound_length
        return self.sound_features


class _FakeSoundModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(sound_pad_to_clip_duration=False)

    def forward(self, sound_clips, sound_length):
        del sound_clips, sound_length
        embeddings = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        return embeddings, torch.tensor([2, 1])


class _SoundEncoderBoundaryModel(NemotronOmniModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.sound_model = _FakeSoundModel()
        self.sound_projection = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.sound_projection.weight.copy_(torch.eye(2, dtype=torch.bfloat16))


class _RecordingProjection(nn.Module):
    def __init__(self, *, fp8: bool):
        super().__init__()
        self.config = SimpleNamespace(fp8="hybrid" if fp8 else None, fp8_recipe="tensorwise")
        self.input_shape = None

    def forward(self, hidden_states):
        self.input_shape = hidden_states.shape
        return hidden_states * 2


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


def test_canonical_model_advertises_collator_owned_packing():
    assert NemotronOmniModel.model_owns_packing is False
    assert NemotronOmniModel.model_owns_mtp_loss_mask_packing is False
    assert NemotronOmniModel.model_slices_context_parallel_inputs is True


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


def test_vision_backbone_fp8_policy_does_not_disable_language_or_projection_fp8():
    provider = NemotronOmniModelProvider(
        fp8="hybrid",
        fp8_param=True,
        use_vision_backbone_fp8_arch=False,
    )

    vision_config = provider._build_vision_config(provider)
    vision_projection_config = provider._build_vision_projection_config(provider)

    assert vision_config.fp8 is None
    assert vision_config.fp8_param is False
    assert provider.fp8 == "hybrid"
    assert provider.fp8_param is True
    assert vision_projection_config.fp8 == "hybrid"
    assert vision_projection_config.fp8_param is True


def test_multimodal_projection_pads_only_temporary_fp8_rows():
    projection = _RecordingProjection(fp8=True)
    embeddings = torch.arange(7 * 2 * 3, dtype=torch.float32).reshape(7, 2, 3).requires_grad_()

    projected = _project_multimodal_embeddings(projection, embeddings)
    projected.sum().backward()

    assert projection.input_shape == (16, 1, 3)
    assert projected.shape == embeddings.shape
    assert torch.equal(projected, embeddings * 2)
    assert torch.equal(embeddings.grad, torch.full_like(embeddings, 2))


def test_multimodal_projection_does_not_pad_bf16_rows():
    projection = _RecordingProjection(fp8=False)
    embeddings = torch.arange(7 * 2 * 3, dtype=torch.float32).reshape(7, 2, 3)

    projected = _project_multimodal_embeddings(projection, embeddings)

    assert projection.input_shape == (14, 1, 3)
    assert torch.equal(projected, embeddings * 2)


def test_radio_cpe_uses_square_interpolate_then_crop_by_default():
    provider = _TinyOmniProvider()

    assert provider.radio_interpolate_only_cpe is False


def test_llava_provider_preserves_existing_radio_cpe_default():
    provider = NemotronOmniLlavaModelProvider(nemotron_omni_contract=NEMOTRON_OMNI_LLAVA_CONTRACT)

    assert provider.radio_interpolate_only_cpe is True


def test_llava_model_emits_deprecation_notice(monkeypatch):
    monkeypatch.setattr(NemotronVLModel, "__init__", lambda *_args, **_kwargs: None)

    with pytest.warns(FutureWarning, match="NemotronOmniLlavaModel is deprecated"):
        NemotronOmniLlavaModel()


def test_llava_provider_emits_deprecation_notice(monkeypatch):
    provider = NemotronOmniLlavaModelProvider(nemotron_omni_contract=NEMOTRON_OMNI_LLAVA_CONTRACT)
    legacy_model = object()
    monkeypatch.setattr(provider, "_provide_llava", lambda **_: legacy_model)

    with pytest.warns(FutureWarning, match="NemotronOmniLlavaModelProvider is deprecated"):
        assert provider.provide() is legacy_model


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


def test_even_patch_grid_is_returned_unchanged():
    features = torch.randn(1, 32 * 32, 8)

    padded, height, width = _pad_patch_grid_to_even(features, height=32, width=32)

    assert padded is features
    assert (height, width) == (32, 32)


def test_odd_patch_grid_is_zero_padded_so_pixel_shuffle_accepts_it():
    # The CP vision split injects 1x1-patch placeholder images to keep every
    # rank non-empty; pixel shuffle rejects odd grids outright.
    features = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)

    padded, height, width = _pad_patch_grid_to_even(features, height=1, width=1)

    assert (height, width) == (2, 2)
    assert torch.equal(padded[0, 0], features[0, 0])
    assert padded[0, 1:].abs().sum() == 0
    assert _pixel_shuffle_dynamic_resolution(padded, height=height, width=width).shape == (1, 1, 32)


def test_odd_patch_grid_padding_keeps_real_patches_differentiable():
    features = torch.randn(1, 3 * 5, 8, requires_grad=True)

    padded, height, width = _pad_patch_grid_to_even(features, height=3, width=5)
    _pixel_shuffle_dynamic_resolution(padded, height=height, width=width).sum().backward()

    assert (height, width) == (4, 6)
    assert torch.count_nonzero(features.grad) == features.numel()


def test_vision_dp_over_cp_rejects_process_group_mismatch():
    # The MCore splitter resolves the CP group from the global parallel state,
    # so a pg_collection that disagrees with the config would shard against the
    # wrong ranks instead of failing.
    model = NemotronOmniModel.__new__(NemotronOmniModel)
    nn.Module.__init__(model)
    model.context_parallel_lm = 2
    model.patch_dim = 16
    model.config = SimpleNamespace(fp8_recipe=None)
    model.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: 1))

    with pytest.raises(ValueError, match="config=2, group=1"):
        model._split_images_across_context_parallel_ranks(
            torch.zeros(1, 3, 32, 32),
            torch.tensor([[32, 32]], dtype=torch.int32),
            None,
            num_frames=None,
            temporal_patch_size=1,
        )


class _FakeDynamicVisionModel(nn.Module):
    """Returns one feature row per patch implied by the sizes it is handed."""

    temporal_patch_dim = 1
    add_class_token = False

    def __init__(self, hidden_size: int, patch_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))
        self.hidden_size = hidden_size
        self.patch_dim = patch_dim
        self.seen = {}

    def forward(self, images, *, imgs_sizes, packed_seq_params, num_frames):
        self.seen = {"images": images, "imgs_sizes": imgs_sizes, "num_frames": num_frames}
        patches = sum(
            (int(height) // self.patch_dim) * (int(width) // self.patch_dim) for height, width in imgs_sizes.tolist()
        )
        return torch.arange(patches * self.hidden_size, dtype=torch.float32).reshape(1, patches, self.hidden_size)


def test_sharded_encode_projects_locally_then_gathers_the_global_features():
    # The distributed equivalence test covers the numerics, but it runs in a
    # torchrun subprocess. This exercises the same Bridge-side wiring in-process:
    # the split feeds the tower, the placeholder grid gets padded, and the
    # projector runs before the gather rather than after it.
    from megatron.bridge.models.nemotron_omni import modeling_nemotron_omni as modeling

    hidden_size = 8
    patch_dim = 16
    model = NemotronOmniModel.__new__(NemotronOmniModel)
    nn.Module.__init__(model)
    model.patch_dim = patch_dim
    model.context_parallel_lm = 2
    model.vision_dp_over_cp = True
    model.config = SimpleNamespace(fp8_recipe=None)
    model.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: 2))
    model.vision_model = _FakeDynamicVisionModel(hidden_size, patch_dim)
    model.vision_projection = nn.Linear(hidden_size * 4, 5, bias=False)

    # This rank keeps a single 1x1-patch placeholder, the shape MCore produces
    # when a microbatch has fewer images than CP ranks.
    local_num_frames = torch.tensor([1], dtype=torch.int32)
    calls = {}

    def fake_split(images, imgs_sizes, packed_seq_params, **kwargs):
        calls["split"] = kwargs
        local_sizes = torch.tensor([[patch_dim, patch_dim]], dtype=torch.int32)
        return images[:, :1, :], local_sizes, packed_seq_params, False, 1, local_num_frames

    def fake_gather(projected, num_padded_ranks):
        calls["gather"] = {"width": projected.shape[-1], "num_padded_ranks": num_padded_ranks}
        return projected

    original_split = modeling.split_to_context_parallel_ranks_dynamic_res
    original_gather = modeling.gather_from_context_parallel_ranks_dynamic_res
    modeling.split_to_context_parallel_ranks_dynamic_res = fake_split
    modeling.gather_from_context_parallel_ranks_dynamic_res = fake_gather
    try:
        encoded = model._encode_images(
            torch.randn(1, 3, 32, 32),
            torch.tensor([[32, 32]], dtype=torch.int32),
            None,
            None,
        )
    finally:
        modeling.split_to_context_parallel_ranks_dynamic_res = original_split
        modeling.gather_from_context_parallel_ranks_dynamic_res = original_gather

    assert calls["split"]["patch_dim"] == patch_dim
    # The tower must see this rank's shard, including the frame counts the
    # splitter recomputed for it rather than the microbatch-wide ones.
    assert model.vision_model.seen["images"].shape[1] == 1
    assert model.vision_model.seen["num_frames"] is local_num_frames
    # A 1x1 placeholder grid only survives pixel shuffle once padded to 2x2.
    assert encoded.shape == (1, 5)
    # Width 5 is the projector's output, so the gather ran on projected
    # features; gathering first would have handed it the wider encoder output.
    assert calls["gather"] == {"width": 5, "num_padded_ranks": 1}


@pytest.mark.gpu
def test_vision_dp_over_cp_is_disabled_when_cp_is_one(single_rank_model_parallel):
    # Sharding images over a one-rank CP group is a no-op wrapped in two
    # collectives, so the model must ignore the request rather than pay for it.
    provider = _TinyOmniProvider(vision_dp_over_cp=True)
    provider.finalize()

    model = provider.provide()

    assert provider.vision_dp_over_cp is True
    assert model.vision_dp_over_cp is False


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


def test_image_forward_does_not_use_mcore_causal_mask_as_token_validity():
    image_features = torch.tensor([[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]])
    model = _BoundaryModel(image_features)
    input_ids = torch.tensor([[7, 18, 18, 9]])
    causal_attention_mask = torch.triu(torch.ones(1, 1, 4, 4, dtype=torch.bool), diagonal=1)

    output = model(
        input_ids=input_ids,
        attention_mask=causal_attention_mask,
        images=torch.ones(1),
    )

    assert output.shape == (4, 1, 3)
    assert torch.equal(output[1, 0], image_features[0])
    assert torch.equal(output[2, 0], image_features[1])
    assert torch.equal(model.language_model.last_kwargs["attention_mask"], causal_attention_mask)


def test_audio_forward_replaces_expanded_placeholders_without_changing_length():
    sound_features = torch.tensor([[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]])
    model = _BoundaryModel(torch.empty(0, 3), sound_features)
    input_ids = torch.tensor([[7, 19, 19, 9]])

    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        sound_clips=torch.ones(1, 8, 2),
        sound_length=torch.tensor([8]),
    )

    assert output.shape == (4, 1, 3)
    assert torch.equal(output[0, 0], torch.tensor([7.0, 7.0, 7.0]))
    assert torch.equal(output[1, 0], sound_features[0])
    assert torch.equal(output[2, 0], sound_features[1])
    assert torch.equal(output[3, 0], torch.tensor([9.0, 9.0, 9.0]))


def test_sound_encoder_drops_padded_rows_and_preserves_sample_order():
    model = _SoundEncoderBoundaryModel()

    encoded = model._encode_sound(
        torch.ones(2, 8, 2),
        torch.tensor([8, 4]),
    )
    assert torch.equal(
        encoded,
        torch.tensor(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [6.0, 7.0],
            ],
            dtype=torch.bfloat16,
        ),
    )


def test_bridge_sound_encoder_reconstructs_parakeet_feature_mask_from_valid_length():
    from megatron.bridge.models.nemotron_omni.nemotron_omni_sound import BridgeSoundEncoder

    class _RecordingEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_mask = None

        def forward(self, *, input_features, attention_mask):
            self.attention_mask = attention_mask
            return SimpleNamespace(last_hidden_state=input_features)

        def _get_subsampling_output_length(self, lengths):
            for _ in range(3):
                lengths = (lengths + 1) // 2
            return lengths

    config = SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        num_mel_bins=8,
        subsampling_factor=8,
        conv_kernel_size=9,
        use_bias=False,
    )
    model = BridgeSoundEncoder(config)
    recording_encoder = _RecordingEncoder()
    model.encoder = recording_encoder

    _, embedding_lengths = model(torch.ones(1, 9, config.num_mel_bins), torch.tensor([8]))

    assert recording_encoder.attention_mask.tolist() == [[True] * 8 + [False]]
    assert embedding_lengths.tolist() == [1]


@pytest.mark.parametrize(
    ("sound_length", "match"),
    [
        (torch.tensor([0]), "0 < length"),
        (torch.tensor([10]), "physical sound_clips width"),
        (torch.tensor([8.0]), "integral dtype"),
    ],
)
def test_bridge_sound_encoder_rejects_invalid_valid_lengths(sound_length, match):
    from megatron.bridge.models.nemotron_omni.nemotron_omni_sound import BridgeSoundEncoder

    config = SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        num_mel_bins=8,
        subsampling_factor=8,
        conv_kernel_size=9,
        use_bias=False,
    )
    model = BridgeSoundEncoder(config)

    with pytest.raises(ValueError, match=match):
        model(torch.ones(1, 9, config.num_mel_bins), sound_length)


@pytest.mark.run_only_on("GPU")
def test_bridge_sound_encoder_validates_cuda_lengths_asynchronously(monkeypatch):
    from megatron.bridge.models.nemotron_omni.nemotron_omni_sound import BridgeSoundEncoder

    class _RecordingEncoder(nn.Module):
        def forward(self, *, input_features, attention_mask):
            del attention_mask
            return SimpleNamespace(last_hidden_state=input_features)

        def _get_subsampling_output_length(self, lengths):
            return lengths

    config = SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        num_mel_bins=8,
        subsampling_factor=8,
        conv_kernel_size=9,
        use_bias=False,
    )
    model = BridgeSoundEncoder(config)
    model.encoder = _RecordingEncoder()
    assertions = []
    original_assert_async = torch._assert_async

    def _record_assertion(condition, message):
        assertions.append(condition)
        original_assert_async(condition, message)

    monkeypatch.setattr(torch, "_assert_async", _record_assertion)

    model(
        torch.ones(1, 9, config.num_mel_bins, device="cuda"),
        torch.tensor([8], device="cuda"),
    )

    assert len(assertions) == 1
    assert assertions[0].is_cuda


def test_real_parakeet_sound_encoder_matches_subsampled_placeholder_count():
    from megatron.bridge.models.nemotron_omni.nemotron_omni_sound import BridgeSoundEncoder

    config = SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        num_mel_bins=8,
        subsampling_factor=8,
        conv_kernel_size=9,
        use_bias=False,
        sound_pad_to_clip_duration=False,
    )
    model = _SoundEncoderBoundaryModel()
    model.sound_model = BridgeSoundEncoder(config)
    model.sound_projection = nn.Linear(config.hidden_size, 3, bias=False)
    sound_length = torch.tensor([64, 40])

    encoded = model._encode_sound(torch.randn(2, 64, config.num_mel_bins), sound_length)

    expected_lengths = model.sound_model.encoder._get_subsampling_output_length(sound_length)
    assert encoded.shape == (int(expected_lengths.sum().item()), 3)
    assert torch.isfinite(encoded).all()


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


def test_text_containing_the_placeholder_trains_with_a_caller_mask():
    """A row with no media may still spell the placeholder in ordinary prose.

    The media token is a normal vocabulary entry, so text can contain it -- a
    problem statement quoting ``<image>``, for instance. Derived masks answer
    "is this a real token", which cannot distinguish that from an anchor, so
    without a caller-supplied mask the merge demands a feature that was never
    meant to exist.
    """
    model = _BoundaryModel(torch.empty(0, 3))
    input_ids = torch.tensor([[7, 18, 9]])

    with pytest.raises(ValueError, match="1 valid placeholders for 0 projected features"):
        model(input_ids=input_ids)

    output = model(
        input_ids=input_ids,
        media_token_validity_mask=torch.tensor([[True, False, True]]),
    )

    # The spared placeholder keeps the language embedding the forward gave it.
    # That embedding is of token id 0, because the forward masks media tokens
    # out of the text before embedding regardless of the validity mask.
    assert torch.equal(output, torch.tensor([[[7.0] * 3], [[0.0] * 3], [[9.0] * 3]]))


def test_caller_mask_takes_precedence_over_the_derived_one():
    """An explicit mask must win, or the caller cannot express this at all."""
    model = _BoundaryModel(torch.empty(0, 3))
    input_ids = torch.tensor([[7, 18, 9]])

    # A padding mask marks every position valid, so on its own it would treat
    # the placeholder as an anchor and raise.
    output = model(
        input_ids=input_ids,
        padding_mask=torch.zeros(1, 3, dtype=torch.bool),
        media_token_validity_mask=torch.tensor([[True, False, True]]),
    )

    assert output.shape == (3, 1, 3)


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


def test_collator_owned_packing_is_preserved_while_model_applies_cp_shard(monkeypatch):
    cp_index = torch.tensor([0, 1, 4, 7], dtype=torch.long)
    seen = {}

    def fake_get_packed_seq_cp_partition_indices(packed_seq_params, **kwargs):
        seen["packed_seq_params"] = packed_seq_params
        seen.update(kwargs)
        return cp_index

    monkeypatch.setattr(
        "megatron.bridge.models.nemotron_omni.modeling_nemotron_omni.get_packed_seq_cp_partition_indices",
        fake_get_packed_seq_cp_partition_indices,
    )

    image_features = torch.tensor([[101.0, 102.0, 103.0]])
    model = _BoundaryModel(image_features)
    model.context_parallel_lm = 2
    model.pg_collection = SimpleNamespace(
        cp=SimpleNamespace(size=lambda: 2, rank=lambda: 0),
        tp=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
    )
    input_ids = torch.tensor([[7, 18, 9, 0, 11, 12, 13, 0]])
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    labels = input_ids.clone()
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    padding_mask = torch.tensor([[False, False, False, True, False, False, False, True]])
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=4,
        max_seqlen_kv=4,
        total_tokens=8,
    )

    output, local_loss_mask = model(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        loss_mask=loss_mask,
        padding_mask=padding_mask,
        packed_seq_params=packed_seq_params,
        images=torch.ones(1),
    )

    assert seen["packed_seq_params"] is packed_seq_params
    assert seen["total_tokens"] == 8
    assert seen["cp_size"] == 2
    assert seen["cp_rank"] == 0
    assert output.shape == (4, 1, 3)
    assert torch.equal(output[0, 0], torch.tensor([7.0, 7.0, 7.0]))
    assert torch.equal(output[1, 0], image_features[0])
    assert torch.equal(output[2, 0], torch.tensor([11.0, 11.0, 11.0]))
    assert torch.equal(output[3, 0], torch.zeros(3))
    assert torch.equal(local_loss_mask, loss_mask.index_select(1, cp_index))
    assert model.language_model.last_kwargs["packed_seq_params"] is packed_seq_params
    assert torch.equal(model.language_model.last_kwargs["labels"], labels.index_select(1, cp_index))
    assert "padding_mask" not in model.language_model.last_kwargs
    assert model.language_model.last_kwargs["attention_mask"] is None


def test_dense_expanded_sequence_is_cp_sharded_after_media_insertion():
    image_features = torch.tensor([[101.0, 102.0, 103.0]])
    model = _BoundaryModel(image_features)
    model.context_parallel_lm = 2
    model.pg_collection = SimpleNamespace(
        cp=SimpleNamespace(size=lambda: 2, rank=lambda: 0),
        tp=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
    )
    input_ids = torch.tensor([[7, 18, 9, 10, 11, 12, 13, 14]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    labels = input_ids.clone()
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    expected_index = torch.tensor([0, 1, 6, 7])

    output, local_loss_mask = model(
        input_ids=input_ids,
        position_ids=torch.arange(8).unsqueeze(0),
        attention_mask=attention_mask,
        labels=labels,
        loss_mask=loss_mask,
        images=torch.ones(1),
    )

    assert output.shape == (4, 1, 3)
    assert torch.equal(output[0, 0], torch.tensor([7.0, 7.0, 7.0]))
    assert torch.equal(output[1, 0], image_features[0])
    assert torch.equal(output[2, 0], torch.tensor([13.0, 13.0, 13.0]))
    assert torch.equal(output[3, 0], torch.tensor([14.0, 14.0, 14.0]))
    assert torch.equal(local_loss_mask, loss_mask.index_select(1, expected_index))
    assert torch.equal(model.language_model.last_kwargs["labels"], labels.index_select(1, expected_index))
    assert torch.equal(
        model.language_model.last_kwargs["attention_mask"],
        attention_mask.index_select(1, expected_index),
    )


@pytest.mark.run_only_on("GPU")
def test_real_radio_image_forward_with_collator_owned_cp1_packing(
    single_rank_model_parallel,
):
    del single_rank_model_parallel
    provider = _TinyOmniProvider()
    provider.finalize()
    model = provider.provide().cuda().eval()
    input_ids = torch.tensor([[7, 18, 9, 10]], device="cuda")
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    caller_packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=4,
        max_seqlen_kv=4,
        total_tokens=4,
    )

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            padding_mask=torch.zeros_like(input_ids, dtype=torch.bool),
            packed_seq_params=caller_packed_seq_params,
            pixel_values=torch.randn(1, 3, 32, 32, device="cuda"),
            imgs_sizes=torch.tensor([[32, 32]], dtype=torch.int32, device="cuda"),
            num_frames=torch.tensor([1], dtype=torch.int32, device="cuda"),
        )

    assert output.shape == (1, 4, 128)
    assert torch.isfinite(output).all()


@pytest.mark.run_only_on("GPU")
def test_real_packed_multimodal_optimizer_step(single_rank_model_parallel):
    del single_rank_model_parallel
    provider = _TinyOmniProvider(temporal_patch_dim=1, separate_video_embedder=False)
    provider.finalize()
    model = provider.provide().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.tensor([[7, 18, 9, 0, 11, 18, 12, 0]], device="cuda")
    labels = torch.tensor([[18, 9, -100, -100, 18, 12, -100, -100]], device="cuda")
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]], device="cuda")
    padding_mask = torch.tensor([[False, False, False, True, False, False, False, True]], device="cuda")
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
    cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=4,
        max_seqlen_kv=4,
        total_tokens=8,
    )

    optimizer.zero_grad(set_to_none=True)
    per_token_loss = model(
        input_ids=input_ids,
        labels=labels,
        loss_mask=loss_mask,
        padding_mask=padding_mask,
        packed_seq_params=packed_seq_params,
        pixel_values=torch.randn(2, 3, 32, 32, device="cuda"),
        imgs_sizes=torch.tensor([[32, 32], [32, 32]], dtype=torch.int32, device="cuda"),
    )
    loss = (per_token_loss.float() * loss_mask).sum() / loss_mask.sum()
    loss.backward()

    updated_parameter = next(
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
    )
    parameter_before_step = updated_parameter.detach().clone()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(updated_parameter, parameter_before_step)


@pytest.mark.run_only_on("GPU")
def test_real_radio_multiframe_video_forward(single_rank_model_parallel):
    del single_rank_model_parallel
    provider = _TinyOmniProvider()
    provider.finalize()
    model = provider.provide().cuda().eval()
    input_ids = torch.tensor([[7, 18, 9, 10]], device="cuda")

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            pixel_values=torch.randn(2, 3, 32, 32, device="cuda"),
            imgs_sizes=torch.tensor([[32, 32], [32, 32]], dtype=torch.int32, device="cuda"),
            num_frames=torch.tensor([2], dtype=torch.int32, device="cuda"),
        )

    assert output.shape == (1, 4, 128)
    assert torch.isfinite(output).all()


@pytest.mark.run_only_on("GPU")
def test_packed_mamba_resets_state_between_samples(single_rank_model_parallel):
    del single_rank_model_parallel
    provider = _TinyOmniProvider()
    provider.finalize()
    model = provider.provide().cuda().eval()

    def forward(input_ids, padding_mask, cu_seqlens, cu_seqlens_padded):
        caller_packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=4,
            max_seqlen_kv=4,
            total_tokens=input_ids.size(1),
        )
        return model(
            input_ids=input_ids,
            padding_mask=padding_mask,
            packed_seq_params=caller_packed_seq_params,
        )

    input_ids = torch.tensor([[7, 8, 9, 0, 11, 12, 0, 0]], device="cuda")
    padding_mask = torch.tensor([[False, False, False, True, False, False, True, True]], device="cuda")
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        packed_output = forward(input_ids, padding_mask, cu_seqlens, cu_seqlens_padded)
        first_output = forward(
            input_ids[:, :4],
            padding_mask[:, :4],
            torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        )
        second_output = forward(
            input_ids[:, 4:],
            padding_mask[:, 4:],
            torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
            torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        )

    expected = torch.cat((first_output, second_output), dim=1)
    torch.testing.assert_close(packed_output, expected)
