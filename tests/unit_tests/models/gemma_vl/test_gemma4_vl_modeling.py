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

"""Unit tests for Gemma4VLModel helpers (no GPU / Megatron distributed required)."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import (
    Gemma4VLModel,
    _keep_hf_precision_buffers_in_fp32,
    _SimpleAudioEmbedder,
    _SimpleVisionEmbedder,
)


IMAGE_TOKEN_ID = 258_880


# ---------------------------------------------------------------------------
# Helpers to build a minimal Gemma4VLModel without actual GPU/Megatron init
# ---------------------------------------------------------------------------


def _make_model(image_token_id=IMAGE_TOKEN_ID):
    """Build a Gemma4VLModel with all heavy dependencies mocked out."""
    config = Mock()
    config.image_token_id = image_token_id
    config.vision_config = Mock()
    config.text_config = Mock()
    config.share_embeddings_and_output_weights = True
    config.sequence_parallel = False
    config._pg_collection = None

    # Patch out __init__ dependencies that require distributed env
    with (
        patch("megatron.bridge.models.gemma_vl.modeling_gemma4_vl.AutoModel") as mock_am,
        patch.object(Gemma4VLModel, "_init_embed_vision"),
        patch.object(Gemma4VLModel, "config", config, create=True),
    ):
        mock_am.from_config.return_value = Mock()
        # Bypass MegatronModule.__init__ which needs distributed state
        model = object.__new__(Gemma4VLModel)
        model.config = config
        model.pre_process = True
        model.post_process = True
        model.vp_stage = None
    return model


# ---------------------------------------------------------------------------
# _compute_attention_mask
# ---------------------------------------------------------------------------


class TestComputeAttentionMask:
    """Test Gemma4VLModel._compute_attention_mask (pure tensor logic, CPU-only)."""

    IMAGE_TOKEN = IMAGE_TOKEN_ID
    TEXT_TOKEN = 1  # arbitrary non-image token

    def _make_ids(self, pattern: list[int]) -> torch.Tensor:
        """Build [1, seq_len] input_ids from a flat list."""
        return torch.tensor([pattern], dtype=torch.long)

    def test_pure_text_returns_causal_mask(self):
        """No image tokens: mask should be causal (lower-triangular)."""
        model = _make_model()
        seq = [self.TEXT_TOKEN] * 6
        input_ids = self._make_ids(seq)
        mask = model._compute_attention_mask(input_ids)

        assert mask is not None
        assert mask.shape == (1, 1, 6, 6)
        # causal: positions (i,j) are masked (True) where j > i
        # The returned mask is True where attention is BLOCKED
        for i in range(6):
            for j in range(6):
                expected_blocked = j > i
                assert mask[0, 0, i, j].item() == expected_blocked, (
                    f"pos ({i},{j}): expected blocked={expected_blocked}, got {mask[0, 0, i, j].item()}"
                )

    def test_image_block_gets_bidirectional_attention(self):
        """Image tokens within the same block should attend to each other bidirectionally."""
        model = _make_model()
        # Pattern: 2 text tokens, 3 image tokens, 2 text tokens
        seq = [self.TEXT_TOKEN, self.TEXT_TOKEN] + [self.IMAGE_TOKEN] * 3 + [self.TEXT_TOKEN, self.TEXT_TOKEN]
        input_ids = self._make_ids(seq)
        mask = model._compute_attention_mask(input_ids)

        assert mask.shape == (1, 1, 7, 7)
        # Image positions 2, 3, 4 should attend to each other (bidirectional = not blocked)
        for i in range(2, 5):
            for j in range(2, 5):
                assert not mask[0, 0, i, j].item(), f"Image pos ({i},{j}) should be unblocked (bidirectional)"

    def test_text_after_image_cannot_attend_back_to_image_beyond_causal(self):
        """Text token after image block uses causal attention (cannot look into future)."""
        model = _make_model()
        # 3 image tokens then 2 text tokens
        seq = [self.IMAGE_TOKEN] * 3 + [self.TEXT_TOKEN, self.TEXT_TOKEN]
        input_ids = self._make_ids(seq)
        mask = model._compute_attention_mask(input_ids)

        # Text token at pos 3 can look back to pos 0,1,2 (causal allows it), but pos 4 is blocked
        assert mask[0, 0, 3, 4].item() is True  # future token: blocked

    def test_two_separate_image_blocks(self):
        """Two distinct image blocks do not attend across blocks."""
        model = _make_model()
        # img_block_1 (pos 0-1), text (pos 2), img_block_2 (pos 3-4)
        seq = [self.IMAGE_TOKEN, self.IMAGE_TOKEN, self.TEXT_TOKEN, self.IMAGE_TOKEN, self.IMAGE_TOKEN]
        input_ids = self._make_ids(seq)
        mask = model._compute_attention_mask(input_ids)

        # Within block 1: positions 0,1 attend to each other
        assert not mask[0, 0, 0, 1].item(), "block1 pos 0→1 should be unblocked"
        assert not mask[0, 0, 1, 0].item(), "block1 pos 1→0 should be unblocked"

        # Within block 2: positions 3,4 attend to each other
        assert not mask[0, 0, 3, 4].item(), "block2 pos 3→4 should be unblocked"
        assert not mask[0, 0, 4, 3].item(), "block2 pos 4→3 should be unblocked"

        # Across blocks: block2 cannot attend back to block1 bidirectionally
        # (pos 3 looking at pos 0: this is causal, so it's allowed, but NOT because of bidirectional mask)
        # The key point: block1 CANNOT attend forward to block2 (causal blocks it)
        assert mask[0, 0, 0, 3].item() is True, "block1 pos 0 should be blocked from future block2 pos 3"

    def test_not_pre_process_returns_none(self):
        """Returns None when pre_process=False (PP pipeline stage)."""
        model = _make_model()
        model.pre_process = False
        input_ids = self._make_ids([self.TEXT_TOKEN] * 4)
        result = model._compute_attention_mask(input_ids)
        assert result is None

    def test_output_shape_batch_size_2(self):
        """Mask shape is [B, 1, S, S] for batch_size=2."""
        model = _make_model()
        seq = [self.TEXT_TOKEN, self.IMAGE_TOKEN, self.TEXT_TOKEN]
        input_ids = torch.tensor([seq, seq], dtype=torch.long)
        mask = model._compute_attention_mask(input_ids)
        assert mask.shape == (2, 1, 3, 3)

    def test_audio_tokens_follow_causal_mask(self):
        """Audio tokens do not receive image-style bidirectional attention."""
        model = _make_model()
        model.config.audio_token_id = 258_881
        seq = [model.config.audio_token_id, model.config.audio_token_id, self.TEXT_TOKEN]
        input_ids = self._make_ids(seq)

        mask = model._compute_attention_mask(input_ids)

        assert mask[0, 0, 0, 1].item() is True
        assert mask[0, 0, 1, 0].item() is False


class TestHFPrecisionBuffers:
    def test_keep_hf_precision_buffers_in_fp32(self):
        class RopeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = object()
                self.rope_type = "default"
                self.attention_scaling = None
                self.register_buffer("inv_freq", torch.ones(2, dtype=torch.bfloat16), persistent=False)
                self.register_buffer("original_inv_freq", torch.ones(2, dtype=torch.bfloat16), persistent=False)

            def compute_default_rope_parameters(self, config, device):
                del config
                return torch.arange(2, device=device, dtype=torch.float32), 1.5

        class AudioPositionModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 4
                self.register_buffer("inv_timescales", torch.ones(1, 1, 2, dtype=torch.bfloat16), persistent=False)

        class SoftcapModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("softcap", torch.tensor(30.0, dtype=torch.bfloat16), persistent=False)

        module = torch.nn.Module()
        module.rope = RopeModule()
        module.audio_position = AudioPositionModule()
        module.softcap_module = SoftcapModule()

        _keep_hf_precision_buffers_in_fp32(module)

        assert module.rope.inv_freq.dtype == torch.float32
        assert module.rope.original_inv_freq.dtype == torch.float32
        assert module.rope.attention_scaling == 1.5
        assert module.audio_position.inv_timescales.dtype == torch.float32
        assert module.softcap_module.softcap.dtype == torch.float32


class TestFallbackEmbedders:
    def test_simple_vision_embedder_projects_to_text_hidden(self):
        embedder = _SimpleVisionEmbedder(vision_hidden=3, text_hidden=5, eps=1e-6)

        out = embedder(torch.ones(2, 4, 3))

        assert out.shape == (2, 4, 5)

    def test_simple_audio_embedder_projects_to_text_hidden(self):
        embedder = _SimpleAudioEmbedder(audio_proj_dim=3, text_hidden=5, eps=1e-6)

        out = embedder(torch.ones(2, 4, 3))

        assert out.shape == (2, 4, 5)


class TestScatterModalityFeatures:
    def test_scatter_modality_features_replaces_token_slots(self):
        model = _make_model()
        inputs = torch.zeros(1, 3, 4)
        input_ids = torch.tensor([[IMAGE_TOKEN_ID, 7, IMAGE_TOKEN_ID]])
        features = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])

        out = model._scatter_modality_features(inputs, input_ids, features, IMAGE_TOKEN_ID, "image")

        torch.testing.assert_close(out[0, 0], features[0, 0])
        torch.testing.assert_close(out[0, 1], torch.zeros(4))
        torch.testing.assert_close(out[0, 2], features[0, 1])

    def test_scatter_modality_features_rejects_mismatched_counts(self):
        model = _make_model()
        inputs = torch.zeros(1, 3, 4)
        input_ids = torch.tensor([[IMAGE_TOKEN_ID, 7, IMAGE_TOKEN_ID]])
        features = torch.ones(1, 1, 4)

        with pytest.raises(ValueError, match="image token count mismatch"):
            model._scatter_modality_features(inputs, input_ids, features, IMAGE_TOKEN_ID, "image")

    def test_forward_scatters_audio_features(self):
        model = _make_model()
        model.config.audio_token_id = 258_881
        model.config.text_config.pad_token_id = 0
        model.language_model = Mock()
        model.language_model.config = SimpleNamespace(scale_embeddings_by_hidden_size=False, hidden_size=4)
        model.language_model.embedding.return_value = torch.zeros(3, 1, 4)
        model.language_model.forward.return_value = torch.zeros(3, 1, 8)
        model.audio_tower = Mock()
        model.get_audio_features = Mock(return_value=torch.full((1, 2, 4), 9.0))
        input_ids = torch.tensor([[model.config.audio_token_id, model.config.audio_token_id, 5]])

        Gemma4VLModel.forward(model, input_ids=input_ids, input_features=torch.ones(1, 8, 128))

        decoder_input = model.language_model.forward.call_args.kwargs["decoder_input"]
        assert decoder_input.shape == (3, 1, 4)
        torch.testing.assert_close(decoder_input[:2], torch.full((2, 1, 4), 9.0))
        torch.testing.assert_close(decoder_input[2], torch.zeros(1, 4))

    def test_forward_scatters_sequence_parallel_decoder_input(self):
        model = _make_model()
        model.config.sequence_parallel = True
        model.config.audio_token_id = 258_881
        model.language_model = Mock()
        model.language_model.forward.return_value = "outputs"
        inputs_embeds = torch.ones(1, 2, 4)
        input_ids = torch.tensor([[7, 8]])
        calls = []

        def fake_scatter(tensor, *, group=None):
            assert group is None
            calls.append(tensor)
            return tensor + 1.0

        with patch(
            "megatron.bridge.models.gemma_vl.modeling_gemma4_vl.scatter_to_sequence_parallel_region", fake_scatter
        ):
            out = Gemma4VLModel.forward(model, input_ids=input_ids, inputs_embeds=inputs_embeds)

        assert out == "outputs"
        torch.testing.assert_close(calls[0], inputs_embeds.transpose(1, 0).contiguous())
        torch.testing.assert_close(model.language_model.forward.call_args.kwargs["decoder_input"], calls[0] + 1.0)


class TestFeatureExtractionAndFreeze:
    class _Tower(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(last_hidden_state=self.output)

    class _Embedder(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, x):
            return x + self.offset

    class _ParamHolder:
        def __init__(self):
            self.param = torch.nn.Parameter(torch.ones(1))

        def parameters(self):
            return [self.param]

    def test_get_image_features_runs_tower_and_embedder(self):
        model = _make_model()
        hidden = torch.ones(1, 2, 3)
        object.__setattr__(model, "vision_tower", self._Tower(hidden))
        object.__setattr__(model, "embed_vision", self._Embedder(offset=2.0))
        pixel_values = torch.zeros(1, 2, 3)
        image_position_ids = torch.zeros(1, 2, 2, dtype=torch.long)

        out = Gemma4VLModel.get_image_features(model, pixel_values, image_position_ids=image_position_ids)

        torch.testing.assert_close(out, hidden + 2.0)
        assert model.vision_tower.calls[-1]["pixel_values"] is pixel_values
        assert model.vision_tower.calls[-1]["pixel_position_ids"] is image_position_ids

    def test_get_audio_features_runs_tower_and_embedder(self):
        model = _make_model()
        hidden = torch.ones(1, 2, 3)
        object.__setattr__(model, "audio_tower", self._Tower(hidden))
        object.__setattr__(model, "embed_audio", self._Embedder(offset=3.0))
        input_features = torch.zeros(1, 8, 128)

        out = Gemma4VLModel.get_audio_features(model, input_features)

        torch.testing.assert_close(out, hidden + 3.0)
        assert model.audio_tower.calls[-1]["input_features"] is input_features

    def test_freeze_updates_requested_modules_only(self):
        model = SimpleNamespace(
            language_model=self._ParamHolder(),
            vision_tower=self._ParamHolder(),
            embed_vision=self._ParamHolder(),
            audio_tower=self._ParamHolder(),
            embed_audio=self._ParamHolder(),
        )

        Gemma4VLModel.freeze(
            model,
            freeze_language_model=True,
            freeze_vision_model=False,
            freeze_vision_projection=True,
            freeze_audio_model=True,
            freeze_audio_projection=False,
        )

        assert model.language_model.param.requires_grad is False
        assert model.vision_tower.param.requires_grad is True
        assert model.embed_vision.param.requires_grad is False
        assert model.audio_tower.param.requires_grad is False
        assert model.embed_audio.param.requires_grad is True

    def test_freeze_ignores_requested_but_missing_optional_modules(self):
        model = SimpleNamespace(language_model=self._ParamHolder())

        Gemma4VLModel.freeze(
            model,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=True,
            freeze_audio_model=True,
            freeze_audio_projection=True,
        )

        assert model.language_model.param.requires_grad is False
