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

"""Unit tests for Gemma4VLModel helpers (no GPU / Megatron distributed required)."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel


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
                    f"pos ({i},{j}): expected blocked={expected_blocked}, got {mask[0,0,i,j].item()}"
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
                assert not mask[0, 0, i, j].item(), (
                    f"Image pos ({i},{j}) should be unblocked (bidirectional)"
                )

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
