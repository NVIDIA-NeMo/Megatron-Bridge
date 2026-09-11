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

import numpy as np
import pytest
import torch

import megatron.bridge.models.nemotron_omni.nemotron_omni_utils as omni_utils
from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import (
    compute_mel_features,
    compute_mel_features_with_length,
    inference_expanded_image_token_counts,
    inference_merged_sequence_length,
    inference_num_image_tiles,
    select_inference_next_token,
    temporal_model_frames,
    valid_audio_feature_lengths,
)
from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens


def test_compute_mel_features_preserves_physical_rows_and_mask_length(monkeypatch):
    extractor_kwargs = {}

    class _Extractor:
        def __call__(self, waveform, **kwargs):
            del waveform
            extractor_kwargs.update(kwargs)
            return {
                "input_features": torch.ones(1, 9, 4),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]]),
            }

    monkeypatch.setattr(omni_utils, "_parakeet_feature_extractor", lambda num_mel_bins, sampling_rate: _Extractor())

    mel, valid_length = compute_mel_features_with_length([0.0] * 1280, num_mel_bins=4)

    assert mel.shape == (9, 4)
    assert valid_length == 8
    assert extractor_kwargs["return_attention_mask"] is True
    assert compute_mel_features([0.0] * 1280, num_mel_bins=4).shape == (9, 4)


@pytest.mark.parametrize(
    ("attention_mask", "match"),
    [
        (torch.tensor([[1, 0, 1]]), "contiguous valid prefix"),
        (torch.tensor([[0, 0, 0]]), "at least one valid frame"),
        (torch.tensor([[1.0, 0.5, 0.0]]), "binary values"),
    ],
)
def test_valid_audio_feature_lengths_rejects_invalid_masks(attention_mask, match):
    with pytest.raises(ValueError, match=match):
        valid_audio_feature_lengths(attention_mask, num_frames=3)


def test_real_parakeet_extractor_marks_centered_stft_boundary_frame_invalid():
    mel, valid_length = compute_mel_features_with_length(np.zeros(1280, dtype=np.float32))

    assert mel.shape == (9, 128)
    assert valid_length == 8


def test_temporal_model_frames_duplicates_single_frame_for_temporal_embedder():
    frame = object()

    assert temporal_model_frames([frame], 2) == [frame, frame]


def test_temporal_model_frames_preserves_odd_trailing_group_for_mcore_padding():
    frames = [object(), object(), object()]

    assert temporal_model_frames(frames, 2) == frames


def test_temporal_model_frames_preserves_zero_and_non_temporal_inputs():
    frame = object()

    assert temporal_model_frames([], 2) == []
    assert temporal_model_frames([frame], 1) == [frame]


def test_temporal_model_frames_rejects_invalid_patch_size():
    with pytest.raises(ValueError, match="must be greater than 0"):
        temporal_model_frames([object()], 0)


def test_inference_num_image_tiles_uses_post_shuffle_dynamic_image_counts():
    imgs_sizes = torch.tensor([[512, 512], [512, 256], [256, 256]])

    counts = inference_num_image_tiles(imgs_sizes, patch_dim=16)

    assert counts.tolist() == [256, 128, 64]


def test_inference_num_image_tiles_uses_one_entry_per_temporal_tubelet():
    imgs_sizes = torch.tensor([[512, 512]] * 7)

    counts = inference_num_image_tiles(
        imgs_sizes,
        patch_dim=16,
        num_frames=torch.tensor([2, 3, 2]),
        temporal_patch_size=2,
    )

    assert counts.tolist() == [1, 1, 1, 1]


def test_inference_num_image_tiles_rejects_inconsistent_temporal_metadata():
    with pytest.raises(ValueError, match="account for every row"):
        inference_num_image_tiles(
            torch.tensor([[512, 512], [512, 512]]),
            patch_dim=16,
            num_frames=torch.tensor([3]),
            temporal_patch_size=2,
        )


def test_inference_num_image_tiles_rejects_unshufflable_image_grid():
    with pytest.raises(ValueError, match="pixel_shuffle_factor"):
        inference_num_image_tiles(torch.tensor([[528, 512]]), patch_dim=16)


def test_inference_expanded_image_token_counts_aggregates_dynamic_tiles_by_media():
    counts = inference_expanded_image_token_counts(
        torch.tensor([256, 128, 64]),
        torch.tensor([2, 1]),
    )

    assert counts.tolist() == [384, 64]


def test_inference_expanded_image_token_counts_applies_temporal_feature_width():
    counts = inference_expanded_image_token_counts(
        torch.ones(3, dtype=torch.int),
        torch.ones(3, dtype=torch.int),
        feature_multiplier=256,
    )

    assert counts.tolist() == [256, 256, 256]


def test_inference_expanded_image_token_counts_rejects_incomplete_tile_ownership():
    with pytest.raises(ValueError, match="account for every tile"):
        inference_expanded_image_token_counts(torch.tensor([256, 128]), torch.tensor([1]))


def test_canonical_dynamic_pre_expansion_preserves_legacy_merged_length():
    image_token_id = -200
    img_start_id = -201
    img_end_id = -202
    processor_input_ids = torch.tensor([[10, img_start_id, image_token_id, img_end_id, 11]])
    tile_feature_counts = inference_num_image_tiles(
        torch.tensor([[512, 512], [512, 256]]),
        patch_dim=16,
    )

    legacy_compact_ids = adjust_image_tokens(
        processor_input_ids,
        torch.tensor([2]),
        img_start_id,
        img_end_id,
    )
    with pytest.warns(FutureWarning, match="deprecated"):
        legacy_merged_length = inference_merged_sequence_length(
            legacy_compact_ids,
            image_token_index=image_token_id,
            num_image_tiles=tile_feature_counts,
            image_seq_len=1,
        )

    expanded_counts = inference_expanded_image_token_counts(tile_feature_counts, torch.tensor([2]))
    canonical_input_ids = adjust_image_tokens(
        processor_input_ids,
        expanded_counts,
        img_start_id,
        img_end_id,
    )

    assert tile_feature_counts.tolist() == [256, 128]
    assert expanded_counts.tolist() == [384]
    assert canonical_input_ids.shape[1] == legacy_merged_length
    assert int((canonical_input_ids == image_token_id).sum()) == 384


def test_canonical_temporal_pre_expansion_preserves_legacy_merged_length():
    image_token_id = -200
    img_start_id = -201
    img_end_id = -202
    processor_input_ids = torch.tensor(
        [[10, img_start_id, image_token_id, img_end_id, 11, img_start_id, image_token_id, img_end_id, 12]]
    )
    tubelet_counts = inference_num_image_tiles(
        torch.tensor([[512, 512]] * 4),
        patch_dim=16,
        num_frames=torch.tensor([4]),
        temporal_patch_size=2,
    )

    with pytest.warns(FutureWarning, match="deprecated"):
        legacy_merged_length = inference_merged_sequence_length(
            processor_input_ids,
            image_token_index=image_token_id,
            num_image_tiles=tubelet_counts,
            image_seq_len=256,
        )

    expanded_counts = inference_expanded_image_token_counts(
        tubelet_counts,
        torch.ones_like(tubelet_counts),
        feature_multiplier=256,
    )
    canonical_input_ids = adjust_image_tokens(
        processor_input_ids,
        expanded_counts,
        img_start_id,
        img_end_id,
    )

    assert tubelet_counts.tolist() == [1, 1]
    assert expanded_counts.tolist() == [256, 256]
    assert canonical_input_ids.shape[1] == legacy_merged_length
    assert int((canonical_input_ids == image_token_id).sum()) == 512


def test_inference_merged_sequence_length_uses_exact_image_replacements():
    input_ids = torch.tensor([[10, -200, 11, -200, 12]])

    with pytest.warns(FutureWarning, match="deprecated"):
        dynamic_length = inference_merged_sequence_length(
            input_ids,
            image_token_index=-200,
            num_image_tiles=torch.tensor([3, 2]),
            image_seq_len=1,
        )
    with pytest.warns(FutureWarning, match="deprecated"):
        temporal_length = inference_merged_sequence_length(
            input_ids,
            image_token_index=-200,
            num_image_tiles=torch.tensor([1, 1]),
            image_seq_len=256,
        )

    assert dynamic_length == 8
    assert temporal_length == 515


def test_select_inference_next_token_ignores_pipeline_padding_logits():
    logits = torch.zeros(1, 10, 4)
    logits[0, 7, 1] = 5
    logits[0, -1, 3] = 50

    next_token = select_inference_next_token(logits, merged_sequence_length=8)

    assert next_token.tolist() == [[1]]


def test_inference_merged_sequence_length_rejects_misaligned_image_metadata():
    with pytest.warns(FutureWarning, match="deprecated"):
        with pytest.raises(ValueError, match="Expected 2 num_image_tiles entries"):
            inference_merged_sequence_length(
                torch.tensor([[10, -200, 11, -200, 12]]),
                image_token_index=-200,
                num_image_tiles=torch.tensor([3]),
                image_seq_len=1,
            )
