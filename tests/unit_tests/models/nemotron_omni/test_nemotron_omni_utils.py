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

import pytest
import torch

from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import (
    inference_merged_sequence_length,
    inference_num_image_tiles,
    select_inference_next_token,
    temporal_model_frames,
)


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


def test_inference_merged_sequence_length_uses_exact_image_replacements():
    input_ids = torch.tensor([[10, -200, 11, -200, 12]])

    dynamic_length = inference_merged_sequence_length(
        input_ids,
        image_token_index=-200,
        num_image_tiles=torch.tensor([3, 2]),
        image_seq_len=1,
    )
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
    with pytest.raises(ValueError, match="Expected 2 num_image_tiles entries"):
        inference_merged_sequence_length(
            torch.tensor([[10, -200, 11, -200, 12]]),
            image_token_index=-200,
            num_image_tiles=torch.tensor([3]),
            image_seq_len=1,
        )
