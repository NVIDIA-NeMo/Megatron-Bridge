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

from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.rope import (
    get_llm_pos_ids_for_vision,
    get_rope_index,
)


IMAGE_TOKEN_ID = 900
VIDEO_TOKEN_ID = 901
AUDIO_TOKEN_ID = 902
VISION_START_TOKEN_ID = 903
AUDIO_START_TOKEN_ID = 904
VISION_END_TOKEN_ID = 905
AUDIO_END_TOKEN_ID = 906


@pytest.mark.unit
@pytest.mark.parametrize(
    "temporal_positions",
    [
        [0.0, 3.25, 6.5, 9.75],
        [0.0, 6.5, 13.0, 19.5],
        [0.0, 12.5, 25.0, 37.5],
        [0.0, 13.0, 26.0, 39.0],
    ],
    ids=["8fps", "4fps", "half_second_25_positions", "integer_positions"],
)
def test_vision_positions_preserve_temporal_coordinates(temporal_positions: list[float]) -> None:
    position_ids = get_llm_pos_ids_for_vision(
        start_idx=7,
        vision_idx=0,
        spatial_merge_size=2,
        t_index=torch.tensor(temporal_positions),
        grid_hs=torch.tensor([4]),
        grid_ws=torch.tensor([4]),
    )
    expected = (
        torch.stack(
            [
                torch.tensor(temporal_positions).repeat_interleave(4),
                torch.tensor([0.0, 0.0, 1.0, 1.0]).repeat(4),
                torch.tensor([0.0, 1.0, 0.0, 1.0]).repeat(4),
            ]
        )
        + 7
    )

    torch.testing.assert_close(position_ids, expected, rtol=0, atol=0, check_dtype=False)
    assert position_ids.dtype == torch.float32


@pytest.mark.unit
def test_video_rope_preserves_fractional_positions_and_text_offset() -> None:
    input_ids = torch.LongTensor([[17, VISION_START_TOKEN_ID] + [VIDEO_TOKEN_ID] * 8 + [VISION_END_TOKEN_ID, 18]])
    position_ids, deltas = get_rope_index(
        spatial_merge_size=2,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        position_id_per_seconds=13,
        input_ids=input_ids,
        video_grid_thw=torch.LongTensor([[2, 4, 4]]),
        second_per_grids=torch.tensor([0.25]),
    )
    expected = torch.tensor(
        [
            [[0, 1, 2, 2, 2, 2, 5.25, 5.25, 5.25, 5.25, 6.25, 7.25]],
            [[0, 1, 2, 2, 3, 3, 2, 2, 3, 3, 6.25, 7.25]],
            [[0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 6.25, 7.25]],
        ]
    )

    torch.testing.assert_close(position_ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(deltas, torch.tensor([[-3.75]]), rtol=0, atol=0)


@pytest.mark.unit
def test_video_audio_rope_preserves_fractional_interleaving() -> None:
    input_ids = torch.LongTensor(
        [
            [17, VISION_START_TOKEN_ID, AUDIO_START_TOKEN_ID]
            + [VIDEO_TOKEN_ID]
            + [AUDIO_TOKEN_ID] * 4
            + [VIDEO_TOKEN_ID, AUDIO_END_TOKEN_ID, VISION_END_TOKEN_ID, 18]
        ]
    )
    position_ids, deltas = get_rope_index(
        spatial_merge_size=2,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
        position_id_per_seconds=13,
        input_ids=input_ids,
        video_grid_thw=torch.LongTensor([[2, 2, 2]]),
        audio_seqlens=torch.LongTensor([27]),
        second_per_grids=torch.tensor([0.25]),
        use_audio_in_video=True,
    )
    expected = torch.tensor(
        [
            [[0, 1, 2, 3, 3, 4, 5, 6, 6.25, 7.25, 8.25, 9.25]],
            [[0, 1, 2, 3, 3, 4, 5, 6, 3, 7.25, 8.25, 9.25]],
            [[0, 1, 2, 3, 3, 4, 5, 6, 3, 7.25, 8.25, 9.25]],
        ]
    )

    torch.testing.assert_close(position_ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(deltas, torch.tensor([[-1.75]]), rtol=0, atol=0)
