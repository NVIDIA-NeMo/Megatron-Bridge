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

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.diffusion.common.flow_matching.flow_matching_pipeline import LinearInterpolationSchedule
from megatron.bridge.diffusion.models.wan.longlive_wan_utils import (
    apply_longlive_noising,
    build_longlive_loss_mask,
    build_teacher_forcing_self_attention_mask,
    select_longlive_chunks,
    split_self_attention_mask_rows,
    temporal_frame_token_spans,
)
from megatron.bridge.diffusion.models.wan.utils import sequence_parallel_partition_indices


def test_temporal_frame_token_spans_follow_wan_patchify_order():
    spans = temporal_frame_token_spans((3, 2, 4), sample_start=5)

    assert spans == [(5, 13), (13, 21), (21, 29)]


def test_select_longlive_chunks_first_valid_uses_clean_prefix_then_target():
    chunks = select_longlive_chunks(
        grid_sizes=torch.tensor([[4, 2, 2]], dtype=torch.int32),
        seq_len_q=torch.tensor([16], dtype=torch.int32),
        seq_len_q_padded=torch.tensor([16], dtype=torch.int32),
        strategy="first-valid",
    )

    chunk = chunks[0]
    assert chunk is not None
    assert chunk.target_frame_start == 1
    assert chunk.target_start == 4
    assert chunk.target_end == 8


def test_select_longlive_chunks_random_stays_in_valid_target_range():
    generator = torch.Generator().manual_seed(1234)
    chunks = select_longlive_chunks(
        grid_sizes=torch.tensor([[6, 1, 1]], dtype=torch.int32),
        seq_len_q=torch.tensor([6], dtype=torch.int32),
        seq_len_q_padded=torch.tensor([6], dtype=torch.int32),
        strategy="random",
        generator=generator,
    )

    chunk = chunks[0]
    assert chunk is not None
    assert 1 <= chunk.target_frame_start <= 5


def test_select_longlive_chunks_marks_too_short_samples_for_fallback():
    chunks = select_longlive_chunks(
        grid_sizes=torch.tensor([[1, 2, 2]], dtype=torch.int32),
        seq_len_q=torch.tensor([4], dtype=torch.int32),
        seq_len_q_padded=torch.tensor([4], dtype=torch.int32),
        strategy="first-valid",
    )

    assert chunks == [None]


def test_apply_longlive_noising_only_replaces_target_tokens():
    chunks = select_longlive_chunks(
        grid_sizes=torch.tensor([[3, 1, 1]], dtype=torch.int32),
        seq_len_q=torch.tensor([3], dtype=torch.int32),
        seq_len_q_padded=torch.tensor([3], dtype=torch.int32),
        strategy="first-valid",
    )
    clean_latents = torch.zeros(1, 3, 2)
    noise = torch.ones_like(clean_latents)
    sigma = torch.tensor([0.25])

    mixed = apply_longlive_noising(clean_latents, noise, sigma, chunks, LinearInterpolationSchedule())

    assert torch.equal(mixed[:, 0], clean_latents[:, 0])
    assert torch.allclose(mixed[:, 1], torch.full((1, 2), 0.25))
    assert torch.equal(mixed[:, 2], clean_latents[:, 2])


def test_build_longlive_loss_mask_keeps_only_target_tokens_and_preserves_padding():
    chunks = select_longlive_chunks(
        grid_sizes=torch.tensor([[3, 1, 1]], dtype=torch.int32),
        seq_len_q=torch.tensor([3], dtype=torch.int32),
        seq_len_q_padded=torch.tensor([4], dtype=torch.int32),
        strategy="first-valid",
    )
    base_mask = torch.tensor([[1.0], [1.0], [1.0], [0.0]])

    loss_mask = build_longlive_loss_mask(base_mask, chunks)

    assert torch.equal(loss_mask, torch.tensor([[0.0], [1.0], [0.0], [0.0]]))


def test_teacher_forcing_mask_blocks_target_future_and_history_target_attention():
    chunks = select_longlive_chunks(
        grid_sizes=torch.tensor([[4, 1, 1]], dtype=torch.int32),
        seq_len_q=torch.tensor([4], dtype=torch.int32),
        seq_len_q_padded=torch.tensor([4], dtype=torch.int32),
        strategy="first-valid",
    )

    mask = build_teacher_forcing_self_attention_mask(total_seq_len=4, chunks=chunks, device=torch.device("cpu"))
    mask_2d = mask[0, 0]

    assert mask.shape == (1, 1, 4, 4)
    assert not mask_2d[0, 0]
    assert mask_2d[0, 1]
    assert mask_2d[0, 2]
    assert not mask_2d[1, 0]
    assert not mask_2d[1, 1]
    assert mask_2d[1, 2]


def test_cp_row_split_preserves_target_loss_count_once_with_disjoint_partitions():
    chunks = select_longlive_chunks(
        grid_sizes=torch.tensor([[4, 1, 1]], dtype=torch.int32),
        seq_len_q=torch.tensor([4], dtype=torch.int32),
        seq_len_q_padded=torch.tensor([4], dtype=torch.int32),
        strategy="first-valid",
    )
    loss_mask = build_longlive_loss_mask(torch.ones(4, 1), chunks)
    rank0_rows = torch.tensor([0, 2], dtype=torch.long)
    rank1_rows = torch.tensor([1, 3], dtype=torch.long)

    rank0_count = loss_mask.index_select(dim=0, index=rank0_rows).sum()
    rank1_count = loss_mask.index_select(dim=0, index=rank1_rows).sum()

    assert int(rank0_count.item() + rank1_count.item()) == 1


def test_split_self_attention_mask_rows_selects_local_query_rows():
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[:, :, 2, :] = True

    local = split_self_attention_mask_rows(mask, torch.tensor([0, 2]))

    assert local.shape == (1, 1, 2, 4)
    assert not local[0, 0, 0].any()
    assert local[0, 0, 1].all()


def test_sequence_parallel_partition_indices_match_megatron_contiguous_split():
    rank2 = sequence_parallel_partition_indices(
        total_s=16,
        tp_size=4,
        tp_rank=2,
    )

    assert torch.equal(rank2, torch.tensor([8, 9, 10, 11]))


def test_sequence_parallel_partition_indices_require_even_split():
    with pytest.raises(ValueError, match="divisible by tensor parallel size"):
        sequence_parallel_partition_indices(
            total_s=15,
            tp_size=4,
            tp_rank=0,
        )


def test_longlive_wan_forward_step_uses_longlive_pipeline():
    from megatron.bridge.diffusion.models.wan.longlive_wan_step import LongLiveWanForwardStep

    step = LongLiveWanForwardStep(chunk_selection_strategy="first-valid")

    assert step.chunk_selection_strategy == "first-valid"
    assert step.diffusion_pipeline.target_chunk_frames == 1
    assert step.diffusion_pipeline.teacher_forcing_mask_max_tokens == 0


def test_longlive_pipeline_uses_dense_mask_only_when_decoder_supports_it():
    from megatron.bridge.diffusion.models.wan.longlive_wan_step import (
        LongLiveWanAdapter,
        LongLiveWanFlowMatchingPipeline,
    )

    class DecoderWithMask:
        def forward(self, hidden_states, attention_mask, self_attention_mask=None):
            return hidden_states

    class DecoderWithoutMask:
        def forward(self, hidden_states, attention_mask):
            return hidden_states

    pipeline = LongLiveWanFlowMatchingPipeline(
        model_adapter=LongLiveWanAdapter(),
        teacher_forcing_mask_max_tokens=4,
    )

    assert (
        pipeline.should_build_explicit_self_attention_mask(
            4,
            SimpleNamespace(config=SimpleNamespace(), decoder=DecoderWithMask()),
        )
        is True
    )
    assert (
        pipeline.should_build_explicit_self_attention_mask(
            281600,
            SimpleNamespace(config=SimpleNamespace(window_size=(24 * 22 * 40, 0)), decoder=DecoderWithoutMask()),
        )
        is False
    )
    assert (
        pipeline.should_build_explicit_self_attention_mask(
            4,
            SimpleNamespace(config=SimpleNamespace(window_size=(24 * 22 * 40, 0)), decoder=DecoderWithoutMask()),
        )
        is False
    )

    with pytest.raises(ValueError, match="Configure model.window_size"):
        pipeline.should_build_explicit_self_attention_mask(
            4,
            SimpleNamespace(config=SimpleNamespace(), decoder=DecoderWithoutMask()),
        )


def test_longlive_pipeline_default_requires_windowed_attention():
    from megatron.bridge.diffusion.models.wan.longlive_wan_step import (
        LongLiveWanAdapter,
        LongLiveWanFlowMatchingPipeline,
    )

    class DecoderWithoutMask:
        def forward(self, hidden_states, attention_mask):
            return hidden_states

    pipeline = LongLiveWanFlowMatchingPipeline(model_adapter=LongLiveWanAdapter())

    assert (
        pipeline.should_build_explicit_self_attention_mask(
            4,
            SimpleNamespace(config=SimpleNamespace(window_size=(24 * 22 * 40, 0)), decoder=DecoderWithoutMask()),
        )
        is False
    )

    with pytest.raises(ValueError, match="requires model.window_size"):
        pipeline.should_build_explicit_self_attention_mask(
            4,
            SimpleNamespace(config=SimpleNamespace(), decoder=DecoderWithoutMask()),
        )


def test_run_recipe_registry_loads_longlive_wan_step():
    from scripts.training.run_recipe import STEP_FUNCTIONS, load_forward_step

    from megatron.bridge.diffusion.models.wan.longlive_wan_step import LongLiveWanForwardStep

    assert "longlive_wan_step" in STEP_FUNCTIONS
    assert isinstance(load_forward_step("longlive_wan_step", mode="pretrain"), LongLiveWanForwardStep)
