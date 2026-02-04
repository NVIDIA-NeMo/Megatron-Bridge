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

import pytest
import torch

from megatron.bridge.training.utils.visual_inputs import Qwen2_5_VLVisualInputs
from megatron.bridge.training.vlm_step import (
    forward_step,
    get_batch,
    get_batch_from_iterator,
    pack_batch_sequences,
)


class _Iterator:
    def __init__(self, batch):
        self.batch = batch
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return self.batch


def _make_batch(device="cpu"):
    # Minimal text tensors
    tokens = torch.tensor([[1, 2, 3]], device=device)
    input_ids = tokens.clone()
    position_ids = torch.tensor([[0, 1, 2]], device=device)
    labels = torch.tensor([[2, 3, 4]], device=device)
    loss_mask = torch.ones_like(labels, dtype=torch.float, device=device)
    attention_mask = torch.ones_like(tokens, dtype=torch.bool, device=device)

    # Visual inputs container
    pixel_values = torch.randn(1, 2, 3, 4, 4, device=device)
    image_grid_thw = torch.tensor([[[1, 2, 2], [1, 2, 2]]], device=device)
    vi = Qwen2_5_VLVisualInputs(pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    batch = {
        "tokens": tokens,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "visual_inputs": vi,
    }
    return batch


def test_get_batch_from_iterator_moves_visual_inputs_to_cuda(monkeypatch):
    # Simulate Training on CPU-only env by making .cuda a no-op that returns the same tensor
    class _NoCudaTensor(torch.Tensor):
        def cuda(self, non_blocking=False):  # type: ignore[override]
            return self

    def _as_nocuda(t):
        return t.as_subclass(_NoCudaTensor)

    batch = _make_batch()
    # Replace tensors with _NoCudaTensor so calling .cuda works without a GPU
    for k in ["tokens", "input_ids", "position_ids", "labels", "loss_mask", "attention_mask"]:
        batch[k] = _as_nocuda(batch[k])
    vi = batch["visual_inputs"]
    vi.pixel_values = _as_nocuda(vi.pixel_values)
    vi.image_grid_thw = _as_nocuda(vi.image_grid_thw)

    it = _Iterator(batch)
    out = get_batch_from_iterator(
        it,
        use_mtp=False,
        skip_getting_attention_mask_from_dataset=True,
        is_first_pp_stage=True,
        is_last_pp_stage=True,
    )

    assert "visual_inputs" in out
    out_vi = out["visual_inputs"]
    assert isinstance(out_vi, Qwen2_5_VLVisualInputs)
    # Verify fields are preserved
    assert out_vi.pixel_values is not None and out_vi.image_grid_thw is not None


class _MockProcessGroup:
    """Mock process group with rank/size methods for testing."""

    def rank(self):
        return 0

    def size(self):
        return 1


class _MockPGCollection:
    """Mock PG collection for testing."""

    def __init__(self, cp_size=1):
        self.pp = _MockProcessGroup()
        self._cp_size = cp_size

    @property
    def cp(self):
        pg = _MockProcessGroup()
        pg.size = lambda: self._cp_size
        return pg


def test_get_batch_padding_paths(monkeypatch):
    # Simulate both first and last pipeline stages so tensors are returned
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_first_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_last_stage", lambda pg: True, raising=True)

    # Disable context parallel slicing effects
    monkeypatch.setattr(
        "megatron.core.utils.get_batch_on_this_cp_rank",
        lambda x: x,
        raising=True,
    )

    # Minimal cfg
    cfg = type("Cfg", (), {})()
    cfg.model = type(
        "M",
        (),
        {
            "seq_length": 32,
            "seq_len_interpolation_factor": 1.0,
            "seq_length_interpolation_factor": 1.0,
            "seq_length_interpolation": None,
            "seq_length_interpolation_power": 1.0,
            "pipeline_model_parallel_size": 1,
        },
    )()  # noqa: E501
    cfg.dataset = type("D", (), {"skip_getting_attention_mask_from_dataset": True})()

    # Make batch shorter than 128 to trigger ceil-to-128 padding path
    short_tokens = torch.tensor([[1, 2, 3, 4]])
    vi = Qwen2_5_VLVisualInputs(pixel_values=torch.randn(1, 1, 3, 4, 4), image_grid_thw=torch.tensor([[[1, 2, 2]]]))
    batch = {
        "input_ids": short_tokens,
        "labels": torch.tensor([[2, 3, 4, -100]]),
        "loss_mask": torch.ones_like(short_tokens, dtype=torch.float),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_mask": torch.ones_like(short_tokens, dtype=torch.bool),
        "visual_inputs": vi,
    }

    # Iterator
    it = _Iterator(batch)

    tokens, labels, loss_mask, attention_mask, position_ids, *_ = get_batch(
        it, cfg, use_mtp=False, pg_collection=_MockPGCollection()
    )
    # Length padded up to min(seq_cap, ceil_to_128(4)) == 32
    assert tokens.shape[1] == 32
    assert labels.shape[1] == 32
    assert loss_mask.shape[1] == 32
    assert position_ids.shape[1] == 32


def test_get_batch_enable_packing_path(monkeypatch):
    """Test get_batch with pack_sequences_in_batch=True (enable_packing path)."""
    # Simulate both first and last pipeline stages so tensors are returned
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_first_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_last_stage", lambda pg: True, raising=True)

    # Disable context parallel slicing effects
    monkeypatch.setattr(
        "megatron.core.utils.get_batch_on_this_cp_rank",
        lambda x: x,
        raising=True,
    )

    # Config with packing enabled
    cfg = type("Cfg", (), {})()
    cfg.model = type(
        "M",
        (),
        {
            "seq_length": 64,
            "pipeline_model_parallel_size": 1,
        },
    )()
    cfg.dataset = type(
        "D",
        (),
        {
            "skip_getting_attention_mask_from_dataset": True,
            "pack_sequences_in_batch": True,  # Enable packing
        },
    )()

    # Batch with 2 sequences of different lengths (with padding)
    # Seq 1: [1, 2, 3, 0, 0, 0, 0, 0] - length 3
    # Seq 2: [4, 5, 6, 7, 8, 0, 0, 0] - length 5
    tokens = torch.tensor(
        [
            [1, 2, 3, 0, 0, 0, 0, 0],
            [4, 5, 6, 7, 8, 0, 0, 0],
        ]
    )
    labels = torch.tensor(
        [
            [2, 3, -100, -100, -100, -100, -100, -100],
            [5, 6, 7, 8, -100, -100, -100, -100],
        ]
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    position_ids = torch.arange(8).unsqueeze(0).expand(2, -1).clone()

    vi = Qwen2_5_VLVisualInputs(pixel_values=torch.randn(1, 1, 3, 4, 4), image_grid_thw=torch.tensor([[[1, 2, 2]]]))
    batch = {
        "input_ids": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": None,
        "visual_inputs": vi,
    }

    it = _Iterator(batch)

    (
        out_tokens,
        out_labels,
        out_loss_mask,
        out_attention_mask,
        out_position_ids,
        cu_seqlens,
        max_seqlen,
        visual_inputs,
    ) = get_batch(it, cfg, use_mtp=False, pg_collection=_MockPGCollection())

    # Verify packing occurred
    # With pad_to_multiple_of=1 (cp_size=1), total packed length = 3 + 5 = 8
    assert out_tokens.shape == (1, 8), f"Expected packed shape (1, 8), got {out_tokens.shape}"
    assert out_labels.shape == (1, 8)
    assert out_loss_mask.shape == (1, 8)
    assert out_position_ids.shape == (1, 8)

    # Verify cu_seqlens is populated (not None)
    assert cu_seqlens is not None, "cu_seqlens should be set when packing is enabled"
    assert cu_seqlens.tolist() == [0, 3, 8], f"Expected cu_seqlens [0, 3, 8], got {cu_seqlens.tolist()}"

    # Verify max_seqlen
    assert max_seqlen is not None, "max_seqlen should be set when packing is enabled"
    assert max_seqlen.item() == 5, f"Expected max_seqlen 5, got {max_seqlen.item()}"

    # Verify attention_mask is None for packed sequences
    assert out_attention_mask is None, "attention_mask should be None for packed sequences"

    # Verify packed tokens content
    expected_tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    assert torch.equal(out_tokens.cpu(), expected_tokens), f"Expected {expected_tokens}, got {out_tokens}"

    # Verify visual_inputs passed through
    assert visual_inputs is not None


def test_get_batch_enable_packing_with_cp(monkeypatch):
    """Test get_batch packing with context parallelism (pad_to_multiple_of > 1)."""
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_first_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_last_stage", lambda pg: True, raising=True)
    monkeypatch.setattr(
        "megatron.core.utils.get_batch_on_this_cp_rank",
        lambda x: x,
        raising=True,
    )

    cfg = type("Cfg", (), {})()
    cfg.model = type("M", (), {"seq_length": 64, "pipeline_model_parallel_size": 1})()
    cfg.dataset = type(
        "D",
        (),
        {
            "skip_getting_attention_mask_from_dataset": True,
            "pack_sequences_in_batch": True,
        },
    )()

    # Sequences: length 3 and length 5
    # With CP=2, pad_to_multiple_of = 2*2 = 4
    # Seq 1: 3 -> padded to 4
    # Seq 2: 5 -> padded to 6
    # Total: 4 + 6 = 10
    tokens = torch.tensor(
        [
            [1, 2, 3, 0, 0, 0, 0, 0],
            [4, 5, 6, 7, 8, 0, 0, 0],
        ]
    )
    labels = torch.tensor(
        [
            [2, 3, -100, -100, -100, -100, -100, -100],
            [5, 6, 7, 8, -100, -100, -100, -100],
        ]
    )
    loss_mask = torch.ones_like(tokens, dtype=torch.float)
    position_ids = torch.arange(8).unsqueeze(0).expand(2, -1).clone()

    batch = {
        "input_ids": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": None,
        "visual_inputs": None,
    }

    it = _Iterator(batch)

    # Use CP size of 2
    out_tokens, out_labels, out_loss_mask, _, out_position_ids, cu_seqlens, max_seqlen, _ = get_batch(
        it, cfg, use_mtp=False, pg_collection=_MockPGCollection(cp_size=2)
    )

    # With CP=2, pad_to_multiple_of = 4
    # Seq 1: 3 -> 4, Seq 2: 5 -> 8 (next multiple of 4)
    # Total: 4 + 8 = 12
    assert out_tokens.shape[1] == 12, f"Expected packed length 12, got {out_tokens.shape[1]}"
    assert cu_seqlens.tolist() == [0, 4, 12], f"Expected cu_seqlens [0, 4, 12], got {cu_seqlens.tolist()}"
    assert max_seqlen.item() == 8, f"Expected max_seqlen 8, got {max_seqlen.item()}"


def test_forward_step_schedule_plan(monkeypatch):
    # Configure pipeline last/first to enable labels & loss_mask path
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_first_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_last_stage", lambda pg: True, raising=True)

    # No-op CUDA and CP functions
    monkeypatch.setattr("megatron.core.utils.get_batch_on_this_cp_rank", lambda x: x, raising=True)

    # Create a proper mock process group with rank/size methods
    class _MockProcessGroup:
        def rank(self):
            return 0

        def size(self):
            return 1

    # Create mock pg_collection with proper process groups
    class _MockPGCollection:
        def __init__(self):
            self.pp = _MockProcessGroup()
            self.cp = _MockProcessGroup()

    # Dummy model with required interface
    class _Model:
        def __init__(self):
            self.config = type("C", (), {"mtp_num_layers": 0, "overlap_moe_expert_parallel_comm": True})()
            self._pg_collection = _MockPGCollection()

        @property
        def pg_collection(self):
            return self._pg_collection

        def build_schedule_plan(self, tokens, position_ids, attention_mask, labels=None, loss_mask=None):  # noqa: ARG002
            return torch.tensor(1)

        def __call__(self, **kwargs):  # noqa: ARG002
            return torch.tensor(0.0)

    # Return model config
    monkeypatch.setattr("megatron.core.utils.get_model_config", lambda m: m.config, raising=True)

    # Dummy timers/straggler_timer
    class _Timer:
        def __call__(self, *a, **k):  # noqa: ARG002
            return self

        def start(self):
            return self

        def stop(self):
            return self

    class _Strag:
        def __call__(self, *a, **k):  # noqa: ARG002
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: ARG002
            return False

    class _State:
        def __init__(self):
            self.cfg = type(
                "Cfg",
                (),
                {
                    "rerun_state_machine": type(
                        "R", (), {"check_for_nan_in_loss": False, "check_for_spiky_loss": False}
                    )()
                },
            )()  # noqa: E501
            self.timers = _Timer()
            self.straggler_timer = _Strag()

    # Reuse small iterator producing already-sized batch
    vi = Qwen2_5_VLVisualInputs(pixel_values=torch.randn(1, 1, 3, 4, 4), image_grid_thw=torch.tensor([[[1, 2, 2]]]))
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[2, 3, 4, -100]]),
        "loss_mask": torch.ones(1, 4),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "visual_inputs": vi,
    }
    it = _Iterator(batch)

    # Minimal cfg for get_batch within forward_step
    cfg = type(
        "C2",
        (),
        {
            "model": type("M", (), {"seq_length": 16, "pipeline_model_parallel_size": 1})(),
            "dataset": type("D", (), {"skip_getting_attention_mask_from_dataset": True})(),
            "rerun_state_machine": type("R", (), {"check_for_nan_in_loss": False, "check_for_spiky_loss": False})(),
        },
    )()  # noqa: E501

    state = _State()
    state.cfg = cfg
    model = _Model()

    # Execute schedule plan path
    plan, loss_fn = forward_step(state, it, model, return_schedule_plan=True)
    assert isinstance(plan, torch.Tensor)


class TestPackBatchSequences:
    """Tests for the pack_batch_sequences function."""

    def test_basic_packing(self):
        """Test basic sequence packing functionality."""
        batch_size, seq_len = 2, 8
        # Tokens with padding at the end (pad_token_id=0)
        tokens = torch.tensor(
            [
                [1, 2, 3, 0, 0, 0, 0, 0],  # length 3
                [4, 5, 6, 7, 0, 0, 0, 0],  # length 4
            ]
        )
        labels = torch.tensor(
            [
                [2, 3, -100, -100, -100, -100, -100, -100],
                [5, 6, 7, -100, -100, -100, -100, -100],
            ]
        )
        loss_mask = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        attention_mask = None

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, packed_labels, packed_loss_mask, packed_attn, packed_pos, cu_seqlens, max_seqlen = result

        # Packed output should have shape [1, total_valid_len]
        assert packed_tokens.shape[0] == 1
        total_len = packed_tokens.shape[1]
        assert total_len == 7  # 3 + 4

        # cu_seqlens should have num_sequences + 1 elements
        assert len(cu_seqlens) == 3  # [0, 3, 7]
        assert cu_seqlens[0] == 0
        assert cu_seqlens[1] == 3  # first sequence length
        assert cu_seqlens[2] == 7  # total length

        # max_seqlen should be max of sequence lengths
        assert max_seqlen.item() == 4

        # Attention mask should be None for packed sequences
        assert packed_attn is None

    def test_packing_with_pad_to_multiple_of(self):
        """Test packing with padding to a multiple (for CP compatibility)."""
        batch_size = 2
        tokens = torch.tensor(
            [
                [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],  # length 3 -> padded to 4 (mult of 2)
                [4, 5, 6, 7, 8, 0, 0, 0, 0, 0],  # length 5 -> padded to 6 (mult of 2)
            ]
        )
        labels = torch.tensor(
            [
                [2, 3, -100, -100, -100, -100, -100, -100, -100, -100],
                [5, 6, 7, 8, -100, -100, -100, -100, -100, -100],
            ]
        )
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.arange(10).unsqueeze(0).expand(batch_size, -1)

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=2,  # Pad each sequence to multiple of 2
        )

        packed_tokens, packed_labels, packed_loss_mask, packed_attn, packed_pos, cu_seqlens, max_seqlen = result

        # Total length should be 4 + 6 = 10 (padded lengths)
        assert packed_tokens.shape[1] == 10

        # cu_seqlens should use padded lengths
        assert cu_seqlens[0] == 0
        assert cu_seqlens[1] == 4  # 3 -> 4 (padded)
        assert cu_seqlens[2] == 10  # 5 -> 6, total = 4 + 6

        # max_seqlen should be 6 (longest padded sequence)
        assert max_seqlen.item() == 6

    def test_packing_with_larger_multiple(self):
        """Test packing with larger pad_to_multiple_of (e.g., for CP=4)."""
        tokens = torch.tensor(
            [
                [1, 2, 0, 0, 0, 0, 0, 0],  # length 2 -> padded to 4
                [3, 4, 5, 0, 0, 0, 0, 0],  # length 3 -> padded to 4
            ]
        )
        labels = torch.tensor(
            [
                [2, -100, -100, -100, -100, -100, -100, -100],
                [4, 5, -100, -100, -100, -100, -100, -100],
            ]
        )
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=4,
        )

        packed_tokens, *_, cu_seqlens, max_seqlen = result

        # Both sequences padded to 4, total = 8
        assert packed_tokens.shape[1] == 8
        assert cu_seqlens.tolist() == [0, 4, 8]
        assert max_seqlen.item() == 4

    def test_packing_single_sequence(self):
        """Test packing a single sequence."""
        tokens = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])  # length 5
        labels = torch.tensor([[2, 3, 4, 5, -100, -100, -100, -100]])
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.arange(8).unsqueeze(0)

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, *_, cu_seqlens, max_seqlen = result

        assert packed_tokens.shape[1] == 5
        assert cu_seqlens.tolist() == [0, 5]
        assert max_seqlen.item() == 5

    def test_packing_no_padding_sequences(self):
        """Test packing sequences with no padding."""
        tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ]
        )
        labels = torch.tensor(
            [
                [2, 3, 4, -100],
                [6, 7, 8, -100],
            ]
        )
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.arange(4).unsqueeze(0).expand(2, -1)

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, *_, cu_seqlens, max_seqlen = result

        # Both sequences full length
        assert packed_tokens.shape[1] == 8
        assert cu_seqlens.tolist() == [0, 4, 8]

    def test_packing_preserves_loss_mask_zeros(self):
        """Test that loss_mask zeros are preserved during packing."""
        tokens = torch.tensor([[1, 2, 3, 0, 0]])
        labels = torch.tensor([[2, 3, -100, -100, -100]])
        loss_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]])  # Second token masked
        position_ids = torch.arange(5).unsqueeze(0)

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, packed_labels, packed_loss_mask, *_ = result

        # Only first 3 tokens should be kept
        assert packed_loss_mask.shape[1] == 3
        assert packed_loss_mask[0, 0].item() == 1.0
        assert packed_loss_mask[0, 1].item() == 0.0  # Preserved
        assert packed_loss_mask[0, 2].item() == 1.0

    def test_packing_position_ids_reset(self):
        """Test that position_ids are correctly packed."""
        tokens = torch.tensor(
            [
                [1, 2, 0, 0],
                [3, 4, 5, 0],
            ]
        )
        labels = torch.zeros_like(tokens)
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ]
        )

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        _, _, _, _, packed_pos, *_ = result

        # Position IDs should be extracted from original sequences
        assert packed_pos.shape[1] == 5  # 2 + 3
        assert packed_pos[0, 0].item() == 0  # First seq, pos 0
        assert packed_pos[0, 1].item() == 1  # First seq, pos 1
        assert packed_pos[0, 2].item() == 0  # Second seq, pos 0
        assert packed_pos[0, 3].item() == 1  # Second seq, pos 1
        assert packed_pos[0, 4].item() == 2  # Second seq, pos 2

    def test_packing_empty_batch_warning(self, caplog):
        """Test that all-padding batch returns empty tensors with warning."""
        tokens = torch.tensor([[0, 0, 0, 0]])  # All padding
        labels = torch.tensor([[-100, -100, -100, -100]])
        loss_mask = torch.zeros(1, 4)
        position_ids = torch.arange(4).unsqueeze(0)

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, packed_labels, packed_loss_mask, packed_attn, packed_pos, cu_seqlens, max_seqlen = result

        # No valid sequences found, should return empty tensors
        assert packed_tokens.shape == (1, 0)
        assert packed_labels.shape == (1, 0)
        assert packed_loss_mask.shape == (1, 0)
        assert packed_pos.shape == (1, 0)
        # cu_seqlens should have just [0] for empty batch
        assert len(cu_seqlens) == 1
        assert cu_seqlens[0].item() == 0
        assert max_seqlen.item() == 0

    def test_packing_different_dtypes(self):
        """Test packing with different tensor dtypes."""
        tokens = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
        labels = torch.tensor([[2, 3, -100, -100]], dtype=torch.long)
        loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, packed_labels, packed_loss_mask, _, packed_pos, cu_seqlens, _ = result

        # Dtypes should be preserved
        assert packed_tokens.dtype == torch.long
        assert packed_labels.dtype == torch.long
        assert packed_loss_mask.dtype == torch.float32
        assert packed_pos.dtype == torch.long
        assert cu_seqlens.dtype == torch.int32

    def test_packing_padding_extends_position_ids(self):
        """Test that padding extends position_ids correctly."""
        tokens = torch.tensor([[1, 2, 3, 0]])  # length 3
        labels = torch.zeros_like(tokens)
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.tensor([[0, 1, 2, 3]])

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=4,  # Pad to 4
        )

        _, _, _, _, packed_pos, cu_seqlens, _ = result

        # Length should be 4 (padded)
        assert packed_pos.shape[1] == 4

        # Original positions should be preserved
        assert packed_pos[0, 0].item() == 0
        assert packed_pos[0, 1].item() == 1
        assert packed_pos[0, 2].item() == 2
        # Padding position should be extended
        assert packed_pos[0, 3].item() == 3

    def test_packing_cu_seqlens_dtype(self):
        """Test that cu_seqlens is int32 as expected by attention kernels."""
        tokens = torch.tensor([[1, 2, 0]])
        labels = torch.zeros_like(tokens)
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.arange(3).unsqueeze(0)

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
        )

        _, _, _, _, _, cu_seqlens, _ = result

        assert cu_seqlens.dtype == torch.int32

    def test_packing_gpu_tensor(self):
        """Test packing works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tokens = torch.tensor([[1, 2, 3, 0, 0]], device="cuda")
        labels = torch.tensor([[2, 3, -100, -100, -100]], device="cuda")
        loss_mask = torch.ones_like(tokens, dtype=torch.float, device="cuda")
        position_ids = torch.arange(5, device="cuda").unsqueeze(0)

        result = pack_batch_sequences(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
        )

        packed_tokens, _, _, _, _, cu_seqlens, _ = result

        assert packed_tokens.device.type == "cuda"
        assert cu_seqlens.device.type == "cuda"
