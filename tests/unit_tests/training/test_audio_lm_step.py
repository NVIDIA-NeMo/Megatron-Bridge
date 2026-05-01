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

import torch

from megatron.bridge.training.audio_lm_step import (
    forward_step,
    get_batch,
    get_batch_from_iterator,
)
from megatron.bridge.training.utils.visual_inputs import Qwen2AudioInputs


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


def _make_audio_batch(device="cpu"):
    """Create a minimal batch with audio_inputs instead of visual_inputs."""
    tokens = torch.tensor([[1, 2, 3]], device=device)
    input_ids = tokens.clone()
    position_ids = torch.tensor([[0, 1, 2]], device=device)
    labels = torch.tensor([[2, 3, 4]], device=device)
    loss_mask = torch.ones_like(labels, dtype=torch.float, device=device)
    attention_mask = torch.ones_like(tokens, dtype=torch.bool, device=device)

    # Audio inputs container
    input_features = torch.randn(1, 128, 80, device=device)
    feature_attention_mask = torch.ones(1, 128, dtype=torch.bool, device=device)
    ai = Qwen2AudioInputs(input_features=input_features, feature_attention_mask=feature_attention_mask)

    batch = {
        "tokens": tokens,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "audio_inputs": ai,
    }
    return batch


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
        self.tp = _MockProcessGroup()
        self._cp_size = cp_size

    @property
    def cp(self):
        pg = _MockProcessGroup()
        pg.size = lambda: self._cp_size
        return pg


def test_get_batch_from_iterator_moves_audio_inputs_to_cuda(monkeypatch):
    """Simulate CUDA move on CPU-only env by making .cuda a no-op."""

    class _NoCudaTensor(torch.Tensor):
        def cuda(self, non_blocking=False):  # type: ignore[override]
            return self

    def _as_nocuda(t):
        return t.as_subclass(_NoCudaTensor)

    batch = _make_audio_batch()
    # Replace tensors with _NoCudaTensor so calling .cuda works without a GPU
    for k in ["tokens", "input_ids", "position_ids", "labels", "loss_mask", "attention_mask"]:
        batch[k] = _as_nocuda(batch[k])
    ai = batch["audio_inputs"]
    ai.input_features = _as_nocuda(ai.input_features)
    ai.feature_attention_mask = _as_nocuda(ai.feature_attention_mask)

    it = _Iterator(batch)
    out = get_batch_from_iterator(
        it,
        use_mtp=False,
        skip_getting_attention_mask_from_dataset=True,
        is_first_pp_stage=True,
        is_last_pp_stage=True,
    )

    assert "audio_inputs" in out
    assert "visual_inputs" not in out
    out_ai = out["audio_inputs"]
    assert isinstance(out_ai, Qwen2AudioInputs)
    # Verify fields are preserved
    assert out_ai.input_features is not None and out_ai.feature_attention_mask is not None


def test_get_batch_padding_paths(monkeypatch):
    """Short tokens should be padded to ceil-128, capped at seq_length."""
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_first_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_last_stage", lambda pg: True, raising=True)
    monkeypatch.setattr(
        "megatron.core.utils.get_batch_on_this_cp_rank",
        lambda x: x,
        raising=True,
    )

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
    )()
    cfg.dataset = type("D", (), {"skip_getting_attention_mask_from_dataset": True})()

    # Make batch shorter than 128 to trigger ceil-to-128 padding path
    short_tokens = torch.tensor([[1, 2, 3, 4]])
    ai = Qwen2AudioInputs(
        input_features=torch.randn(1, 128, 80),
        feature_attention_mask=torch.ones(1, 128, dtype=torch.bool),
    )
    batch = {
        "input_ids": short_tokens,
        "labels": torch.tensor([[2, 3, 4, -100]]),
        "loss_mask": torch.ones_like(short_tokens, dtype=torch.float),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_mask": torch.ones_like(short_tokens, dtype=torch.bool),
        "audio_inputs": ai,
    }

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
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_first_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_last_stage", lambda pg: True, raising=True)
    monkeypatch.setattr(
        "megatron.core.utils.get_batch_on_this_cp_rank",
        lambda x: x,
        raising=True,
    )

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
            "pack_sequences_in_batch": True,
        },
    )()

    # Batch with 2 sequences of different lengths (with padding)
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

    ai = Qwen2AudioInputs(
        input_features=torch.randn(1, 128, 80),
        feature_attention_mask=torch.ones(1, 128, dtype=torch.bool),
    )
    batch = {
        "input_ids": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": None,
        "audio_inputs": ai,
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
        audio_inputs,
    ) = get_batch(it, cfg, use_mtp=False, pg_collection=_MockPGCollection())

    # Verify packing occurred — total packed length = 3 + 5 = 8
    assert out_tokens.shape == (1, 8), f"Expected packed shape (1, 8), got {out_tokens.shape}"
    assert out_labels.shape == (1, 8)
    assert out_loss_mask.shape == (1, 8)
    assert out_position_ids.shape == (1, 8)

    # Verify cu_seqlens is populated
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

    # Verify audio_inputs passed through
    assert audio_inputs is not None
    assert isinstance(audio_inputs, Qwen2AudioInputs)


def test_forward_step(monkeypatch):
    """Test forward_step returns (output_tensor, loss_function) tuple."""
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_first_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.pipeline_parallel.utils.is_pp_last_stage", lambda pg: True, raising=True)
    monkeypatch.setattr("megatron.core.utils.get_batch_on_this_cp_rank", lambda x: x, raising=True)
    monkeypatch.setattr("megatron.core.utils.get_model_config", lambda m: m.config, raising=True)

    class _Model:
        def __init__(self):
            self.config = type("C", (), {"mtp_num_layers": 0, "overlap_moe_expert_parallel_comm": True})()
            self._pg_collection = _MockPGCollection()

        @property
        def pg_collection(self):
            return self._pg_collection

        def __call__(self, **kwargs):  # noqa: ARG002
            return torch.tensor(0.0)

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
                "C2",
                (),
                {
                    "model": type("M", (), {"seq_length": 16, "pipeline_model_parallel_size": 1})(),
                    "dataset": type("D", (), {"skip_getting_attention_mask_from_dataset": True})(),
                    "rerun_state_machine": type(
                        "R", (), {"check_for_nan_in_loss": False, "check_for_spiky_loss": False}
                    )(),
                },
            )()
            self.timers = _Timer()
            self.straggler_timer = _Strag()

    ai = Qwen2AudioInputs(
        input_features=torch.randn(1, 128, 80),
        feature_attention_mask=torch.ones(1, 128, dtype=torch.bool),
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[2, 3, 4, -100]]),
        "loss_mask": torch.ones(1, 4),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "audio_inputs": ai,
    }
    it = _Iterator(batch)

    state = _State()
    model = _Model()

    output_tensor, loss_function = forward_step(state, it, model)
    assert isinstance(output_tensor, torch.Tensor)
    assert callable(loss_function)
