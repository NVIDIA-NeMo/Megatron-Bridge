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
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SwiGLU fc1 checkpoint layout: contiguous -> load (interleave) -> save (de-interleave) -> contiguous."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.bridge.training.checkpointing import (
    _get_model_glu_interleave_sizes,
    _interleave_glu_tensor,
    _process_state_dict_for_glu_interleaving,
    _process_state_dict_for_model_glu_interleaving,
)


_CKPT_MOD = "megatron.bridge.training.checkpointing"

MOE_FC1_KEY = "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight"
MOE_FC1_BIAS_KEY = "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.bias"
DENSE_FC1_KEY = "decoder.layers.0.mlp.linear_fc1.weight"
DENSE_FC1_BIAS_KEY = "decoder.layers.0.mlp.linear_fc1.bias"
SHARED_FC1_KEY = "decoder.layers.0.mlp.shared_experts.linear_fc1.weight"
SHARED_FC1_BIAS_KEY = "decoder.layers.0.mlp.shared_experts.linear_fc1.bias"


@pytest.fixture
def patch_print_rank_0():
    with patch(f"{_CKPT_MOD}.print_rank_0"):
        yield


# ---------------------------------------------------------------------------
# Lightweight DTensor fakes for the Megatron-FSDP code path.
# Simulates single-rank sharding where the local shard IS the full tensor.
# ---------------------------------------------------------------------------


class _ChunkMeta:
    """Mimics chunk metadata returned by DTensor internals."""

    __slots__ = ("offsets", "sizes")

    def __init__(self, offsets: tuple[int, ...], sizes: tuple[int, ...]):
        self.offsets = offsets
        self.sizes = sizes


class _FakeInnerTensor:
    """Stand-in for ``DTensor._local_tensor`` with ``__create_chunk_list__``."""

    def __init__(self, data: torch.Tensor):
        self._data = data

    def __create_chunk_list__(self):
        return [
            _ChunkMeta(
                # Trivial global DTensor chunk metadata, i.e. zero offsets.
                offsets=tuple(0 for _ in self._data.shape),
                sizes=tuple(self._data.shape),
            )
        ]


class _FakeDTensor:
    """Single-rank DTensor stub (local shard == full tensor)."""

    def __init__(self, data: torch.Tensor):
        self._local_tensor = _FakeInnerTensor(data)
        self.device_mesh = None
        self.placements = None
        self._shape = data.shape
        self._stride = data.stride()

    @property
    def shape(self):
        return self._shape

    def stride(self):
        return self._stride

    @staticmethod
    def from_local(local_tensor, *, device_mesh, placements, shape, stride):
        return _FakeDTensor(local_tensor)


def _mock_gather(dtensor):
    """Single-rank gather: local shard is the full tensor."""
    return SimpleNamespace(_local_tensor=dtensor._local_tensor._data)


class TestCheckpointLoadSaveRoundTrip:
    """Contiguous checkpoint layout -> load (interleave) -> save (de-interleave) matches original tensors."""

    def test_get_model_glu_interleave_sizes_uses_shared_model_config(self, monkeypatch):
        """Shared expert GLU interleave size is config-driven, independent from routed/dense size."""
        monkeypatch.delenv("USE_GROUPED_GEMM_FOR_SHARED_EXPERT", raising=False)
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                moe_mlp_glu_interleave_size=8,
                moe_shared_expert_glu_interleave_size=16,
                use_grouped_gemm_for_shared_expert=True,
            )
        )

        assert _get_model_glu_interleave_sizes([], cfg) == (8, 16)

    @pytest.mark.parametrize("interleave_size", [4, 8])
    def test_moe_state_dict_round_trip_recover_contiguous(self, interleave_size, patch_print_rank_0):
        """MoE fc1 weight + bias + unrelated tensor: full round-trip recovers originals."""
        w = torch.randn(2 * interleave_size * 4, 16)
        b = torch.randn(2 * interleave_size * 2)
        passthrough = torch.randn(3, 7)
        original = {
            MOE_FC1_KEY: w.clone(),
            MOE_FC1_BIAS_KEY: b.clone(),
            "decoder.layers.0.mlp.linear_fc2.weight": passthrough.clone(),
        }
        after_load = _process_state_dict_for_glu_interleaving(
            {k: v.clone() for k, v in original.items()}, interleave_size, interleave=True
        )
        after_save = _process_state_dict_for_glu_interleaving(after_load, interleave_size, interleave=False)
        assert torch.equal(after_save[MOE_FC1_KEY], original[MOE_FC1_KEY])
        assert torch.equal(after_save[MOE_FC1_BIAS_KEY], original[MOE_FC1_BIAS_KEY])
        assert torch.equal(
            after_save["decoder.layers.0.mlp.linear_fc2.weight"],
            original["decoder.layers.0.mlp.linear_fc2.weight"],
        )

    @pytest.mark.parametrize("interleave_size", [4, 8])
    def test_dense_state_dict_round_trip_with_fusion_env(self, interleave_size, monkeypatch, patch_print_rank_0):
        """Dense fc1 participates only with USE_ACT_FUSION_FOR_DENSE=1; round-trip recovers contiguous tensors."""
        monkeypatch.setenv("USE_ACT_FUSION_FOR_DENSE", "1")
        w = torch.randn(2 * interleave_size * 3, 8)
        b = torch.randn(2 * interleave_size * 5)
        original = {
            DENSE_FC1_KEY: w.clone(),
            DENSE_FC1_BIAS_KEY: b.clone(),
        }
        after_load = _process_state_dict_for_glu_interleaving(
            {k: v.clone() for k, v in original.items()}, interleave_size, interleave=True
        )
        after_save = _process_state_dict_for_glu_interleaving(after_load, interleave_size, interleave=False)
        assert torch.equal(after_save[DENSE_FC1_KEY], original[DENSE_FC1_KEY])
        assert torch.equal(after_save[DENSE_FC1_BIAS_KEY], original[DENSE_FC1_BIAS_KEY])

    def test_shared_expert_state_dict_round_trip_recover_contiguous(self, patch_print_rank_0):
        """Shared expert fc1 uses its own interleave size and leaves routed/dense tensors untouched."""
        shared_interleave_size = 32
        routed_interleave_size = 8
        shared_w = torch.randn(2 * shared_interleave_size * 2, 16)
        shared_b = torch.randn(2 * shared_interleave_size * 2)
        routed_w = torch.randn(2 * routed_interleave_size * 2, 16)
        dense_w = torch.randn(2 * shared_interleave_size * 2, 16)
        original = {
            SHARED_FC1_KEY: shared_w.clone(),
            SHARED_FC1_BIAS_KEY: shared_b.clone(),
            MOE_FC1_KEY: routed_w.clone(),
            DENSE_FC1_KEY: dense_w.clone(),
        }
        after_load = _process_state_dict_for_model_glu_interleaving(
            {k: v.clone() for k, v in original.items()},
            routed_interleave_size=None,
            shared_interleave_size=shared_interleave_size,
            interleave=True,
        )
        assert not torch.equal(after_load[SHARED_FC1_KEY], original[SHARED_FC1_KEY])
        assert torch.equal(after_load[MOE_FC1_KEY], original[MOE_FC1_KEY])
        assert torch.equal(after_load[DENSE_FC1_KEY], original[DENSE_FC1_KEY])

        after_save = _process_state_dict_for_model_glu_interleaving(
            after_load,
            routed_interleave_size=None,
            shared_interleave_size=shared_interleave_size,
            interleave=False,
        )
        assert torch.equal(after_save[SHARED_FC1_KEY], original[SHARED_FC1_KEY])
        assert torch.equal(after_save[SHARED_FC1_BIAS_KEY], original[SHARED_FC1_BIAS_KEY])
        assert torch.equal(after_save[MOE_FC1_KEY], original[MOE_FC1_KEY])
        assert torch.equal(after_save[DENSE_FC1_KEY], original[DENSE_FC1_KEY])

    def test_shared_expert_load_uses_configured_interleave_size(self, patch_print_rank_0):
        """Loading the same checkpoint swizzles shared expert fc1 only when shared interleaving is configured."""
        shared_interleave_size = 32
        shared_w = torch.arange(2 * shared_interleave_size * 2 * 3, dtype=torch.float32).reshape(
            2 * shared_interleave_size * 2, 3
        )
        shared_b = torch.arange(2 * shared_interleave_size * 2, dtype=torch.float32)
        checkpoint_state = {
            SHARED_FC1_KEY: shared_w.clone(),
            SHARED_FC1_BIAS_KEY: shared_b.clone(),
        }

        non_swizzled = _process_state_dict_for_model_glu_interleaving(
            {k: v.clone() for k, v in checkpoint_state.items()},
            routed_interleave_size=None,
            shared_interleave_size=None,
            interleave=True,
        )
        swizzled = _process_state_dict_for_model_glu_interleaving(
            {k: v.clone() for k, v in checkpoint_state.items()},
            routed_interleave_size=None,
            shared_interleave_size=shared_interleave_size,
            interleave=True,
        )

        assert torch.equal(non_swizzled[SHARED_FC1_KEY], shared_w)
        assert torch.equal(non_swizzled[SHARED_FC1_BIAS_KEY], shared_b)
        assert torch.equal(
            swizzled[SHARED_FC1_KEY],
            _interleave_glu_tensor(shared_w, shared_interleave_size),
        )
        assert torch.equal(
            swizzled[SHARED_FC1_BIAS_KEY],
            _interleave_glu_tensor(shared_b, shared_interleave_size),
        )

    def test_model_glu_interleaving_round_trip_with_different_routed_and_shared_sizes(
        self, monkeypatch, patch_print_rank_0
    ):
        """Routed/dense and shared fc1 tensors use independent interleave sizes in the two-pass path."""
        monkeypatch.setenv("USE_ACT_FUSION_FOR_DENSE", "1")
        routed_interleave_size = 8
        shared_interleave_size = 32
        routed_w = torch.arange(2 * routed_interleave_size * 2 * 3, dtype=torch.float32).reshape(
            2 * routed_interleave_size * 2, 3
        )
        routed_b = torch.arange(2 * routed_interleave_size * 2, dtype=torch.float32)
        dense_w = torch.arange(2 * routed_interleave_size * 3 * 5, dtype=torch.float32).reshape(
            2 * routed_interleave_size * 3, 5
        )
        dense_b = torch.arange(2 * routed_interleave_size * 3, dtype=torch.float32)
        shared_w = torch.arange(2 * shared_interleave_size * 2 * 7, dtype=torch.float32).reshape(
            2 * shared_interleave_size * 2, 7
        )
        shared_b = torch.arange(2 * shared_interleave_size * 2, dtype=torch.float32)
        passthrough = torch.arange(11, dtype=torch.float32)
        original = {
            MOE_FC1_KEY: routed_w.clone(),
            MOE_FC1_BIAS_KEY: routed_b.clone(),
            DENSE_FC1_KEY: dense_w.clone(),
            DENSE_FC1_BIAS_KEY: dense_b.clone(),
            SHARED_FC1_KEY: shared_w.clone(),
            SHARED_FC1_BIAS_KEY: shared_b.clone(),
            "decoder.layers.0.mlp.linear_fc2.weight": passthrough.clone(),
        }

        with patch(
            f"{_CKPT_MOD}._process_state_dict_for_glu_interleaving",
            wraps=_process_state_dict_for_glu_interleaving,
        ) as mock_process:
            after_load = _process_state_dict_for_model_glu_interleaving(
                {k: v.clone() for k, v in original.items()},
                routed_interleave_size=routed_interleave_size,
                shared_interleave_size=shared_interleave_size,
                interleave=True,
            )

        assert mock_process.call_count == 2
        assert mock_process.call_args_list[0].args[1] == routed_interleave_size
        assert mock_process.call_args_list[0].kwargs["include_routed_experts"] is True
        assert mock_process.call_args_list[0].kwargs["include_shared_experts"] is False
        assert mock_process.call_args_list[0].kwargs["include_dense"] is True
        assert mock_process.call_args_list[1].args[1] == shared_interleave_size
        assert mock_process.call_args_list[1].kwargs["include_routed_experts"] is False
        assert mock_process.call_args_list[1].kwargs["include_shared_experts"] is True
        assert mock_process.call_args_list[1].kwargs["include_dense"] is False

        assert torch.equal(
            after_load[MOE_FC1_KEY], _interleave_glu_tensor(original[MOE_FC1_KEY], routed_interleave_size)
        )
        assert torch.equal(
            after_load[MOE_FC1_BIAS_KEY],
            _interleave_glu_tensor(original[MOE_FC1_BIAS_KEY], routed_interleave_size),
        )
        assert torch.equal(
            after_load[DENSE_FC1_KEY],
            _interleave_glu_tensor(original[DENSE_FC1_KEY], routed_interleave_size),
        )
        assert torch.equal(
            after_load[DENSE_FC1_BIAS_KEY],
            _interleave_glu_tensor(original[DENSE_FC1_BIAS_KEY], routed_interleave_size),
        )
        assert torch.equal(
            after_load[SHARED_FC1_KEY],
            _interleave_glu_tensor(original[SHARED_FC1_KEY], shared_interleave_size),
        )
        assert torch.equal(
            after_load[SHARED_FC1_BIAS_KEY],
            _interleave_glu_tensor(original[SHARED_FC1_BIAS_KEY], shared_interleave_size),
        )
        assert torch.equal(after_load["decoder.layers.0.mlp.linear_fc2.weight"], passthrough)

        after_save = _process_state_dict_for_model_glu_interleaving(
            after_load,
            routed_interleave_size=routed_interleave_size,
            shared_interleave_size=shared_interleave_size,
            interleave=False,
        )
        for key, value in original.items():
            assert torch.equal(after_save[key], value)

    def test_model_glu_interleaving_round_trip_with_equal_routed_and_shared_sizes(
        self, monkeypatch, patch_print_rank_0
    ):
        """Equal routed/shared interleave sizes use the one-pass fast path."""
        monkeypatch.setenv("USE_ACT_FUSION_FOR_DENSE", "1")
        interleave_size = 8
        routed_w = torch.arange(2 * interleave_size * 2 * 3, dtype=torch.float32).reshape(2 * interleave_size * 2, 3)
        routed_b = torch.arange(2 * interleave_size * 2, dtype=torch.float32)
        dense_w = torch.arange(2 * interleave_size * 3 * 5, dtype=torch.float32).reshape(2 * interleave_size * 3, 5)
        dense_b = torch.arange(2 * interleave_size * 3, dtype=torch.float32)
        shared_w = torch.arange(2 * interleave_size * 4 * 7, dtype=torch.float32).reshape(2 * interleave_size * 4, 7)
        shared_b = torch.arange(2 * interleave_size * 4, dtype=torch.float32)
        original = {
            MOE_FC1_KEY: routed_w.clone(),
            MOE_FC1_BIAS_KEY: routed_b.clone(),
            DENSE_FC1_KEY: dense_w.clone(),
            DENSE_FC1_BIAS_KEY: dense_b.clone(),
            SHARED_FC1_KEY: shared_w.clone(),
            SHARED_FC1_BIAS_KEY: shared_b.clone(),
        }

        with patch(
            f"{_CKPT_MOD}._process_state_dict_for_glu_interleaving",
            wraps=_process_state_dict_for_glu_interleaving,
        ) as mock_process:
            after_load = _process_state_dict_for_model_glu_interleaving(
                {k: v.clone() for k, v in original.items()},
                routed_interleave_size=interleave_size,
                shared_interleave_size=interleave_size,
                interleave=True,
            )

        assert mock_process.call_count == 1
        assert mock_process.call_args.args[1] == interleave_size
        assert mock_process.call_args.kwargs["include_routed_experts"] is True
        assert mock_process.call_args.kwargs["include_shared_experts"] is True
        assert mock_process.call_args.kwargs["include_dense"] is True

        for key, value in original.items():
            assert torch.equal(after_load[key], _interleave_glu_tensor(value, interleave_size))

        after_save = _process_state_dict_for_model_glu_interleaving(
            after_load,
            routed_interleave_size=interleave_size,
            shared_interleave_size=interleave_size,
            interleave=False,
        )
        for key, value in original.items():
            assert torch.equal(after_save[key], value)


class TestMegatronFSDPCheckpointRoundTrip:
    """
    Megatron-FSDP DTensor path: contiguous → interleave → de-interleave recovers originals.

    NOTE(@cspades): These do NOT test DTensor or Megatron-FSDP un-even sharded gather.
    This only tests the non-distributed logic in _process_state_dict_for_glu_interleaving.
    """

    @pytest.fixture(autouse=True)
    def _fsdp_mocks(self, patch_print_rank_0):
        with (
            patch(f"{_CKPT_MOD}.preprocess_state_dict_for_uneven_dtensor", side_effect=lambda d: d),
            patch(f"{_CKPT_MOD}.gather_uneven_dtensor_to_full_tensor", side_effect=_mock_gather),
            patch(f"{_CKPT_MOD}.DTensor", _FakeDTensor),
        ):
            yield

    @pytest.mark.parametrize("interleave_size", [4, 8])
    def test_moe_fsdp_round_trip_recovers_contiguous(self, interleave_size):
        """MoE fc1 weight + bias + passthrough: FSDP DTensor round-trip recovers contiguous originals."""
        w = torch.randn(2 * interleave_size * 4, 16)
        b = torch.randn(2 * interleave_size * 2)
        passthrough = torch.randn(3, 7)
        original_w, original_b, original_pt = w.clone(), b.clone(), passthrough.clone()

        state = {
            MOE_FC1_KEY: _FakeDTensor(w),
            MOE_FC1_BIAS_KEY: _FakeDTensor(b),
            "decoder.layers.0.mlp.linear_fc2.weight": passthrough,
        }

        after_load = _process_state_dict_for_glu_interleaving(
            state,
            interleave_size,
            interleave=True,
            use_megatron_fsdp=True,
        )
        assert not torch.equal(after_load[MOE_FC1_KEY]._local_tensor._data, original_w)

        after_save = _process_state_dict_for_glu_interleaving(
            after_load,
            interleave_size,
            interleave=False,
            use_megatron_fsdp=True,
        )
        assert torch.equal(after_save[MOE_FC1_KEY]._local_tensor._data, original_w)
        assert torch.equal(after_save[MOE_FC1_BIAS_KEY]._local_tensor._data, original_b)
        assert torch.equal(
            after_save["decoder.layers.0.mlp.linear_fc2.weight"],
            original_pt,
        )

    @pytest.mark.parametrize("interleave_size", [4, 8])
    def test_dense_fsdp_round_trip_with_fusion_env(self, interleave_size, monkeypatch):
        """Dense fc1 with USE_ACT_FUSION_FOR_DENSE=1: FSDP DTensor round-trip recovers contiguous tensors."""
        monkeypatch.setenv("USE_ACT_FUSION_FOR_DENSE", "1")
        w = torch.randn(2 * interleave_size * 3, 8)
        b = torch.randn(2 * interleave_size * 5)
        original_w, original_b = w.clone(), b.clone()

        state = {
            DENSE_FC1_KEY: _FakeDTensor(w),
            DENSE_FC1_BIAS_KEY: _FakeDTensor(b),
        }

        after_load = _process_state_dict_for_glu_interleaving(
            state,
            interleave_size,
            interleave=True,
            use_megatron_fsdp=True,
        )
        after_save = _process_state_dict_for_glu_interleaving(
            after_load,
            interleave_size,
            interleave=False,
            use_megatron_fsdp=True,
        )
        assert torch.equal(after_save[DENSE_FC1_KEY]._local_tensor._data, original_w)
        assert torch.equal(after_save[DENSE_FC1_BIAS_KEY]._local_tensor._data, original_b)
