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

"""Unit tests for skipping source-only HF passthrough reads on non-writer ranks."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.models.conversion.model_bridge import (
    HFWeightTuple,
    MegatronModelBridge,
    suppress_hf_passthrough,
)
from megatron.bridge.training.checkpointing import _save_hf_weights


class _DummyPassthroughBridge(MegatronModelBridge):
    """Minimal bridge subclass exercising ``_should_emit_hf_passthrough`` during export."""

    _HF_PASSTHROUGH_PREFIXES = ("engram.", "dfly.")

    def provider_bridge(self, hf_pretrained):  # pragma: no cover - unused in export test
        return None

    def mapping_registry(self):  # pragma: no cover - unused in export test
        return None

    def stream_weights_megatron_to_hf(
        self,
        megatron_model,
        hf_pretrained,
        cpu=True,
        show_progress=True,
        conversion_tasks=None,
        merge_adapter_weights=True,
        weight_dtype=None,
        with_megatron_names=False,
        current_pp_stage_only=False,
    ):
        del megatron_model, cpu, show_progress, conversion_tasks, merge_adapter_weights, weight_dtype
        del with_megatron_names
        # Simulate collective mapped weight exported on every rank.
        yield HFWeightTuple("model.embed_tokens.weight", torch.tensor([1.0]))
        if not self._should_emit_hf_passthrough([], current_pp_stage_only=current_pp_stage_only):
            return
        state = hf_pretrained.state
        for key in state.source.get_all_keys():
            if key.startswith(self._HF_PASSTHROUGH_PREFIXES):
                yield HFWeightTuple(key, state[key])


def test_suppress_hf_passthrough_restores_context_on_error():
    bridge = _DummyPassthroughBridge()
    assert bridge._should_emit_hf_passthrough([], current_pp_stage_only=False) is True

    with pytest.raises(RuntimeError, match="boom"):
        with suppress_hf_passthrough():
            assert bridge._should_emit_hf_passthrough([], current_pp_stage_only=False) is False
            raise RuntimeError("boom")

    assert bridge._should_emit_hf_passthrough([], current_pp_stage_only=False) is True


@pytest.mark.parametrize("rank", [0, 1])
def test_save_hf_weights_skips_passthrough_reads_on_non_writer_ranks(monkeypatch, tmp_path, rank):
    read_keys: list[str] = []
    saved_keys: list[str] = []

    tensors = {
        "model.embed_tokens.weight": torch.tensor([1.0]),
        "engram.table.weight": torch.tensor([2.0]),
        "dfly.head.weight": torch.tensor([3.0]),
    }

    class _TrackingSource:
        @staticmethod
        def get_all_keys():
            return list(tensors)

        @staticmethod
        def save_generator(generator, iter_dir, strict=False):
            del iter_dir, strict
            for name, _tensor in generator:
                saved_keys.append(name)

    class _TrackingState:
        source = _TrackingSource()

        def __getitem__(self, key):
            read_keys.append(key)
            return tensors[key]

    hf_pretrained = SimpleNamespace(state=_TrackingState(), save_artifacts=MagicMock())
    model_bridge = _DummyPassthroughBridge()

    def _export_hf_weights(
        model,
        cpu=True,
        show_progress=True,
        conversion_tasks=None,
        merge_adapter_weights=True,
        weight_dtype=None,
        with_megatron_names=False,
        current_pp_stage_only=False,
    ):
        return model_bridge.stream_weights_megatron_to_hf(
            model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            merge_adapter_weights=merge_adapter_weights,
            weight_dtype=weight_dtype,
            with_megatron_names=with_megatron_names,
            current_pp_stage_only=current_pp_stage_only,
        )

    fake_bridge = SimpleNamespace(
        _model_bridge=model_bridge,
        hf_pretrained=hf_pretrained,
        export_hf_weights=_export_hf_weights,
    )

    state = SimpleNamespace(
        cfg=SimpleNamespace(
            peft=None,
            model=SimpleNamespace(),
            checkpoint=SimpleNamespace(hf_distributed_save=False, hf_save_every_n_ranks=1),
        )
    )

    with (
        patch("megatron.bridge.training.checkpointing._resolve_hf_source", return_value="/hf/src"),
        patch("megatron.bridge.training.checkpointing._build_auto_bridge_for_save", return_value=fake_bridge),
        patch("megatron.bridge.training.checkpointing.ensure_directory_exists"),
        patch("megatron.bridge.training.checkpointing.get_rank_safe", return_value=rank),
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.barrier"),
    ):
        _save_hf_weights(state, [], str(tmp_path))

    if rank == 0:
        assert read_keys == ["engram.table.weight", "dfly.head.weight"]
        assert saved_keys == ["model.embed_tokens.weight", "engram.table.weight", "dfly.head.weight"]
        hf_pretrained.save_artifacts.assert_called_once()
    else:
        assert read_keys == []
        assert saved_keys == []
        hf_pretrained.save_artifacts.assert_not_called()
