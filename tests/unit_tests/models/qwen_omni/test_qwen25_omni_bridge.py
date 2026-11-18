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

from megatron.bridge.models.conversion.model_bridge import HFSourcedWeightTuple, HFWeightTuple, MegatronModelBridge
from megatron.bridge.models.qwen_omni.qwen25_omni_bridge import Qwen25OmniBridge


def _pretrained(tensors: dict[str, torch.Tensor]) -> SimpleNamespace:
    class _Source:
        @staticmethod
        def get_all_keys() -> list[str]:
            return list(tensors)

    class _State:
        source = _Source()

        def __getitem__(self, name: str) -> torch.Tensor:
            return tensors[name]

    return SimpleNamespace(state=_State())


def _thinker_events(*_args, **_kwargs):
    yield HFWeightTuple("thinker.lm_head.weight", torch.tensor([1.0]))
    yield None


class TestQwen25OmniBridgeExportPassthrough:
    """Tests for Qwen25OmniBridge export of HF-only talker and token2wav weights."""

    def test_talker_and_token2wav_weights_are_copied_from_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Talker and token2wav tensors follow the thinker export, each as its own group."""
        talker = torch.tensor([2.0])
        token2wav = torch.tensor([3.0])
        tensors = {
            "thinker.lm_head.weight": torch.tensor([9.0]),
            "talker.codec_head.weight": talker,
            "token2wav.code2wav_dit_model.proj.weight": token2wav,
        }
        monkeypatch.setattr(MegatronModelBridge, "_stream_weight_events_megatron_to_hf", _thinker_events)

        groups = list(Qwen25OmniBridge()._stream_weight_groups_megatron_to_hf([], _pretrained(tensors)))

        assert [[item.param_name for item in group] for group in groups] == [
            ["thinker.lm_head.weight"],
            ["talker.codec_head.weight"],
            ["token2wav.code2wav_dit_model.proj.weight"],
        ]
        torch.testing.assert_close(groups[1][0].weight, talker, rtol=0, atol=0)
        torch.testing.assert_close(groups[2][0].weight, token2wav, rtol=0, atol=0)
        assert Qwen25OmniBridge.stream_weights_megatron_to_hf is MegatronModelBridge.stream_weights_megatron_to_hf

    def test_passthrough_weights_are_sourceless_with_megatron_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With megatron names requested, passthrough tensors carry no Megatron source."""
        tensors = {"talker.codec_head.weight": torch.tensor([2.0])}
        seen_kwargs: dict = {}

        def fake_events(*_args, **kwargs):
            seen_kwargs.update(kwargs)
            yield HFSourcedWeightTuple("thinker.lm_head.weight", torch.tensor([1.0]), ("output_layer.weight",))
            yield None

        monkeypatch.setattr(MegatronModelBridge, "_stream_weight_events_megatron_to_hf", fake_events)
        monkeypatch.setattr(Qwen25OmniBridge, "_should_emit_hf_passthrough", lambda *_args, **_kwargs: True)

        result = list(
            Qwen25OmniBridge().stream_weights_megatron_to_hf(
                [], _pretrained(tensors), with_megatron_names=True, current_pp_stage_only=True
            )
        )

        assert seen_kwargs["with_megatron_names"] is True
        assert seen_kwargs["current_pp_stage_only"] is True
        assert [type(item) for item in result] == [HFSourcedWeightTuple, HFSourcedWeightTuple]
        assert result[1].param_name == "talker.codec_head.weight"
        assert result[1].megatron_param_names == ()

    @pytest.mark.parametrize(
        "pretrained, emit",
        [
            (_pretrained({"talker.codec_head.weight": torch.tensor([2.0])}), False),
            (SimpleNamespace(state=None), True),
        ],
        ids=["non_owning_pipeline_stage", "no_source_checkpoint"],
    )
    def test_only_mapped_weights_without_passthrough(
        self, monkeypatch: pytest.MonkeyPatch, pretrained: SimpleNamespace, emit: bool
    ) -> None:
        """No passthrough is emitted on a non-owning pipeline stage or without a source checkpoint."""
        monkeypatch.setattr(MegatronModelBridge, "_stream_weight_events_megatron_to_hf", _thinker_events)
        monkeypatch.setattr(Qwen25OmniBridge, "_should_emit_hf_passthrough", lambda *_args, **_kwargs: emit)

        result = list(Qwen25OmniBridge().stream_weights_megatron_to_hf([], pretrained))

        assert [item.param_name for item in result] == ["thinker.lm_head.weight"]
