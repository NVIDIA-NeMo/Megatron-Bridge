# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from megatron.bridge.models.megatron_mimo.conversion import MIMOComponent
from megatron.bridge.models.megatron_mimo.conversion import orchestrator as hf_io


pytestmark = pytest.mark.unit


class _RankedGroup:
    def __init__(self, rank: int):
        self._rank = rank

    def rank(self) -> int:
        return self._rank


class _PgCollection:
    def __init__(self, *, tp: int = 0, pp: int = 0, cp: int = 0, dp: int = 0):
        self.tp = _RankedGroup(tp)
        self.pp = _RankedGroup(pp)
        self.cp = _RankedGroup(cp)
        self.dp = _RankedGroup(dp)


def _routes() -> list[MIMOComponent]:
    return [
        MIMOComponent("language", "language_model.", "language_model"),
        MIMOComponent("images", "vision_model.", "modality_submodules.images.encoders.qwen_visual"),
    ]


def _install_fake_dist(monkeypatch, *, rank: int, world_size: int = 8, remote_chunks_by_call: list | None = None):
    gather_calls = []
    remote_chunks_by_call = list(remote_chunks_by_call or [])

    monkeypatch.setattr(hf_io.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(hf_io.dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(hf_io.dist, "get_world_size", lambda: world_size)

    def gather_object(obj, gathered, dst):
        assert dst == 0
        gather_calls.append(obj)
        remote_chunks = remote_chunks_by_call.pop(0) if remote_chunks_by_call else []
        if gathered is not None:
            gathered[:] = [None] * world_size
            for idx, chunk in enumerate(remote_chunks, start=1):
                gathered[idx] = chunk

    monkeypatch.setattr(hf_io.dist, "gather_object", gather_object)
    return gather_calls


def _install_fake_export(monkeypatch):
    export_calls = []
    route_tensors = {
        "language": [("model.embed_tokens.weight", torch.tensor([1.0]))],
        "images": [("visual.patch_embed.proj.weight", torch.tensor([2.0]))],
    }

    def fake_export_megatron_mimo_to_hf(**kwargs):
        route = kwargs["routes"][0]
        export_calls.append(
            {
                "route": route.name,
                "cpu": kwargs["cpu"],
                "show_progress": kwargs["show_progress"],
            }
        )
        yield from route_tensors[route.name]

    monkeypatch.setattr(hf_io, "export_megatron_mimo_to_hf", fake_export_megatron_mimo_to_hf)
    return export_calls


class TestStreamMimoWeightsToRank0:
    def test_rank0_streams_owned_route_and_yields_remote_representative_route(self, monkeypatch):
        remote_image_chunk = [("visual.patch_embed.proj.weight", torch.tensor([2.0]))]
        gather_calls = _install_fake_dist(
            monkeypatch,
            rank=0,
            remote_chunks_by_call=[
                [],
                [remote_image_chunk],
            ],
        )
        export_calls = _install_fake_export(monkeypatch)

        emitted = list(
            hf_io._stream_mimo_weights_to_rank0(
                source_bridge=object(),
                hf_pretrained=object(),
                mimo_model=object(),
                routes=_routes(),
                pg_collections={"language": _PgCollection(), "images": None},
                show_progress=True,
            )
        )

        assert [name for name, _ in emitted] == [
            "model.embed_tokens.weight",
            "visual.patch_embed.proj.weight",
        ]
        assert export_calls == [{"route": "language", "cpu": True, "show_progress": True}]
        assert gather_calls == [None, None]

    def test_non_rank0_representative_drains_and_gathers_route_without_yielding(self, monkeypatch):
        gather_calls = _install_fake_dist(monkeypatch, rank=4)
        export_calls = _install_fake_export(monkeypatch)

        emitted = list(
            hf_io._stream_mimo_weights_to_rank0(
                source_bridge=object(),
                hf_pretrained=object(),
                mimo_model=object(),
                routes=_routes(),
                pg_collections={"language": None, "images": _PgCollection()},
                show_progress=True,
            )
        )

        assert emitted == []
        assert export_calls == [{"route": "images", "cpu": True, "show_progress": False}]
        assert gather_calls[0] is None
        assert [name for name, _ in gather_calls[1]] == ["visual.patch_embed.proj.weight"]

    def test_non_representative_rank_drains_route_without_retaining_tensors(self, monkeypatch):
        gather_calls = _install_fake_dist(monkeypatch, rank=5)
        export_calls = _install_fake_export(monkeypatch)

        emitted = list(
            hf_io._stream_mimo_weights_to_rank0(
                source_bridge=object(),
                hf_pretrained=object(),
                mimo_model=object(),
                routes=_routes(),
                pg_collections={"language": None, "images": _PgCollection(tp=1)},
                show_progress=True,
            )
        )

        assert emitted == []
        assert export_calls == [{"route": "images", "cpu": False, "show_progress": False}]
        assert gather_calls == [None, None]


class TestIsComponentExportRepresentative:
    @pytest.mark.parametrize(
        "ranks, expected",
        [
            ({}, True),
            ({"tp": 1}, False),
            ({"pp": 1}, False),
            ({"cp": 1}, False),
            ({"dp": 1}, False),
        ],
    )
    def test_requires_zero_rank_in_all_component_groups(self, ranks, expected):
        pg = _PgCollection(**ranks)

        assert hf_io._is_component_export_representative(pg) is expected
