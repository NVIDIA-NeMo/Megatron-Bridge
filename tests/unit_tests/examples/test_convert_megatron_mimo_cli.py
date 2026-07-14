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

"""Unit tests for the generic ``examples/conversion/convert_megatron_mimo.py`` CLI parser.

Covers only the model-agnostic ``--component name=tp=N,...`` parsing logic.
The end-to-end conversion path needs GPUs and torchrun.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CLI_PATH = _REPO_ROOT / "examples" / "conversion" / "convert_megatron_mimo.py"


@pytest.fixture(scope="module")
def cli():
    """Load the CLI as a module under a stable name without polluting sys.modules permanently."""
    spec = importlib.util.spec_from_file_location("convert_megatron_mimo_cli_under_test", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


class TestParseComponentSpec:
    def test_minimal_tp_only(self, cli):
        name, parallelism = cli._parse_component_spec("language=tp=2")
        assert name == "language"
        assert parallelism.tensor_model_parallel_size == 2
        # Other dims should keep their dataclass defaults.
        assert parallelism.pipeline_model_parallel_size == 1
        assert parallelism.expert_model_parallel_size == 1
        assert parallelism.data_parallel_size is None

    def test_all_dims(self, cli):
        name, parallelism = cli._parse_component_spec("vision=tp=2,pp=2,dp=4,cp=1,ep=3,etp=1,rank_offset=4")
        assert name == "vision"
        assert parallelism.tensor_model_parallel_size == 2
        assert parallelism.pipeline_model_parallel_size == 2
        assert parallelism.data_parallel_size == 4
        assert parallelism.context_parallel_size == 1
        assert parallelism.expert_model_parallel_size == 3
        assert parallelism.expert_tensor_parallel_size == 1
        assert parallelism.rank_offset == 4


class TestBuildParallelismConfig:
    def test_multiple_components_no_world_size(self, cli):
        config = cli._build_parallelism_config(["language=tp=2", "vision=tp=1,dp=2"])
        assert set(config.module_parallelisms) == {"language", "vision"}
        assert config.module_parallelisms["language"].tensor_model_parallel_size == 2
        # No world_size → no auto-fill; user-supplied dp stays.
        assert config.module_parallelisms["language"].data_parallel_size is None
        assert config.module_parallelisms["vision"].data_parallel_size == 2

    def test_auto_fill_layout_two_components_two_ranks(self, cli):
        """world_size=2, two tp=1 components → each gets dp=1 and sequential rank_offset."""
        config = cli._build_parallelism_config(["language=tp=1", "vision=tp=1"], world_size=2)
        lang = config.module_parallelisms["language"]
        vis = config.module_parallelisms["vision"]
        assert lang.data_parallel_size == 1
        assert lang.rank_offset == 0
        assert vis.data_parallel_size == 1
        assert vis.rank_offset == 1

    def test_auto_fill_respects_user_dp(self, cli):
        """Components with explicit dp= are left alone; auto-fill skips them."""
        config = cli._build_parallelism_config(["language=tp=1,dp=1", "vision=tp=1"], world_size=2)
        lang = config.module_parallelisms["language"]
        vis = config.module_parallelisms["vision"]
        # User-supplied: untouched.
        assert lang.data_parallel_size == 1
        assert lang.rank_offset == 0
        # Auto-filled: offset advances past the user's component.
        assert vis.data_parallel_size == 1
        assert vis.rank_offset == 1

    def test_auto_fill_uses_dense_model_parallel_size(self, cli):
        """EP/ETP refactor the same ranks and do not multiply the dense rank span."""
        config = cli._build_parallelism_config(["language=tp=2,ep=2,etp=2", "vision=tp=1"], world_size=4)
        lang = config.module_parallelisms["language"]
        vis = config.module_parallelisms["vision"]

        assert lang.data_parallel_size == 1
        assert lang.rank_offset == 0
        assert vis.data_parallel_size == 2
        assert vis.rank_offset == 2

    def test_explicit_non_uniform_layout_skips_auto_fill(self, cli):
        """Explicit ``dp`` + ``rank_offset`` layouts may use uneven rank counts."""
        config = cli._build_parallelism_config(
            ["language=tp=4,dp=1,rank_offset=0", "images=tp=1,dp=1,rank_offset=4"],
            world_size=5,
        )
        lang = config.module_parallelisms["language"]
        images = config.module_parallelisms["images"]

        assert lang.data_parallel_size == 1
        assert lang.rank_offset == 0
        assert images.data_parallel_size == 1
        assert images.rank_offset == 4

    def test_auto_fill_rejects_world_size_not_divisible(self, cli):
        with pytest.raises(ValueError, match="not divisible by number of components"):
            cli._build_parallelism_config(["language=tp=1", "vision=tp=1", "audio=tp=1"], world_size=2)

    def test_auto_fill_rejects_ranks_not_divisible_by_mp(self, cli):
        # 2 ranks / 2 components = 1 rank each, but language wants tp=2 → ranks_per_component=1 not divisible by 2.
        with pytest.raises(ValueError, match="not divisible by total_model_parallel_size"):
            cli._build_parallelism_config(["language=tp=2", "vision=tp=1"], world_size=2)

    def test_rejects_empty_components(self, cli):
        with pytest.raises(ValueError, match="At least one --component flag is required"):
            cli._build_parallelism_config([])

    def test_rejects_duplicate_name(self, cli):
        with pytest.raises(ValueError, match="specified more than once"):
            cli._build_parallelism_config(["language=tp=1", "language=tp=2"])


class TestRunImportExport:
    def test_run_import_uses_megatron_mimo_bridge(self, cli, monkeypatch):
        calls = []

        class FakeBridge:
            _model_bridge = object()
            routes = []

            @classmethod
            def from_hf_pretrained(cls, *args, **kwargs):
                calls.append(("from_hf_pretrained", args, kwargs))
                return cls()

            def import_ckpt(self, *args, **kwargs):
                calls.append(("import_ckpt", args, kwargs))

        monkeypatch.setattr(cli, "MegatronMIMOBridge", FakeBridge)
        monkeypatch.setattr(cli.dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(cli.dist, "get_rank", lambda: 0)

        cli.run_import(
            hf_model="hf",
            component_specs=["language=tp=1", "images=tp=1"],
            trust_remote_code=True,
            torch_dtype=cli.torch.bfloat16,
            megatron_path="/ckpt",
        )

        assert calls[0][0] == "from_hf_pretrained"
        assert calls[0][1] == ("hf",)
        assert calls[0][2]["trust_remote_code"] is True
        assert calls[0][2]["torch_dtype"] is cli.torch.bfloat16
        assert set(calls[0][2]["parallelism_config"].module_parallelisms) == {"language", "images"}
        assert calls[1] == (
            "import_ckpt",
            ("/ckpt",),
            {"hf_tokenizer_path": "hf", "hf_tokenizer_kwargs": {"trust_remote_code": True}},
        )

    def test_run_export_uses_megatron_mimo_bridge(self, cli, monkeypatch):
        calls = []

        class FakeBridge:
            @classmethod
            def from_hf_pretrained(cls, *args, **kwargs):
                calls.append(("from_hf_pretrained", args, kwargs))
                return cls()

            def export_ckpt(self, *args, **kwargs):
                calls.append(("export_ckpt", args, kwargs))

        monkeypatch.setattr(cli, "MegatronMIMOBridge", FakeBridge)
        monkeypatch.setattr(cli.dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(cli.dist, "get_rank", lambda: 0)

        cli.run_export(
            hf_model="hf",
            component_specs=["language=tp=1", "images=tp=1"],
            trust_remote_code=False,
            torch_dtype=cli.torch.bfloat16,
            hf_path="/hf",
            megatron_path="/ckpt",
        )

        assert calls[0][0] == "from_hf_pretrained"
        assert calls[0][1] == ("hf",)
        assert calls[0][2]["trust_remote_code"] is False
        assert set(calls[0][2]["parallelism_config"].module_parallelisms) == {"language", "images"}
        assert calls[1] == (
            "export_ckpt",
            (),
            {"megatron_path": "/ckpt", "hf_path": "/hf", "show_progress": True},
        )
