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

import ast
import importlib.util
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from megatron.bridge.utils.gdn_utils import (
    CUDNN_FRONTEND_GDN_NAN_FREE_VERSION,
    CUTLASS_DSL_GDN_MIN_VERSION,
    TRANSFORMER_ENGINE_GDN_FP8_VERSION,
    TRANSFORMER_ENGINE_GDN_MIN_VERSION,
    _release_tuple,
    _version_str,
    cudnn_gdn_stack_issues,
    is_te_gdn_available,
    probe_cudnn_gdn_stack_issues,
    te_gdn_fp8_issues,
    validate_cudnn_gdn_stack,
)
from megatron.bridge.utils.import_utils import get_distribution_version


@pytest.mark.unit
class TestCudnnGdnStackValidation:
    """Tests for validate_cudnn_gdn_stack and cudnn_gdn_stack_issues."""

    _COMPLETE = {
        "have_te_gdn": True,
        "cudnn_frontend_version": "1.29.0",
        "cutlass_importable": True,
        "cutlass_dsl": ("nvidia-cutlass-dsl", "4.8.0"),
    }
    _DSL_FLOOR = _version_str(CUTLASS_DSL_GDN_MIN_VERSION)

    def test_complete_stack_reports_nothing(self):
        assert cudnn_gdn_stack_issues(**self._COMPLETE) == []

    @pytest.mark.parametrize(
        ("gap", "fragment"),
        [
            pytest.param({"have_te_gdn": False}, "no GatedDeltaNetAttention", id="te-missing"),
            pytest.param({"cudnn_frontend_version": None}, "nvidia-cudnn-frontend is not installed", id="fe-missing"),
            pytest.param({"cudnn_frontend_version": "1.26.0"}, "has no cudnn.linear_attention", id="fe-1.26"),
            pytest.param({"cudnn_frontend_version": "1.28.0"}, "can return NaN", id="fe-1.28"),
            pytest.param({"cutlass_importable": False}, "is not importable", id="cutlass-missing"),
            pytest.param({"cutlass_dsl": ("nvidia-cutlass-dsl", "4.5.0")}, "nvidia-cutlass-dsl 4.5.0", id="dsl-4.5.0"),
            pytest.param({"cutlass_dsl": ("nvidia-cutlass-dsl", "4.6")}, "nvidia-cutlass-dsl 4.6 ", id="dsl-4.6"),
            pytest.param({"cutlass_dsl": ("nvidia-cutlass-dsl", "4.6.2rc1")}, "4.6.2rc1", id="dsl-4.6.2rc1"),
        ],
    )
    def test_each_gap_is_reported(self, gap, fragment):
        issues = cudnn_gdn_stack_issues(**{**self._COMPLETE, **gap})
        assert len(issues) == 1
        assert fragment in issues[0]

    @pytest.mark.parametrize("frontend", ["1.26.0", None, "1.28.0"])
    @pytest.mark.parametrize(
        "dsl_gap",
        [
            pytest.param({"cutlass_dsl": ("nvidia-cutlass-dsl", "4.5.0")}, id="dsl-4.5.0"),
            pytest.param({"cutlass_importable": False}, id="dsl-missing"),
        ],
    )
    def test_an_unusable_frontend_hides_the_cute_dsl_gate(self, frontend, dsl_gap):
        """Below cuDNN frontend 1.29 only the frontend gap is reported.

        Frontends 1.27 and 1.28 gate the CuTe DSL differently, and the install hint already names its floor.
        """
        issues = cudnn_gdn_stack_issues(**{**self._COMPLETE, "cudnn_frontend_version": frontend, **dsl_gap})
        assert len(issues) == 1
        assert "cuTile" not in issues[0]

    @pytest.mark.parametrize(
        "cutlass_dsl",
        [
            ("nvidia-cutlass-dsl", "4.7.0"),
            ("nvidia-cutlass-dsl", "4.7.0.dev0"),
            ("nvidia-cutlass-dsl", "4.8.0a0+local"),
            ("nvidia-cutlass-dsl", "dev"),
            ("nvidia-cutlass-dsl-internal", "0.3.0+2026"),
            None,
        ],
    )
    def test_cutlass_dsl_versions_cudnn_frontend_accepts(self, cutlass_dsl):
        """Mirror ``cudnn.frost.buffers.cutedsl_too_old`` (cuDNN frontend 1.29.0).

        Internal builds, unparsable versions and missing metadata are never too old.
        """
        assert cudnn_gdn_stack_issues(**{**self._COMPLETE, "cutlass_dsl": cutlass_dsl}) == []

    def test_unparsable_frontend_version_is_not_flagged(self):
        assert cudnn_gdn_stack_issues(**{**self._COMPLETE, "cudnn_frontend_version": "unknown"}) == []

    @pytest.mark.parametrize("version", ["2.19.0", "2.19.0+5e52befd", "2.19.1", "2.19"])
    @pytest.mark.parametrize(
        ("fp8", "fp4", "fragment"),
        [
            pytest.param(True, False, "training fails in the first forward pass", id="fp8"),
            pytest.param(True, True, "training fails in the first forward pass", id="fp8-and-fp4"),
            pytest.param(False, True, "model.fp4 runs GatedDeltaNet under it in Megatron-Core's", id="fp4-only"),
        ],
    )
    def test_te219_under_fp8_or_fp4_is_reported(self, version, fp8, fp4, fragment):
        """Transformer Engine 2.19's GatedDeltaNetAttention raises under the FP8 autocast that fp8 or fp4 enables."""
        issues = te_gdn_fp8_issues(transformer_engine_version=version, fp8=fp8, fp4=fp4)
        assert len(issues) == 1
        assert f"transformer-engine {version}'s GatedDeltaNetAttention raises under FP8 autocast" in issues[0]
        assert fragment in issues[0]
        assert f"transformer-engine {_version_str(TRANSFORMER_ENGINE_GDN_FP8_VERSION)} and later" in issues[0]

    @pytest.mark.parametrize(
        ("version", "fp8", "fp4"),
        [
            pytest.param("2.19.0", False, False, id="te-2.19-bf16"),
            pytest.param("2.20.0", True, False, id="te-2.20.0-fp8"),
            pytest.param("2.20.1", True, False, id="te-2.20.1-fp8"),
            pytest.param("2.20.2+6ea2a74a", True, True, id="te-2.20.2-fp8-fp4"),
            pytest.param("2.21.0", False, True, id="te-2.21-fp4"),
            # No GatedDeltaNetAttention before 2.19; cudnn_gdn_stack_issues reports that case.
            pytest.param("2.18.0", True, False, id="te-2.18-fp8"),
            pytest.param("dev", True, False, id="unparsable"),
            pytest.param(None, True, False, id="te-missing"),
        ],
    )
    def test_te_fp8_combinations_that_are_not_flagged(self, version, fp8, fp4):
        """Only the 2.19 family is flagged, not every release below 2.20.2."""
        assert te_gdn_fp8_issues(transformer_engine_version=version, fp8=fp8, fp4=fp4) == []

    @pytest.mark.parametrize(
        ("versions", "fragment"),
        [
            pytest.param({"nvidia-cutlass-dsl": "4.5.0"}, "nvidia-cutlass-dsl 4.5.0", id="public-4.5.0"),
            pytest.param({"nvidia-cutlass-dsl-internal": "0.3.0+2026"}, None, id="internal-only"),
            pytest.param({}, None, id="no-dsl-metadata"),
        ],
    )
    def test_probe_reads_this_environment(self, monkeypatch, versions, fragment):
        """The probe feeds installed distribution versions and the import checks into the stack check."""
        versions = {"nvidia-cudnn-frontend": "1.29.0", **versions}
        monkeypatch.setattr("megatron.bridge.utils.gdn_utils.is_te_gdn_available", lambda: True)
        monkeypatch.setattr("megatron.bridge.utils.gdn_utils.is_module_available", lambda name: name == "cutlass")
        monkeypatch.setattr("megatron.bridge.utils.gdn_utils.get_distribution_version", versions.get)

        issues = probe_cudnn_gdn_stack_issues()

        if fragment is None:
            assert issues == []
        else:
            assert len(issues) == 1
            assert fragment in issues[0]

    def test_probe_runs_unpatched(self):
        """The real probe returns a list of messages, whatever this environment has installed."""
        issues = probe_cudnn_gdn_stack_issues()
        assert isinstance(issues, list)
        assert all(isinstance(issue, str) for issue in issues)

    def test_is_te_gdn_available_matches_megatron_core(self):
        """is_te_gdn_available reads HAVE_TE_GDN from the Megatron-Core module that defines it.

        Once Megatron-Core has ``TransformerConfig.gdn_kernel_backend``, a moved or renamed ``HAVE_TE_GDN`` would make
        every Transformer Engine GDN run warn falsely, so it must be found where ``is_te_gdn_available`` looks.
        """
        from megatron.core.transformer.transformer_config import TransformerConfig

        if "gdn_kernel_backend" not in {field.name for field in fields(TransformerConfig)}:
            assert is_te_gdn_available() is False
            pytest.skip("Megatron-Core predates TransformerConfig.gdn_kernel_backend (NVIDIA/Megatron-LM#6645)")
        from megatron.core.extensions import transformer_engine as mcore_te

        assert hasattr(mcore_te, "HAVE_TE_GDN"), "Megatron-Core moved HAVE_TE_GDN; update is_te_gdn_available"
        assert is_te_gdn_available() is bool(mcore_te.HAVE_TE_GDN)

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(SimpleNamespace(), id="no-field"),
            pytest.param(SimpleNamespace(gdn_kernel_backend="fla"), id="fla"),
            pytest.param(SimpleNamespace(gdn_kernel_backend="torch"), id="torch"),
        ],
    )
    @patch("megatron.bridge.utils.gdn_utils.probe_cudnn_gdn_stack_issues")
    @patch("megatron.bridge.utils.gdn_utils.warn_rank_0")
    def test_inert_unless_the_cudnn_backend_is_selected(self, mock_warn, mock_probe, model):
        validate_cudnn_gdn_stack(model)
        mock_probe.assert_not_called()
        mock_warn.assert_not_called()

    @patch("megatron.bridge.utils.gdn_utils.probe_cudnn_gdn_stack_issues", return_value=["ISSUE-A", "ISSUE-B"])
    @patch("megatron.bridge.utils.gdn_utils.warn_rank_0")
    def test_warns_with_the_fix_and_the_fla_override(self, mock_warn, mock_probe):
        model = SimpleNamespace(gdn_kernel_backend="transformer_engine")
        validate_cudnn_gdn_stack(model)
        mock_warn.assert_called_once()
        message = mock_warn.call_args[0][0]
        assert "ISSUE-A; ISSUE-B" in message
        assert f"transformer-engine>={_version_str(TRANSFORMER_ENGINE_GDN_MIN_VERSION)}," in message
        assert f"nvidia-cudnn-frontend>={_version_str(CUDNN_FRONTEND_GDN_NAN_FREE_VERSION)}" in message
        assert f"nvidia-cutlass-dsl[cu13]>={self._DSL_FLOOR}" in message
        assert "model.gdn_kernel_backend=fla" in message
        assert model.gdn_kernel_backend == "transformer_engine"  # never switched

    @patch("megatron.bridge.utils.gdn_utils.probe_cudnn_gdn_stack_issues", return_value=[])
    @patch("megatron.bridge.utils.gdn_utils.warn_rank_0")
    def test_complete_stack_is_silent(self, mock_warn, mock_probe):
        validate_cudnn_gdn_stack(SimpleNamespace(gdn_kernel_backend="transformer_engine"))
        mock_probe.assert_called_once()
        mock_warn.assert_not_called()

    @pytest.mark.parametrize(
        ("quantization", "fragment"),
        [
            pytest.param({"fp8": "e4m3"}, "training fails in the first forward pass", id="fp8-e4m3"),
            pytest.param({"fp8": "hybrid"}, "training fails in the first forward pass", id="fp8-hybrid"),
            pytest.param({"fp4": "e2m1"}, "model.fp4 runs GatedDeltaNet under it", id="fp4-e2m1"),
        ],
    )
    @patch("megatron.bridge.utils.gdn_utils.get_distribution_version", return_value="2.19.0")
    @patch("megatron.bridge.utils.gdn_utils.probe_cudnn_gdn_stack_issues", side_effect=lambda: [])
    @patch("megatron.bridge.utils.gdn_utils.warn_rank_0")
    def test_cudnn_gdn_warns_for_te219_fp8(self, mock_warn, mock_probe, mock_version, quantization, fragment):
        """A complete cuDNN stack still warns when Transformer Engine 2.19 meets an FP8 or FP4 recipe."""
        model = SimpleNamespace(gdn_kernel_backend="transformer_engine", **quantization)
        validate_cudnn_gdn_stack(model)
        mock_version.assert_called_once_with("transformer-engine")
        mock_warn.assert_called_once()
        message = mock_warn.call_args[0][0]
        assert message.count("transformer-engine 2.19.0's GatedDeltaNetAttention raises under FP8 autocast") == 1
        assert fragment in message
        assert f"transformer-engine>={_version_str(TRANSFORMER_ENGINE_GDN_FP8_VERSION)}," in message
        assert "model.gdn_kernel_backend=fla" in message
        assert vars(model) == {
            "gdn_kernel_backend": "transformer_engine",
            **quantization,
        }  # backend and precision kept

    @patch("megatron.bridge.utils.gdn_utils.get_distribution_version", return_value="2.19.0")
    @patch("megatron.bridge.utils.gdn_utils.probe_cudnn_gdn_stack_issues", side_effect=lambda: [])
    @patch("megatron.bridge.utils.gdn_utils.warn_rank_0")
    def test_te219_without_fp8_or_fp4_is_silent(self, mock_warn, mock_probe, mock_version):
        validate_cudnn_gdn_stack(SimpleNamespace(gdn_kernel_backend="transformer_engine", fp8=None, fp4=None))
        mock_version.assert_not_called()
        mock_warn.assert_not_called()

    @patch("megatron.bridge.utils.gdn_utils.get_distribution_version", return_value="2.20.2+6ea2a74a")
    @patch("megatron.bridge.utils.gdn_utils.probe_cudnn_gdn_stack_issues", side_effect=lambda: [])
    @patch("megatron.bridge.utils.gdn_utils.warn_rank_0")
    def test_fp8_with_a_fixed_te_is_silent(self, mock_warn, mock_probe, mock_version):
        """The FP8-CS and MXFP8 recipes on a complete stack with Transformer Engine 2.20.2 do not warn."""
        validate_cudnn_gdn_stack(SimpleNamespace(gdn_kernel_backend="transformer_engine", fp8="hybrid"))
        mock_version.assert_called_once_with("transformer-engine")
        mock_warn.assert_not_called()

    @patch("megatron.bridge.utils.gdn_utils.get_distribution_version", return_value="2.20.2+6ea2a74a")
    @patch("megatron.bridge.utils.gdn_utils.probe_cudnn_gdn_stack_issues", side_effect=lambda: ["GAP"])
    @patch("megatron.bridge.utils.gdn_utils.warn_rank_0")
    def test_fp8_asks_for_the_fixed_te_with_other_gaps(self, mock_warn, mock_probe, mock_version):
        """With FP8 set, the install hint asks for the Transformer Engine that runs GatedDeltaNet under FP8."""
        validate_cudnn_gdn_stack(SimpleNamespace(gdn_kernel_backend="transformer_engine", fp8="e4m3"))
        message = mock_warn.call_args[0][0]
        assert (
            f"environment: GAP. Install transformer-engine>={_version_str(TRANSFORMER_ENGINE_GDN_FP8_VERSION)},"
            in message
        )
        assert "raises under FP8 autocast" not in message

    @staticmethod
    def _require_frost_frontend() -> None:
        frontend = get_distribution_version("nvidia-cudnn-frontend")
        if frontend is None or _release_tuple(frontend) < CUDNN_FRONTEND_GDN_NAN_FREE_VERSION:
            floor = _version_str(CUDNN_FRONTEND_GDN_NAN_FREE_VERSION)
            pytest.skip(f"needs nvidia-cudnn-frontend >= {floor} (installed: {frontend})")

    def test_cutlass_dsl_floor_matches_the_installed_cudnn_frontend(self):
        """CUTLASS_DSL_GDN_MIN_VERSION copies CUTEDSL_MIN_VERSION; this fails when a frontend bump moves it."""
        self._require_frost_frontend()
        spec = importlib.util.find_spec("cudnn")  # top-level lookup; does not import cudnn
        assert spec is not None and spec.submodule_search_locations, "nvidia-cudnn-frontend is installed but missing"
        buffers = Path(list(spec.submodule_search_locations)[0], "frost", "buffers.py")
        assert buffers.is_file(), f"{buffers} moved; re-check cudnn_gdn_stack_issues against the new FROST gate"
        floor = None
        for node in ast.parse(buffers.read_text()).body:
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            if any(isinstance(target, ast.Name) and target.id == "CUTEDSL_MIN_VERSION" for target in targets):
                floor = ast.literal_eval(node.value)
        assert floor is not None, f"{buffers} no longer defines CUTEDSL_MIN_VERSION"
        assert tuple(floor) == CUTLASS_DSL_GDN_MIN_VERSION

    @pytest.mark.parametrize(
        "version", ["4.5.0", "4.6", "4.6.2rc1", "4.6.post1", "4.7.0", "4.7.0.dev0", "4.8.0a0+local", "dev"]
    )
    def test_cutlass_dsl_rule_matches_the_installed_cudnn_frontend(self, version):
        """Bridge flags exactly the public CuTe DSL versions that cuDNN frontend's own FROST gate rejects."""
        self._require_frost_frontend()
        buffers = pytest.importorskip("cudnn.frost.buffers")
        flagged = bool(cudnn_gdn_stack_issues(**{**self._COMPLETE, "cutlass_dsl": ("nvidia-cutlass-dsl", version)}))
        assert flagged == buffers.cutedsl_too_old(("nvidia-cutlass-dsl", version))
