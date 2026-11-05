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

"""CPU tests of :class:`Ernie55ModelConfig` and the ERNIE 5.5 hybrid layout helpers."""

import pytest
import torch
from megatron.core.transformer.experimental_attention_variant import deepseek_v4_hybrid_attention as dsv4

from tests.unit_tests.models.ernie.ernie55_test_utils import allow_conversion_imports_without_te


# Bridge's conversion stack imports its LoRA layers, which need Transformer Engine; without TE, import it with a
# placeholder for them.
allow_conversion_imports_without_te()
pytest.importorskip("megatron.bridge.models.conversion.model_bridge", reason="needs Bridge's conversion stack")

from megatron.bridge.models.ernie.ernie55_bridge import Ernie55Bridge
from megatron.bridge.models.ernie.ernie55_config import (
    Ernie55ModelConfig,
    ernie55_csa_compress_ratios,
    ernie55_hybrid_layer_pattern,
    ernie55_pipeline_layer_pattern,
    parse_ernie55_hybrid_pattern,
)
from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel, mtp_loss
from tests.unit_tests.models.ernie.ernie55_hf_configs import PROXY_CONFIG, PROXY_PATTERN
from tests.unit_tests.models.ernie.test_ernie55_bridge_config import _hf
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import needs_hook


pytestmark = pytest.mark.unit


def _config(**overrides) -> Ernie55ModelConfig:
    """The proxy's model config with the PyTorch DSA fallback (``dsa_kernel_backend="none"``).

    The fused cuDNN backend needs a GPU to finalize.
    """
    bridge = Ernie55Bridge()
    config = bridge.hf_config_to_model_config(_hf(PROXY_CONFIG))
    config.dsa_kernel_backend = "none"
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


class TestLayoutHelpers:
    def test_pattern_from_compress_ratios(self):
        assert (
            ernie55_hybrid_layer_pattern(
                [128, -2, 128, -2, -2], num_hidden_layers=4, first_k_dense_replace=1, num_mtp_layers=1
            )
            == PROXY_PATTERN
        )
        assert (
            ernie55_hybrid_layer_pattern(
                [-2, 128, -2, -2], num_hidden_layers=2, first_k_dense_replace=0, num_mtp_layers=2
            )
            == "DEHE/DE/DE"
        )
        # Dense MTP depth when first_k_dense_replace covers it.
        assert ernie55_hybrid_layer_pattern(
            [-2, -2], num_hidden_layers=1, first_k_dense_replace=2, num_mtp_layers=1
        ) == ("D-/D-")

    def test_csa_compress_ratios(self):
        assert ernie55_csa_compress_ratios(PROXY_PATTERN) == [128, 0, 0, 0, 128, 0, 0, 0, 0, 0]
        assert ernie55_csa_compress_ratios("H-DE|HEDE/DE/DE") == [128, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0]
        assert ernie55_csa_compress_ratios("DE/HE") == [0, 0, 128, 0]

    def test_parse(self):
        layout = parse_ernie55_hybrid_pattern("H-DE|HEDE/DE")
        assert layout.layers == (("H", "-"), ("D", "E"), ("H", "E"), ("D", "E"))
        assert layout.mtp_layers == (("D", "E"),)
        assert layout.mtp_depths == 1
        assert layout.stage_boundaries == (2,)
        assert layout.num_hybrid_layers == 8

    @pytest.mark.parametrize(
        "pattern, message",
        [
            ("H-D", "splits an ERNIE layer"),
            ("C-DE", "layer 0"),
            ("HDDE", "layer 0"),
            ("H-D|E", "splits an ERNIE layer"),
            ("H-DE/D", "MTP pattern 'D' must pair"),
            ("/DE", "no decoder layer"),
        ],
    )
    def test_parse_rejects(self, pattern, message):
        with pytest.raises(ValueError, match=message):
            parse_ernie55_hybrid_pattern(pattern)

    def test_pipeline_split(self):
        assert ernie55_pipeline_layer_pattern(PROXY_PATTERN, 1) == PROXY_PATTERN
        assert ernie55_pipeline_layer_pattern(PROXY_PATTERN, 2) == "H-DE|HEDE/DE"
        assert ernie55_pipeline_layer_pattern(PROXY_PATTERN, 3) == "H-DE|HE|DE/DE"
        # Re-split a pattern saved under another pipeline size.
        assert ernie55_pipeline_layer_pattern("H-DE|HEDE/DE", 4) == "H-|DE|HE|DE/DE"
        with pytest.raises(ValueError, match="cannot fill 5"):
            ernie55_pipeline_layer_pattern(PROXY_PATTERN, 5)


@needs_hook
class TestFinalize:
    def test_proxy(self):
        config = _config()
        config.finalize()
        assert config.hybrid_layer_pattern == PROXY_PATTERN
        assert config.num_layers == 8
        assert config.mtp_num_layers == 1

    def test_csa_compress_ratios_are_derived_when_unset(self):
        config = _config(csa_compress_ratios=None)
        config.finalize()
        assert config.csa_compress_ratios == [128, 0, 0, 0, 128, 0, 0, 0, 0, 0]

    def test_stale_csa_compress_ratios_are_rejected(self):
        with pytest.raises(ValueError, match="csa_compress_ratios"):
            _config(csa_compress_ratios=[128, 0, 128, 0, 128, 0, 0, 0, 0, 0]).finalize()

    def test_mtp_depths_must_match_the_pattern(self):
        with pytest.raises(ValueError, match="MTP depths"):
            _config(mtp_num_layers=2).finalize()
        with pytest.raises(ValueError, match="MTP depths"):
            _config(hybrid_layer_pattern="H-DEHEDE").finalize()

    def test_mtp_override_pattern_is_rejected(self):
        with pytest.raises(ValueError, match="mtp_hybrid_override_pattern"):
            _config(mtp_hybrid_override_pattern="DE").finalize()

    @pytest.mark.parametrize(
        "overrides, message",
        [
            (dict(hybrid_layer_pattern="H-DCHEDE/DE"), "layer 1"),
            (dict(experimental_attention_variant="dsa"), "dsv4_hybrid"),
            (dict(position_embedding_type="rope"), "position_embedding_type"),
            (dict(share_embeddings_and_output_weights=True), "untied"),
            (dict(hybrid_layer_pattern=None), "hybrid_layer_pattern"),
        ],
    )
    def test_envelope(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            _config(**overrides).finalize()

    def test_pipeline_parallel_splits_at_ernie_layers(self):
        config = _config(pipeline_model_parallel_size=2, pipeline_dtype=torch.float32)
        config.finalize()
        assert config.hybrid_layer_pattern == "H-DE|HEDE/DE"
        assert config.csa_compress_ratios == [128, 0, 0, 0, 128, 0, 0, 0, 0, 0]

    def test_explicit_pipeline_segments_are_kept(self):
        config = _config(
            hybrid_layer_pattern="H-|DEHEDE/DE", pipeline_model_parallel_size=2, pipeline_dtype=torch.float32
        )
        config.finalize()
        assert config.hybrid_layer_pattern == "H-|DEHEDE/DE"

    @pytest.mark.skipif(context_parallel.load_cp_layout_api() is None, reason="no CP partition converter (MCore main)")
    def test_context_parallel_stack_is_contiguous(self):
        with pytest.raises(ValueError, match="contiguous"):
            _config(context_parallel_size=2, cp_partition_mode="zigzag").finalize()
        # MTP included: the per-row MTP loss normalizes each depth over the CP group.
        config = _config(
            context_parallel_size=2,
            cp_partition_mode="contiguous",
            cp_comm_type="all_gather",
            ernie_per_row_loss=True,
            calculate_per_token_loss=False,
        )
        config.finalize()
        assert config.cp_partition_mode == "contiguous"
        assert config.mtp_num_layers == 1

    @pytest.mark.skipif(context_parallel.load_cp_layout_api() is None, reason="no CP partition converter (MCore main)")
    def test_per_row_mtp_at_cp_needs_the_grouped_mtp_roll(self, monkeypatch):
        """With the CP converter but without dev's grouped MTP roll, MTP would silently be per-rank: fail instead."""
        monkeypatch.setattr(mtp_loss, "PER_ROW_MTP_LOSS_SUPPORTED", False)
        config = _config(
            context_parallel_size=2,
            cp_partition_mode="contiguous",
            cp_comm_type="all_gather",
            ernie_per_row_loss=True,
            calculate_per_token_loss=False,
        )
        with pytest.raises(RuntimeError, match="grouped MTP roll"):
            config.finalize()

    def test_context_parallelism_without_megatron_core_support_fails_loudly(self, monkeypatch):
        # Megatron-Core main has no CP partition modes; the stack must say so, not fail on a missing field.
        monkeypatch.setattr(context_parallel, "load_cp_layout_api", lambda: None)
        with pytest.raises(RuntimeError, match="Megatron-Core dev"):
            _config(context_parallel_size=2, ernie_per_row_loss=True, calculate_per_token_loss=False).finalize()

    def test_indexer_metric_divisor_counts_ernie_layers_and_mtp_depths(self):
        assert _config().dsa_indexer_metric_num_layers() == 5
        assert _config(hybrid_layer_pattern="H-DEHEDE/DE/DE", mtp_num_layers=2).dsa_indexer_metric_num_layers() == 6


def test_h_layers_need_the_hook(monkeypatch):
    monkeypatch.delattr(dsv4.DSv4HybridAttention, "_post_o_group_proj", raising=False)
    with pytest.raises(RuntimeError, match="_post_o_group_proj"):
        _config().finalize()


@needs_hook
def test_serialization_round_trip():
    config = _config()
    config.finalize()
    restored = Ernie55ModelConfig.from_dict(config.as_dict())
    assert isinstance(restored, Ernie55ModelConfig)
    assert restored.as_dict() == config.as_dict()
    assert restored.get_builder_cls().__name__ == "Ernie55ModelBuilder"
