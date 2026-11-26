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

"""Unit tests for the looped Inkling2 (Inkling 2 Mini) bridge."""

import pytest
import torch


pytest.importorskip("transformers", minversion="5.24")

from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig

from megatron.bridge.models.conversion.model_bridge import get_model_bridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.inkling import Inkling2ModelConfig
from megatron.bridge.models.inkling.inkling2_bridge import (
    LOOP_ARCHITECTURE,
    Inkling2LoopBridge,
)
from megatron.bridge.models.inkling.modeling_inkling2.loop import decoder_block_runs, loop_core_blocks


pytestmark = pytest.mark.unit

TEXT = {
    "hidden_size": 64,
    "num_hidden_layers": 12,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "swa_head_dim": 16,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "vocab_size": 512,
    "unpadded_vocab_size": 500,
    "d_rel": 8,
    "rel_extent": 32,
    "log_scaling_n_floor": 8192,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-6,
    "q_bias": False,
    "o_bias": False,
    "use_value_norm_no_gamma": True,
    "qk_l2_clip_threshold": 22.627416997969522,
    "rel_l2_clip_threshold": 8.0,
    "local_layer_ids": [i for i in range(12) if i % 3 != 2],
    "sconv_kernel_size": 4,
    "final_logit_softcapping": 30.0,
    "sliding_window_size": 16,
    "n_layers_per_attn_res_block": 3,
    "attn_res_normalize_values": True,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 0,
    "intermediate_size": 32,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "use_gate_bias": True,
    "loop_config": {"core_start": 3, "core_end": 9, "num_passes": 3, "gate_rank": 8, "is_varloop": False},
}

# Transformers checks that the image projection's width matches the toy decoder's.
VISION = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "native_patch_size": 4,
    "text_hidden_size": TEXT["hidden_size"],
    "temporal_patch_size": 2,
}


@pytest.fixture
def hf_config():
    return Inkling2LoopConfig.from_dict(
        dict(text_config=dict(TEXT), vision_config=dict(VISION), architectures=[LOOP_ARCHITECTURE])
    )


@pytest.fixture
def bridge(hf_config):
    bridge = Inkling2LoopBridge()
    bridge.hf_config = hf_config
    return bridge


def test_loop_bridge_is_registered_for_the_looped_architecture(hf_config):
    assert type(get_model_bridge(LOOP_ARCHITECTURE, hf_config)) is Inkling2LoopBridge


def test_loop_config_mapping(bridge, hf_config):
    config = bridge.hf_config_to_model_config(hf_config)
    assert isinstance(config, Inkling2ModelConfig)
    t = config.transformer
    assert (t.inkling2_loop_core_start, t.inkling2_loop_core_end) == (3, 9)
    assert (t.inkling2_loop_num_passes, t.inkling2_loop_gate_rank) == (3, 8)
    assert loop_core_blocks(t) == range(3, 9)
    # Prelude, core x 3 passes, coda: one routing per decoder-block run, in execution order.
    runs = decoder_block_runs(t, 12)
    assert len(runs) == 12 + 2 * 6
    assert runs[:4] == [(0, 0), (1, 0), (2, 0), (3, 0)] and runs[9] == (3, 1) and runs[-1] == (11, 0)


def test_loop_config_round_trips(bridge, hf_config):
    exported = Inkling2LoopBridge.megatron_to_hf_config(bridge.hf_config_to_model_config(hf_config))
    assert exported["architectures"] == [LOOP_ARCHITECTURE] and exported["model_type"] == "inkling2_loop"
    text = exported["text_config"]
    assert text["loop_config"] == TEXT["loop_config"]
    assert (text["loop_layer_index_start"], text["loop_layer_index_end"]) == (3, 9)
    reloaded = Inkling2LoopConfig.from_dict(exported)
    assert (reloaded.text_config.loop_layer_index_start, reloaded.text_config.loop_num_passes) == (3, 3)


def test_variable_pass_checkpoints_are_rejected(hf_config):
    text = dict(TEXT, loop_config=dict(TEXT["loop_config"], is_varloop=True))
    config = Inkling2LoopConfig.from_dict(dict(text_config=text, vision_config=dict(VISION)))
    bridge = Inkling2LoopBridge()
    bridge.hf_config = config
    with pytest.raises(ValueError, match="is_varloop"):
        bridge.hf_config_to_model_config(config)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("inkling2_loop_core_start", 2, "whole AttnRes blocks"),
        ("inkling2_loop_core_end", 13, "whole AttnRes blocks"),
        ("inkling2_loop_gate_rank", 0, "gate_rank"),
        ("pipeline_model_parallel_size", 2, "pipeline or context"),
    ],
)
def test_finalize_rejects_invalid_loops(bridge, hf_config, field, value, message):
    config = bridge.hf_config_to_model_config(hf_config)
    setattr(config.transformer, field, value)
    with pytest.raises(ValueError, match=message):
        config.finalize()


def test_loop_mappings(bridge):
    mappings = {mapping.megatron_param: mapping for mapping in bridge.mapping_registry().mappings}
    # Core block 3: attn / k / v controllers on Hybrid layer 6 (attention), mlp on layer 7 (MoE).
    for module, layer in (("attn", 6), ("k", 6), ("v", 6), ("mlp", 7)):
        mapping = mappings[f"decoder.layers.{layer}.controllers.{module}.down_weight"]
        assert isinstance(mapping, ReplicatedMapping)
        assert mapping.hf_param == f"model.llm.layers.3.controllers.{module}.down_weight"
    assert mappings["decoder.layers.17.controllers.mlp.gain_weight"].hf_param == (
        "model.llm.layers.8.controllers.mlp.gain_weight"
    )
    # Blocks outside the core have none.
    assert not any(name.startswith(("decoder.layers.4.controllers", "decoder.layers.18.controllers")) for name in mappings)
    for name in ("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"):
        assert mappings[f"decoder.loop_queries.{name}"].hf_param == f"model.llm.{name}"
    # The unlooped mappings are all still there.
    assert mappings["decoder.layers.23.output_res_query"].hf_param == "model.llm.attn_res.output_query"

