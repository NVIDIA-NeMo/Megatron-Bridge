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

"""Unit tests for the Inkling2 bridge configuration and parameter mappings."""

import json
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, CONFIG_MAPPING_NAMES

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.inkling import Inkling2Bridge, Inkling2Config, Inkling2ModelConfig
from megatron.bridge.models.inkling.configuration_inkling2 import Inkling2TextConfig
from megatron.bridge.models.inkling.inkling2_bridge import _InterleavedGatedExpertMapping, _vision_config_dict
from megatron.bridge.models.inkling.inkling2_quantization import companion_keys
from megatron.bridge.training.config import ConfigContainer


pytestmark = pytest.mark.unit

TEXT = {
    "hidden_size": 64,
    "num_hidden_layers": 7,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 512,
    "unpadded_vocab_size": 500,
    "d_rel": 8,
    "rel_extent": 32,
    "log_scaling_n_floor": 8192,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-6,
    "pre_norm_pre_affine": False,
    "final_norm_pre_affine": False,
    "use_value_norm_no_gamma": True,
    "qk_l2_clip_threshold": 22.627416997969522,
    "rel_l2_clip_threshold": 8.0,
    "local_layer_ids": [0, 1, 2, 3, 4, 6],
    "sconv_kernel_size": 4,
    "final_logit_softcapping": 30.0,
    "sliding_window_size": 16,
    "n_layers_per_attn_res_block": 6,
    "attn_res_normalize_values": True,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 0,
    "intermediate_size": 32,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "use_gate_bias": True,
    "torch_dtype": "bfloat16",
}
VISION = {"hidden_size": 32, "num_hidden_layers": 2, "native_patch_size": 4}


@pytest.fixture
def hf_config():
    return Inkling2Config(
        text_config=dict(TEXT),
        vision_config=dict(VISION),
        image_token_id=500,
        architectures=["Inkling2ForConditionalGeneration"],
    )


@pytest.fixture
def bridge(hf_config):
    bridge = Inkling2Bridge()
    bridge.hf_config = hf_config
    return bridge


# Transformers implements Inkling2 from v5.23.0 and keeps its own config classes; Bridge registers its own otherwise.
NATIVE_INKLING2 = "inkling2" in CONFIG_MAPPING_NAMES


def test_config_types_are_registered():
    # from_dict, as AutoConfig.from_pretrained reads a checkpoint; Transformers' own classes reject unknown keyword
    # arguments in for_model.
    # Transformers' classes check that the image projection's width matches the toy decoder's.
    config = CONFIG_MAPPING["inkling2"].from_dict(
        {
            "model_type": "inkling2",
            "text_config": dict(TEXT),
            "vision_config": {"text_hidden_size": TEXT["hidden_size"]},
        }
    )
    assert config.model_type == "inkling2" and isinstance(config, Inkling2Config) != NATIVE_INKLING2
    assert config.text_config.sliding_window_size == 16
    # Bridge's classes take Transformers' model types, so a saved config.json names the decoder section as
    # Transformers does.
    assert type(config.text_config) is CONFIG_MAPPING["inkling2_text"]
    assert isinstance(config.text_config, Inkling2TextConfig) != NATIVE_INKLING2
    assert json.loads(config.to_json_string())["text_config"]["model_type"] == "inkling2_text"


def test_vision_config_reads_a_config_object_like_the_release_dict():
    release_vision = RELEASE_CONFIG["vision_config"]
    assert _vision_config_dict(release_vision) == release_vision
    assert _vision_config_dict(PretrainedConfig(**release_vision)) == release_vision
    assert _vision_config_dict(None) == {}


def test_model_config_mapping(bridge, hf_config):
    config = bridge.hf_config_to_model_config(hf_config)
    assert isinstance(config, Inkling2ModelConfig)
    t = config.transformer
    # Each of the 7 decoder blocks is a Hybrid attention layer followed by a MoE layer.
    assert (t.num_layers, t.hidden_size, t.num_attention_heads, t.num_query_groups, t.kv_channels) == (
        14,
        64,
        4,
        2,
        16,
    )
    assert config.hybrid_layer_pattern == "*E" * 7
    assert t.num_moe_experts == 8 and t.moe_router_topk == 2 and t.moe_ffn_hidden_size == 32
    assert t.moe_router_score_function == "sigmoid" and t.moe_router_enable_expert_bias
    assert t.moe_router_topk_scaling_factor == 8.0 and t.moe_router_bias_update_rate == 0.0
    assert t.inkling2_local_layers == (0, 1, 2, 3, 4, 6)
    assert t.inkling2_value_norm and t.inkling2_attn_res_normalize_values
    assert (t.inkling2_mlp_gate_softcap, t.inkling2_mlp_up_softcap) == (4.0, 25.0)
    assert t.inkling2_final_logit_softcap == 30.0 and t.inkling2_unpadded_vocab_size == 500
    assert t.params_dtype == torch.bfloat16
    # Layers have different AttnRes query and norm parameters, so each layer gets its own checkpoint keys.
    assert t.hetereogenous_dist_checkpoint is True
    assert config.position_embedding_type == "none" and not config.share_embeddings_and_output_weights
    assert config.vision_config == VISION and config.image_token_id == 500
    assert config.special_token_ids == {"images": 500}


@pytest.mark.parametrize(
    "field, value",
    [("n_shared_experts", 2), ("pre_norm_pre_affine", True), ("swa_num_attention_heads", 8)],
)
def test_unsupported_configurations_are_rejected(field, value):
    text = dict(TEXT, **{field: value})
    config = Inkling2Config(text_config=text, vision_config=dict(VISION))
    bridge = Inkling2Bridge()
    bridge.hf_config = config
    with pytest.raises(ValueError, match="Unsupported Inkling2 configuration"):
        bridge.hf_config_to_model_config(config)


def test_imported_config_defaults_to_no_expert_tensor_parallelism(bridge, hf_config):
    assert bridge.hf_config_to_model_config(hf_config).transformer.expert_tensor_parallel_size == 1


# config.json of the release, thinkingmachines/Inkling-2-Small at revision 78cb6c0e.
RELEASE_CONFIG = json.loads((Path(__file__).parent / "fixtures" / "inkling2_small_78cb6c0e_config.json").read_text())


def test_release_config_loads_and_maps(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(RELEASE_CONFIG))
    config = AutoConfig.from_pretrained(tmp_path)
    assert config.model_type == "inkling2" and isinstance(config, Inkling2Config) != NATIVE_INKLING2
    # Not in the release's config.json: the image placeholder of its processor_config.json.
    assert config.image_token_id == 200054

    bridge = Inkling2Bridge()
    bridge.hf_config = config
    model_config = bridge.hf_config_to_model_config(config)
    t = model_config.transformer
    assert (t.num_layers, t.hidden_size, t.num_moe_experts, t.moe_router_topk) == (96, 4608, 384, 8)
    assert t.inkling2_log_scaling_n_floor == 128000.0 and t.inkling2_attn_res_block_size == 6
    assert t.inkling2_local_layers == tuple(RELEASE_CONFIG["text_config"]["local_layer_ids"])
    assert t.inkling2_final_logit_softcap == 30.0 and t.inkling2_unpadded_vocab_size == 200058
    assert model_config.image_token_id == 200054 and model_config.special_token_ids == {"images": 200054}


def test_config_defaults_are_the_release_values():
    defaults = Inkling2Config().text_config
    assert {key: getattr(defaults, key) for key in RELEASE_CONFIG["text_config"] if key != "torch_dtype"} == {
        key: value for key, value in RELEASE_CONFIG["text_config"].items() if key != "torch_dtype"
    }


@pytest.mark.skipif(NATIVE_INKLING2, reason="Transformers' own Inkling2 classes write their own config layout")
def test_saved_config_keeps_every_release_field(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "config.json").write_text(json.dumps(RELEASE_CONFIG))
    AutoConfig.from_pretrained(tmp_path / "in").save_pretrained(tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "config.json").read_text())

    for section in ("vision_config", "mtp_config"):
        assert saved[section] == RELEASE_CONFIG[section]
    # Transformers writes the text config's ``torch_dtype`` as ``dtype``.
    text = {key: value for key, value in RELEASE_CONFIG["text_config"].items() if key != "torch_dtype"}
    assert {key: saved["text_config"][key] for key in text} == text
    assert saved["text_config"]["dtype"] == RELEASE_CONFIG["text_config"]["torch_dtype"]
    assert {key: saved[key] for key in ("architectures", "model_type", "eos_token_id")} == {
        key: RELEASE_CONFIG[key] for key in ("architectures", "model_type", "eos_token_id")
    }


def _save_run_config(checkpoint_dir: Path, model_config: Inkling2ModelConfig) -> None:
    """Write the run config of a Megatron checkpoint, which the export reads through ``from_auto_config``."""
    checkpoint_dir.mkdir()
    ConfigContainer(
        model=model_config,
        train=None,
        optimizer=None,
        scheduler=None,
        dataset=None,
        logger=None,
        tokenizer=None,
        checkpoint=None,
    ).to_yaml(str(checkpoint_dir / "run_config.yaml"))


def _plain_fields(config, *, exclude: tuple[str, ...] = ()) -> dict:
    """Return the config fields that hold plain values, which compare across instances."""
    plain = (bool, int, float, str, tuple, list, dict, type(None), torch.dtype)
    return {
        f.name: getattr(config, f.name)
        for f in fields(config)
        if f.name not in exclude and isinstance(getattr(config, f.name), plain)
    }


@pytest.fixture
def release_dir(tmp_path) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "config.json").write_text(json.dumps(RELEASE_CONFIG))
    return release_dir


@pytest.mark.skipif(NATIVE_INKLING2, reason="Transformers' own Inkling2 classes write their own config layout")
def test_checkpoint_export_reproduces_the_release_config(tmp_path, release_dir):
    reference = AutoConfig.from_pretrained(release_dir)
    _save_run_config(tmp_path / "checkpoint", AutoBridge.from_hf_config(reference).get_model_config())

    rebuilt = AutoBridge.from_auto_config(str(tmp_path / "checkpoint"), str(release_dir)).hf_pretrained

    # The export writes this JSON as config.json.
    assert rebuilt.to_json_string() == reference.to_json_string()


def test_checkpoint_export_keeps_the_saved_architecture(tmp_path, release_dir):
    """A checkpoint unlike the reference exports its own settings, and the reference fills only the rest."""
    vision = dict(RELEASE_CONFIG["vision_config"], hidden_size=32, num_hidden_layers=2, native_patch_size=4)
    saved = Inkling2Config(
        text_config=dict(TEXT, final_logit_softcapping=12.0),
        vision_config=vision,
        image_token_id=500,
        architectures=["Inkling2ForConditionalGeneration"],
    )
    model_config = AutoBridge.from_hf_config(saved).get_model_config()
    _save_run_config(tmp_path / "checkpoint", model_config)

    rebuilt = AutoBridge.from_auto_config(str(tmp_path / "checkpoint"), str(release_dir)).hf_pretrained

    text = {key: value for key, value in TEXT.items() if key != "torch_dtype"}
    assert {key: getattr(rebuilt.text_config, key) for key in text} == dict(text, final_logit_softcapping=12.0)
    assert rebuilt.vision_config == vision and rebuilt.image_token_id == 500
    # The Megatron model holds neither the draft model nor the token IDs of the tokenizer.
    assert rebuilt.mtp_config == RELEASE_CONFIG["mtp_config"]
    assert rebuilt.eos_token_id == RELEASE_CONFIG["eos_token_id"]

    reimported = AutoBridge.from_hf_config(rebuilt).get_model_config()
    assert _plain_fields(reimported.transformer) == _plain_fields(model_config.transformer)
    # Only the recorded source differs: the rebuilt config names the reference.
    source = ("extra_checkpoint_metadata",)
    assert _plain_fields(reimported, exclude=source) == _plain_fields(model_config, exclude=source)


def test_model_without_the_image_encoder_exports_no_vision_config():
    config = Inkling2Config(text_config=dict(TEXT), architectures=["Inkling2ForConditionalGeneration"])
    bridge = Inkling2Bridge()
    bridge.hf_config = config

    exported = Inkling2Bridge.megatron_to_hf_config(bridge.hf_config_to_model_config(config))

    assert exported["vision_config"] is None and "image_token_id" not in exported


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("hierarchical_context_parallel_sizes", [2, 2], "hierarchical CP"),
        ("virtual_pipeline_model_parallel_size", 2, "virtual pipeline"),
        ("expert_tensor_parallel_size", 2, "expert_tensor_parallel_size=1"),
    ],
)
def test_finalize_rejects_unvalidated_parallelism(bridge, hf_config, field, value, message):
    config = bridge.hf_config_to_model_config(hf_config)
    if not hasattr(config.transformer, field):
        raise ValueError(f"Inkling2 transformer config has no field {field!r}")
    setattr(config.transformer, field, value)
    with pytest.raises(ValueError, match=message):
        config.finalize()


def _resolved(bridge):
    registry = bridge.mapping_registry()
    return {mapping.megatron_param: mapping for mapping in registry.mappings}


def test_layer_type_specific_mappings(bridge):
    """Block ``i`` maps to Hybrid attention layer ``2 * i`` and MoE layer ``2 * i + 1``."""
    mappings = _resolved(bridge)
    local = mappings["decoder.layers.0.self_attention.k_conv_weight"]
    assert isinstance(local, ColumnParallelMapping)
    assert local.hf_param == "model.llm.layers.0.attn.k_sconv.weight"
    assert mappings["decoder.layers.0.input_layernorm.weight"].hf_param == "model.llm.layers.0.attn_norm.weight"
    full_static = mappings["decoder.layers.10.self_attention.k_conv_static"]
    assert isinstance(full_static, RowParallelMapping)
    assert full_static.hf_param == "model.llm.layers.5.attn.k_sconv.static_w"
    assert mappings["decoder.layers.10.input_layernorm.weight"].hf_param == "model.llm.layers.5.q_input_norm.weight"
    assert mappings["decoder.layers.10.v_input_norm.weight"].hf_param == "model.llm.layers.5.v_input_norm.weight"
    assert "decoder.layers.0.attn_res_query" not in mappings
    assert mappings["decoder.layers.2.attn_res_query"].hf_param == "model.llm.attn_res.attn_queries.0"
    assert mappings["decoder.layers.13.mlp_res_query"].hf_param == "model.llm.attn_res.mlp_queries.6"
    assert mappings["decoder.layers.13.output_res_query"].hf_param == "model.llm.attn_res.output_query"
    assert mappings["decoder.layers.13.mlp.router.expert_bias"].hf_param == "model.llm.layers.6.mlp.gate.bias"
    assert mappings["decoder.layers.12.self_attention.linear_proj.weight"].hf_param == (
        "model.llm.layers.6.attn.wo_ud.weight"
    )
    assert mappings["decoder.final_norm.weight"].hf_param == "model.llm.norm.weight"
    # Attention layers hold no MoE tensors and MoE layers no attention tensors.
    assert not any(name.startswith("decoder.layers.1.self_attention") for name in mappings)
    assert not any(name.startswith("decoder.layers.0.mlp") for name in mappings)
    assert isinstance(mappings["vision.**"], ReplicatedMapping)


def test_interleaved_expert_mapping_round_trips_rows():
    mapping = _InterleavedGatedExpertMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.weight3", "model.llm.layers.0.mlp.experts.w13_weight"
    )
    gate = torch.arange(12.0).reshape(3, 4)
    up = -torch.arange(12.0).reshape(3, 4)

    class _Gated:
        def megatron_to_hf(self, weights, module):
            return {"x.gate": gate, "x.up": up}

        def hf_to_megatron(self, weights, module):
            return weights

    mapping._gated_mapping = _Gated()
    interleaved = mapping.megatron_to_hf(None, None)["x"]
    assert torch.equal(interleaved[0::2], gate) and torch.equal(interleaved[1::2], up)
    fused = torch.stack([torch.zeros_like(interleaved)] * 3 + [interleaved])
    split = mapping.hf_to_megatron(fused, None)
    assert torch.equal(split["gate"], gate) and torch.equal(split["up"], up)


def test_plain_export_keeps_float32_and_skips_expert_scales():
    """A plain export writes BF16 experts without scales and keeps the FP32 tensors."""
    suffixes = Inkling2Bridge.PLAIN_EXPORT_IGNORED_SOURCE_KEY_SUFFIXES
    expert = "model.llm.layers.0.mlp.experts.w13_weight"
    assert Inkling2Bridge.PLAIN_EXPORT_KEEPS_FLOAT32 is True
    assert "_scale_inv" in suffixes
    for name in companion_keys(expert, "nvfp4") + companion_keys(expert.replace("w13", "w2"), "mxfp8"):
        assert name.endswith(suffixes), name
    for name in (expert, "model.llm.layers.0.mlp.gate.bias", "model.llm.attn_res.output_query"):
        assert not name.endswith(suffixes), name


# The config.json an export saves through Transformers, which records its version.
_SAVED_HF_CONFIG = {
    "architectures": ["Inkling2ForConditionalGeneration"],
    "model_type": "inkling2",
    "transformers_version": "5.20.2",
}


def _transformers_json(config: dict) -> str:
    """A config dict laid out as Transformers writes ``config.json``."""
    return json.dumps(config, indent=2, sort_keys=True) + "\n"


def _write_export(path, tensors_by_shard, groups):
    path.mkdir()
    weight_map = {}
    for shard, tensors in tensors_by_shard.items():
        save_file(tensors, str(path / shard))
        weight_map.update(dict.fromkeys(tensors, shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    config = {"model": {"parameter_groups": groups}, "quant_method": "inkling2"}
    (path / "hf_quant_config.json").write_text(json.dumps(config, indent=2))
    (path / "config.json").write_text(_transformers_json(_SAVED_HF_CONFIG))


_PLAIN = {"required_activation_quantization": None, "weight_block_shape": [], "weight_transform": "cast"}
_NVFP4_GROUP = {
    "compute_dtype": "nvfp4",
    "output_dtype": "uint8",
    "required_activation_quantization": {"format": "nvfp4_e2m1"},
    "weight_block_shape": [1, 16],
    "weight_scale_format": "amax_denominator",
    "weight_transform": "nvfp4_nearest",
}
_MXFP8_GROUP = {
    "compute_dtype": "mxfp8",
    "output_dtype": "float8_e4m3fn",
    "required_activation_quantization": {
        "block_size": 32,
        "format": "mxfp8_e4m3",
        "rounding": "nearest",
        "scale_dtype": "ue8m0",
    },
    "weight_block_shape": [],
    "weight_transform": "mxfp8_nearest",
}


def test_postprocess_lists_plain_experts_as_bf16(tmp_path):
    expert = "model.llm.layers.0.mlp.experts.w13_weight"
    groups = [
        {
            "compute_dtype": "float32",
            "output_dtype": "float32",
            "tensors": ["model.llm.layers.0.mlp.gate.bias"],
            **_PLAIN,
        },
        {"compute_dtype": "bfloat16", "output_dtype": "bfloat16", "tensors": ["model.llm.embed.weight"], **_PLAIN},
        {**_NVFP4_GROUP, "tensors": [expert]},
    ]
    _write_export(
        tmp_path / "export",
        {
            "model-00001-of-00002.safetensors": {
                expert: torch.zeros(2, 4, 32, dtype=torch.bfloat16),
                "model.llm.layers.0.mlp.gate.bias": torch.zeros(2),
            },
            "model-00002-of-00002.safetensors": {"model.llm.embed.weight": torch.zeros(8, 4, dtype=torch.bfloat16)},
        },
        groups,
    )

    Inkling2Bridge().postprocess_hf_export_weights(tmp_path / "export")

    config = json.loads((tmp_path / "export" / "hf_quant_config.json").read_text())
    assert config["model"]["parameter_groups"] == [
        {
            "compute_dtype": "float32",
            "output_dtype": "float32",
            "tensors": ["model.llm.layers.0.mlp.gate.bias"],
            **_PLAIN,
        },
        {
            "compute_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "tensors": ["model.llm.embed.weight", expert],
            **_PLAIN,
        },
    ]
    assert config["quant_method"] == "inkling2"
    # The recorded version tells Transformers that the BF16 experts are not quantized.
    assert (tmp_path / "export" / "config.json").read_text() == _transformers_json(_SAVED_HF_CONFIG)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_postprocess_keeps_a_quantized_export_in_the_release_format(tmp_path, fmt):
    expert = "model.llm.layers.0.mlp.experts.w13_weight"
    if fmt == "nvfp4":
        group = _NVFP4_GROUP
        tensors = {
            expert: torch.zeros(2, 4, 16, dtype=torch.uint8),
            f"{expert}_scale": torch.zeros(2, 4, 2).to(torch.float8_e4m3fn),
            f"{expert}_amax": torch.ones(2),
            f"{expert}_outer_scale_denominator": torch.ones(1),
        }
    else:
        group = _MXFP8_GROUP
        tensors = {
            expert: torch.zeros(2, 4, 32).to(torch.float8_e4m3fn),
            f"{expert}_scale": torch.zeros(2, 4, 1, dtype=torch.uint8),
        }
    _write_export(tmp_path / "export", {"model.safetensors": tensors}, [{**group, "tensors": [expert]}])
    before = (tmp_path / "export" / "hf_quant_config.json").read_bytes()

    Inkling2Bridge().postprocess_hf_export_weights(tmp_path / "export")

    assert (tmp_path / "export" / "hf_quant_config.json").read_bytes() == before
    # Without the version, as the release's config.json, which Transformers loads with the release's quantization.
    release_like = {key: value for key, value in _SAVED_HF_CONFIG.items() if key != "transformers_version"}
    assert (tmp_path / "export" / "config.json").read_text() == _transformers_json(release_like)
