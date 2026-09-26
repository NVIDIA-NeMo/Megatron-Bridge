# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/conversion/extract_qwen35_text_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("extract_qwen35_text_checkpoint", _SCRIPT)
extractor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(extractor)
pytestmark = pytest.mark.unit


@pytest.fixture
def source_checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "model.visual.patch_embed.proj.weight",
    ]
    tensors = {name: torch.full((2, 3), index + 0.5, dtype=torch.bfloat16) for index, name in enumerate(names)}
    save_file(tensors, source / "model.safetensors")
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {"model_type": "qwen3_5_moe_text", "hidden_size": 3, "tie_word_embeddings": True},
        "tie_word_embeddings": False,
        "bos_token_id": 10,
        "eos_token_id": [11, 12],
        "dtype": "bfloat16",
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "tokenizer_config.json").write_text('{"chat_template": "hello 世界"}')
    (source / "tokenizer.json").write_text('{"version": "1.0"}')
    (source / "chat_templates").mkdir()
    (source / "chat_templates/default.jinja").write_text("{{ messages }}")
    return source, tensors


@pytest.mark.parametrize("sharded", [False, True])
def test_extract_preserves_exact_decoder_mtp_and_tokenizer(source_checkpoint, tmp_path, sharded):
    source, tensors = source_checkpoint
    if sharded:
        (source / "model.safetensors").unlink()
        parts = [dict(list(tensors.items())[:5]), dict(list(tensors.items())[5:-1]), dict(list(tensors.items())[-1:])]
        weight_map = {}
        for index, part in enumerate(parts):
            filename = f"part-{index}.safetensors"
            save_file(part, source / filename)
            weight_map.update(dict.fromkeys(part, filename))
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    output = tmp_path / "text"

    extractor.extract_checkpoint(source=source, output=output)

    index = json.loads((output / "model.safetensors.index.json").read_text())
    expected = {
        extractor.text_weight_name(name): value
        for name, value in tensors.items()
        if not name.startswith("model.visual.")
    }
    assert set(index["weight_map"]) == set(expected)
    assert index["metadata"]["total_size"] == sum(value.numel() * value.element_size() for value in expected.values())
    for name, filename in index["weight_map"].items():
        with safe_open(output / filename, framework="pt", device="cpu") as handle:
            assert handle.metadata() == {"format": "pt"}
            torch.testing.assert_close(handle.get_tensor(name), expected[name], rtol=0, atol=0)
    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "qwen3_5_moe_text"
    assert config["architectures"] == ["Qwen3_5MoeForCausalLM"]
    assert config["tie_word_embeddings"] is False
    assert config["bos_token_id"] == 10
    assert config["eos_token_id"] == [11, 12]
    assert config["dtype"] == "bfloat16"
    assert all(config[key] == 1 for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers", "mtp_num_layers"))
    for filename in ("tokenizer.json", "tokenizer_config.json", "chat_templates/default.jinja"):
        assert (output / filename).read_bytes() == (source / filename).read_bytes()
    assert not list(tmp_path.glob(".text.incomplete-*"))


@pytest.mark.parametrize(
    "mutation", ["quantized", "wrong_model", "missing_mtp", "extra_mtp", "missing_required", "unknown_tensor"]
)
def test_invalid_checkpoint_does_not_publish_output(source_checkpoint, tmp_path, mutation):
    source, tensors = source_checkpoint
    config = json.loads((source / "config.json").read_text())
    if mutation == "quantized":
        config["quantization_config"] = {"quant_method": "fp8"}
    elif mutation == "wrong_model":
        config["text_config"]["model_type"] = "qwen3_5_text"
    elif mutation == "missing_mtp":
        del tensors["mtp.layers.0.self_attn.q_proj.weight"]
    elif mutation == "extra_mtp":
        tensors["mtp.layers.1.self_attn.q_proj.weight"] = torch.ones(2, 3)
    elif mutation == "missing_required":
        del tensors["mtp.fc.weight"]
    else:
        tensors["unknown.weight"] = torch.ones(2, 3)
    (source / "config.json").write_text(json.dumps(config))
    save_file(tensors, source / "model.safetensors")
    output = tmp_path / "text"

    with pytest.raises(ValueError):
        extractor.extract_checkpoint(source=source, output=output)

    assert not output.exists()


@pytest.mark.parametrize(
    "filename", ["../model.safetensors", "/model.safetensors", "a\\b.safetensors", "a:b.safetensors", "model.bin"]
)
def test_index_rejects_embedded_paths(source_checkpoint, tmp_path, filename):
    source, tensors = source_checkpoint
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(tensors, filename)}))

    with pytest.raises(ValueError, match="plain safetensors shard filename"):
        extractor.extract_checkpoint(source=source, output=tmp_path / "text")


def test_incomplete_index_does_not_publish_output(source_checkpoint, tmp_path):
    source, tensors = source_checkpoint
    weight_map = dict.fromkeys(tensors, "model.safetensors")
    del weight_map["model.visual.patch_embed.proj.weight"]
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    with pytest.raises(ValueError, match="index does not match"):
        extractor.extract_checkpoint(source=source, output=tmp_path / "text")

    assert not (tmp_path / "text").exists()


@pytest.mark.parametrize("filename", ["tokenizer_config.json", "tokenizer.json"])
def test_missing_tokenizer_does_not_publish_output(source_checkpoint, tmp_path, filename):
    source, _ = source_checkpoint
    (source / filename).unlink()

    with pytest.raises(FileNotFoundError):
        extractor.extract_checkpoint(source=source, output=tmp_path / "text")

    assert not (tmp_path / "text").exists()


def test_output_must_be_new_and_outside_source(source_checkpoint, tmp_path):
    source, _ = source_checkpoint
    output = tmp_path / "text"
    output.mkdir()
    (output / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        extractor.extract_checkpoint(source=source, output=output)
    assert (output / "keep").read_text() == "existing"
    with pytest.raises(ValueError, match="outside"):
        extractor.extract_checkpoint(source=source, output=source / "text")


def test_hf_snapshot_symlink_shards_are_supported(source_checkpoint, tmp_path):
    source, tensors = source_checkpoint
    blob = tmp_path / "blob"
    (source / "model.safetensors").rename(blob)
    (source / "model.safetensors").symlink_to(blob)
    extractor.extract_checkpoint(source=source, output=tmp_path / "text")
    assert (tmp_path / "text/config.json").is_file()
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(tensors)
