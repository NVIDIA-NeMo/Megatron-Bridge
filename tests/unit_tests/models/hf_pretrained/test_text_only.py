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

import json
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import PretrainedConfig

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource


pytestmark = pytest.mark.unit


@pytest.fixture
def checkpoint(tmp_path):
    weights = {
        "language_model.backbone.embeddings.weight": torch.arange(12).reshape(3, 4).float(),
        "language_model.mtp.layers.0.weight": torch.arange(8).reshape(2, 4).float(),
    }
    save_file(weights, tmp_path / "language.safetensors")
    # Deliberately absent vision shard: neither import nor export may open it.
    index = {"weight_map": {key: "language.safetensors" for key in weights}}
    index["weight_map"]["vision_model.weight"] = "vision.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return tmp_path, weights


def _source(path):
    return SafeTensorsStateSource(path, key_prefix="language_model.")


def _select_text(source, *, config):
    from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import Nemotron35SuperVLBridge

    config.num_nextn_predict_layers = 1
    config.mtp_layers_block_type = ["attention", "moe"]
    source.config.llm_config = config
    return Nemotron35SuperVLBridge().text_only_pretrained(source)


def test_language_keys_and_mtp_are_lazy_and_exact(checkpoint):
    path, weights = checkpoint
    source = _source(path)
    assert source.get_all_keys() == ["backbone.embeddings.weight", "mtp.layers.0.weight"]
    loaded = source.load_tensors(source.get_all_keys())
    for key, tensor in loaded.items():
        assert torch.equal(tensor, weights["language_model." + key])


def test_missing_language_weight_fails_without_media_fallback(checkpoint):
    path, _ = checkpoint
    with pytest.raises(KeyError, match="missing"):
        _source(path).load_tensors(["missing.weight"])


def test_no_language_subtree_fails_on_every_access(tmp_path):
    save_file({"vision.weight": torch.ones(1)}, tmp_path / "model.safetensors")
    source = _source(tmp_path)
    for _ in range(2):
        with pytest.raises(ValueError, match="no weights under"):
            source.get_all_keys()


def test_export_is_standalone_and_strict(checkpoint, tmp_path):
    path, weights = checkpoint
    source = _source(path)
    tensors = {key.removeprefix("language_model."): tensor for key, tensor in weights.items()}
    output = tmp_path / "export"
    source.save_generator(iter(tensors.items()), output)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(tensors)
    exported = load_file(output / "language.safetensors")
    assert set(exported) == set(tensors)
    for key in tensors:
        assert torch.equal(exported[key], tensors[key])
    with pytest.raises(KeyError):
        source.save_generator(iter([("not_a_language_weight", torch.ones(1))]), tmp_path / "invalid")


def test_export_rejects_missing_mtp(checkpoint, tmp_path):
    path, weights = checkpoint
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        _source(path).save_generator(
            iter([("backbone.embeddings.weight", weights["language_model.backbone.embeddings.weight"])]),
            tmp_path / "incomplete",
        )


def test_hub_downloads_only_requested_shards_at_pinned_revision(checkpoint, monkeypatch):
    path, _ = checkpoint
    download = Mock(side_effect=lambda repo, filename, **kwargs: str(path / filename))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    source = SafeTensorsStateSource(
        "org/model", key_prefix="language_model.", revision="immutable-revision", hub_kwargs={"local_files_only": True}
    )
    assert len(source.get_all_keys()) == 2
    assert [call.args[1] for call in download.call_args_list] == ["model.safetensors.index.json"]
    source.load_tensors(["mtp.layers.0.weight"])
    assert [call.args[1] for call in download.call_args_list] == [
        "model.safetensors.index.json",
        "language.safetensors",
    ]
    assert all(call.kwargs["revision"] == "immutable-revision" for call in download.call_args_list)


@pytest.mark.parametrize("prefix", ["", "language_model"])
def test_source_rejects_invalid_prefix(tmp_path, prefix):
    with pytest.raises(ValueError, match="key_prefix"):
        SafeTensorsStateSource(tmp_path, key_prefix=prefix)


def test_prefix_views_do_not_modify_shared_index_or_default_source(tmp_path):
    tensors = {"text.weight": torch.ones(2), "vision.weight": torch.zeros(3)}
    save_file(tensors, tmp_path / "mixed.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "mixed.safetensors" for key in tensors}})
    )
    plain = SafeTensorsStateSource(tmp_path)
    assert set(plain.get_all_keys()) == set(tensors)
    for prefix in ("text.", "vision."):
        selected = SafeTensorsStateSource(tmp_path, key_prefix=prefix)
        assert selected.get_all_keys() == ["weight"]
        assert selected.has_glob("weight")
        assert not selected.has_glob(prefix + "*")
        assert torch.equal(selected.load_tensors(["weight"])["weight"], tensors[prefix + "weight"])
    assert set(plain.key_to_filename_map) == set(tensors)
    assert set(SafeTensorsStateSource(tmp_path).key_to_filename_map) == set(tensors)
    for key, tensor in plain.load_tensors(plain.get_all_keys()).items():
        assert torch.equal(tensor, tensors[key])


def test_selected_source_does_not_scan_other_shards_for_missing_tensor(checkpoint, monkeypatch):
    path, _ = checkpoint
    # The index advertises a weight absent from its designated shard.
    source = _source(path)
    source.key_to_filename_map["absent.weight"] = "language.safetensors"
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("glob.glob", Mock(side_effect=AssertionError("unrelated shard scan")))
    with pytest.raises(KeyError, match="absent.weight"):
        source.load_tensors(["absent.weight"])


def test_hub_single_file_prefix_selection(tmp_path, monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError

    save_file({"text.weight": torch.ones(2), "vision.weight": torch.zeros(3)}, tmp_path / "model.safetensors")

    def download(repo, filename, **kwargs):
        assert kwargs["revision"] == "pinned"
        if filename.endswith(".json"):
            raise EntryNotFoundError("no index")
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    source = SafeTensorsStateSource("org/model", key_prefix="text.", revision="pinned")
    assert source.get_all_keys() == ["weight"]
    assert torch.equal(source.load_tensors(["weight"])["weight"], torch.ones(2))


def test_hub_revision_without_prefix_keeps_original_keys(checkpoint, monkeypatch):
    path, weights = checkpoint
    download = Mock(side_effect=lambda repo, filename, **kwargs: str(path / filename))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    source = SafeTensorsStateSource("org/model", revision="pinned", hub_kwargs={"local_files_only": True})
    assert set(source.get_all_keys()) == {*weights, "vision_model.weight"}
    for key, tensor in source.load_tensors(list(weights)).items():
        assert torch.equal(tensor, weights[key])
    assert all(call.kwargs == {"revision": "pinned", "local_files_only": True} for call in download.call_args_list)


def test_wrapper_retains_revision_and_drops_media_artifacts(checkpoint):
    path, _ = checkpoint
    original = PreTrainedCausalLM.from_pretrained(path, revision="requested", trust_remote_code=True)
    original.config = PretrainedConfig()
    original.config._commit_hash = "resolved"
    config = PretrainedConfig(architectures=["NemotronHForCausalLM"])
    wrapper = _select_text(original, config=config)
    assert type(wrapper) is PreTrainedCausalLM
    assert wrapper._text_only
    assert not original._text_only
    assert original.OPTIONAL_ARTIFACTS == ["generation_config", "processor", "image_processor"]
    assert original.custom_file_patterns == ["*.py"]
    assert wrapper.config is not config
    assert wrapper.config.num_nextn_predict_layers == 2
    assert config.num_nextn_predict_layers == 1
    assert wrapper.init_kwargs["revision"] == "resolved"
    assert wrapper.processor is None
    assert wrapper.image_processor is None
    assert not wrapper.has_model
    assert set(wrapper.state) == {"backbone.embeddings.weight", "mtp.layers.0.weight"}
    with pytest.raises(NotImplementedError, match="standalone HF export"):
        _ = wrapper.model


def test_projection_does_not_change_regular_wrapper_loading(checkpoint, monkeypatch):
    path, _ = checkpoint
    original = PreTrainedCausalLM.from_pretrained(path, device="cpu")
    original.config = PretrainedConfig()
    selected = _select_text(original, config=PretrainedConfig())
    model = Mock()
    model.to.return_value = model
    load = Mock(return_value=model)
    monkeypatch.setattr("megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained", load)
    with pytest.raises(NotImplementedError, match="standalone HF export"):
        _ = selected.model
    load.assert_not_called()
    assert original.model is model
    load.assert_called_once()
    assert load.call_args.args == (path,)


def test_projection_does_not_load_media_artifacts(checkpoint, tmp_path, monkeypatch):
    path, _ = checkpoint
    source = PreTrainedCausalLM.from_pretrained(path)
    source.config = PretrainedConfig()
    selected = _select_text(source, config=PretrainedConfig())
    selected.tokenizer = Mock()
    selected.generation_config = None
    monkeypatch.setattr(PreTrainedCausalLM, "_load_processor", Mock(side_effect=AssertionError("media load")))
    monkeypatch.setattr(PreTrainedCausalLM, "_load_image_processor", Mock(side_effect=AssertionError("media load")))
    selected.save_artifacts(tmp_path / "text-artifacts")
    selected.tokenizer.save_pretrained.assert_called_once()


def test_native_nemotron_text_export_preserves_logits(tmp_path):
    """HF-to-HF projection check; not a substitute for HF-to-MCore parity."""
    from transformers.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM

    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=16,
        layers_block_type=["mamba", "attention", "moe"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        mamba_num_heads=4,
        mamba_head_dim=8,
        ssm_state_size=4,
        n_groups=1,
        chunk_size=4,
        n_routed_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=16,
        moe_latent_size=8,
        use_mamba_kernels=False,
    )
    original = NemotronHForCausalLM(config).eval()
    nested = {"language_model." + name: tensor for name, tensor in original.state_dict().items()}
    nested["vision.weight"] = torch.ones(1)
    save_file(nested, tmp_path / "model.safetensors")
    source = _source(tmp_path)
    export = tmp_path / "text"
    source.save_generator(iter(source.load_tensors(source.get_all_keys()).items()), export)
    config.save_pretrained(export)
    reloaded, info = NemotronHForCausalLM.from_pretrained(export, output_loading_info=True)
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    tokens = torch.tensor([[1, 5, 7, 2]])
    with torch.no_grad():
        expected = original(tokens, use_cache=False).logits
        actual = reloaded.eval()(tokens, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wrapper_artifacts_reload_as_native_text_without_remote_code(checkpoint, tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast
    from transformers.models.nemotron_h import NemotronHConfig

    path, _ = checkpoint
    original = PreTrainedCausalLM.from_pretrained(path, trust_remote_code=True)
    original.config = PretrainedConfig()
    config = NemotronHConfig(num_nextn_predict_layers=2)
    config.architectures = ["NemotronHForCausalLM"]
    wrapper = _select_text(original, config=config)
    wrapper.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    wrapper.generation_config = None
    output = tmp_path / "artifacts"
    wrapper.save_artifacts(output)
    reloaded = AutoConfig.from_pretrained(output, trust_remote_code=False)
    assert type(reloaded) is NemotronHConfig
    assert reloaded.num_nextn_predict_layers == 2
    assert reloaded.architectures == ["NemotronHForCausalLM"]
    assert not getattr(reloaded, "auto_map", None)
    assert not hasattr(reloaded, "vision_config")
    assert AutoTokenizer.from_pretrained(output).get_vocab() == wrapper.tokenizer.get_vocab()
    assert not (output / "processor_config.json").exists()
