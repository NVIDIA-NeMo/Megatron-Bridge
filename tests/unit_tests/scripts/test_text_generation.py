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

"""Unit tests for ``megatron.bridge.inference.text_generation`` (shared helpers)."""

from __future__ import annotations

import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path

import pytest
from megatron.core.inference.sampling.torch_sampling import TorchSampling


_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "src" / "megatron" / "bridge" / "inference" / "text_generation.py"


class _AttnBackend(Enum):
    auto = "auto"
    flash = "flash"
    fused = "fused"
    unfused = "unfused"
    local = "local"


class _MambaInferenceStateConfig:
    @classmethod
    def from_model(cls, model):
        return cls()


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _PassthroughInit:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = {
        "megatron.core.inference.apis": _module(
            "megatron.core.inference.apis",
            SamplingParams=_SamplingParams,
        ),
        "megatron.core.inference.config": _module(
            "megatron.core.inference.config",
            InferenceConfig=_PassthroughInit,
            MambaInferenceStateConfig=_MambaInferenceStateConfig,
        ),
        "megatron.core.transformer.enums": _module(
            "megatron.core.transformer.enums",
            AttnBackend=_AttnBackend,
        ),
        "megatron.core.utils": _module(
            "megatron.core.utils",
            get_attr_wrapped_model=lambda model, attr: getattr(model, attr),
        ),
        "transformers": _module(
            "transformers",
            AutoConfig=_PassthroughInit,
            AutoTokenizer=_PassthroughInit,
        ),
        "megatron.bridge": _module("megatron.bridge", AutoBridge=_PassthroughInit),
        "megatron.bridge.inference._tokenizer": _module(
            "megatron.bridge.inference._tokenizer",
            HFTokenizerAdapter=_PassthroughInit,
        ),
        "megatron.bridge.models.hf_pretrained.utils": _module(
            "megatron.bridge.models.hf_pretrained.utils",
            is_safe_repo=lambda *, hf_path, trust_remote_code: bool(trust_remote_code),
        ),
        "megatron.bridge.training.utils.checkpoint_utils": _module(
            "megatron.bridge.training.utils.checkpoint_utils",
            get_hf_model_id_from_checkpoint=lambda path: None,
        ),
        "megatron.bridge.utils.activation_map": _module(
            "megatron.bridge.utils.activation_map",
            str_to_dtype=lambda name: name,
        ),
        "megatron.bridge.utils.common_utils": _module(
            "megatron.bridge.utils.common_utils",
            disable_mtp_for_inference=lambda model: None,
            print_rank_0=lambda message: None,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def text_generation(monkeypatch):
    _install_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location("bridge_text_generation_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_megatron_checkpoint_overrides_preserve_attention_backend(text_generation):
    provider = types.SimpleNamespace(cache_mla_latents=True)

    overrides = text_generation._megatron_checkpoint_overrides(
        provider,
        tp=2,
        pp=2,
        ep=4,
        etp=1,
        sequence_parallel=True,
        dtype=text_generation.torch.bfloat16,
        attention_backend="local",
        inference_moe_token_dispatcher_type="nvls",
    )

    assert overrides["attention_backend"] is text_generation.AttnBackend.local
    assert overrides["tensor_model_parallel_size"] == 2
    assert overrides["pipeline_model_parallel_size"] == 2
    assert overrides["expert_model_parallel_size"] == 4
    assert overrides["expert_tensor_parallel_size"] == 1
    assert overrides["sequence_parallel"] is True
    assert overrides["params_dtype"] is text_generation.torch.bfloat16
    assert overrides["pipeline_dtype"] is text_generation.torch.bfloat16
    assert overrides["bf16"] is True
    assert overrides["fp16"] is False
    assert overrides["cache_mla_latents"] is True
    assert overrides["inference_moe_token_dispatcher_type"] == "nvls"


def test_build_inference_config_rounds_max_requests_up_to_tp(text_generation):
    model = types.SimpleNamespace(position_embedding_type="rope", max_sequence_length=8192)

    config = text_generation.build_inference_config(
        model=model,
        max_sequence_length=4096,
        max_batch_size=None,
        num_prompts=3,
        tp=2,
        block_size_tokens=256,
        kv_cache_buffer_size_gb=20.0,
        max_tokens=None,
        return_log_probs=False,
        enable_chunked_prefill=False,
    )

    # 3 prompts rounded up to a multiple of tp=2 -> 4; rope is pass-through for max_sequence_length.
    assert config.kwargs["max_requests"] == 4
    assert config.kwargs["max_sequence_length"] == 4096
    assert config.kwargs["materialize_only_last_token_logits"] is True


def test_build_inference_config_auto_sizes_max_requests_when_unset(text_generation):
    """Server path: max_batch_size and num_prompts both None -> max_requests left as None."""
    model = types.SimpleNamespace(position_embedding_type="rope", max_sequence_length=8192)

    config = text_generation.build_inference_config(
        model=model,
        max_sequence_length=4096,
        max_batch_size=None,
        num_prompts=None,
        tp=2,
        block_size_tokens=256,
        kv_cache_buffer_size_gb=20.0,
        max_tokens=None,
        return_log_probs=False,
        enable_chunked_prefill=False,
    )

    assert config.kwargs["max_requests"] is None


def test_build_inference_config_clamps_learned_absolute_sequence_length(text_generation):
    model = types.SimpleNamespace(position_embedding_type="learned_absolute", max_sequence_length=1024)

    config = text_generation.build_inference_config(
        model=model,
        max_sequence_length=4096,
        max_batch_size=2,
        num_prompts=2,
        tp=1,
        block_size_tokens=256,
        kv_cache_buffer_size_gb=20.0,
        max_tokens=None,
        return_log_probs=True,
        enable_chunked_prefill=False,
    )

    # learned_absolute clamps to the model's table size (1024), not the requested 4096.
    assert config.kwargs["max_sequence_length"] == 1024
    assert config.kwargs["materialize_only_last_token_logits"] is False


def test_build_inference_config_explicit_for_divisible_batch(text_generation):
    """max_batch_size already divisible by tp must not raise."""
    model = types.SimpleNamespace(position_embedding_type="rope", max_sequence_length=8192)
    config = text_generation.build_inference_config(
        model=model,
        max_sequence_length=2048,
        max_batch_size=4,
        num_prompts=10,
        tp=2,
        block_size_tokens=256,
        kv_cache_buffer_size_gb=20.0,
        max_tokens=None,
        return_log_probs=False,
        enable_chunked_prefill=False,
    )
    assert config.kwargs["max_requests"] == 4


def test_resolve_hf_model_path_prefers_explicit(text_generation):
    assert text_generation.resolve_hf_model_path("meta-llama/Llama-3.2-1B", None) == "meta-llama/Llama-3.2-1B"


def test_resolve_hf_model_path_falls_back_to_checkpoint_metadata(text_generation, monkeypatch):
    monkeypatch.setattr(text_generation, "get_hf_model_id_from_checkpoint", lambda path: "org/model-from-ckpt")
    assert text_generation.resolve_hf_model_path(None, "/ckpt") == "org/model-from-ckpt"


def test_resolve_hf_model_path_raises_when_unresolvable(text_generation):
    # stub get_hf_model_id_from_checkpoint returns None
    with pytest.raises(ValueError, match="--hf_model_path is required"):
        text_generation.resolve_hf_model_path(None, "/ckpt")


def test_load_prompts_explicit_and_default(text_generation):
    assert text_generation.load_prompts(["a", "b"], None, None, ["default"]) == ["a", "b"]
    assert text_generation.load_prompts([], None, None, ["default"]) == ["default"]


def test_load_prompts_from_file_with_jsonl_and_truncate(text_generation, tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text('{"text": "json prompt"}\nraw prompt\n\nthird\n', encoding="utf-8")

    # JSONL `text` field is extracted; blank lines skipped; raw lines passed through.
    assert text_generation.load_prompts(None, str(prompt_file), None, ["d"]) == [
        "json prompt",
        "raw prompt",
        "third",
    ]
    # truncation caps the count
    assert text_generation.load_prompts(None, str(prompt_file), 2, ["d"]) == ["json prompt", "raw prompt"]


def test_load_prompts_truncates_only_file_prompts(text_generation, tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("file-1\nfile-2\nfile-3\n", encoding="utf-8")

    assert text_generation.load_prompts(["explicit"], str(prompt_file), 2, ["default"]) == [
        "explicit",
        "file-1",
        "file-2",
    ]


def test_validate_sequence_length(text_generation):
    # fits -> no raise
    text_generation.validate_sequence_length(longest_prompt_tokens=100, num_new_tokens=28, max_seq_length=4096)
    # exceeds -> raise
    with pytest.raises(ValueError, match="Longest prompt plus generation needs"):
        text_generation.validate_sequence_length(longest_prompt_tokens=4090, num_new_tokens=30, max_seq_length=4096)


def _parse_sampling_params(text_generation, cli_args):
    parser = text_generation.argparse.ArgumentParser()
    text_generation.add_sampling_args(parser)
    args = parser.parse_args(cli_args)
    return text_generation.build_sampling_params(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        skip_prompt_log_probs=args.skip_prompt_log_probs,
        num_tokens_to_generate=args.max_new_tokens,
        termination_id=args.termination_id,
        top_n_logprobs=args.top_n_logprobs,
        stop_words=args.stop_words,
    )


def test_top_p_sampling_is_compatible_with_default_cli_values(text_generation):
    params = _parse_sampling_params(text_generation, ["--top_p", "0.9"])
    logits = text_generation.torch.tensor([[1.0, 2.0, 3.0]])

    sampled = TorchSampling.sample_from_logits(
        logits,
        params.kwargs["temperature"],
        params.kwargs["top_k"],
        params.kwargs["top_p"],
        generator=text_generation.torch.Generator().manual_seed(0),
    )

    assert params.kwargs["top_k"] == 0
    assert sampled.shape == (1,)


def test_default_sampling_remains_greedy(text_generation):
    params = _parse_sampling_params(text_generation, [])
    logits = text_generation.torch.tensor([[1.0, 2.0, 3.0]])

    sampled = TorchSampling.sample_from_logits(
        logits,
        params.kwargs["temperature"],
        params.kwargs["top_k"],
        params.kwargs["top_p"],
        generator=text_generation.torch.Generator().manual_seed(0),
    )

    assert params.kwargs["top_k"] == 1
    assert sampled.item() == 2


def test_top_p_sampling_accepts_explicitly_disabled_top_k(text_generation):
    params = _parse_sampling_params(text_generation, ["--top_p", "0.9", "--top_k", "0"])

    assert params.kwargs["top_k"] == 0
    assert params.kwargs["top_p"] == 0.9


def test_sampling_rejects_positive_top_p_and_top_k(text_generation):
    with pytest.raises(ValueError, match="cannot both be positive"):
        _parse_sampling_params(text_generation, ["--top_p", "0.9", "--top_k", "2"])
