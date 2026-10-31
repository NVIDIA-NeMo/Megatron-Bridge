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

"""ModelOpt quantized Hugging Face export: the same checkpoint at every parallelism."""

# TODO: Once nvidia-modelopt is bumped to 0.49.0rc0, add tests for its FP8 rounding fix
# (NVIDIA/Model-Optimizer#3005) and KV-cache export (NVIDIA/Model-Optimizer#2981): parity with
# ModelOpt's export_mcore_gpt_to_hf, and k_scale / v_scale with kv_cache_quant_algo.

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    NemotronHConfig,
    NemotronHForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)


REPO_ROOT = Path(__file__).parents[4]
TOY_CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "vocab_size": 256,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
}


@pytest.fixture(scope="module")
def toy_models(tmp_path_factory) -> dict[str, Path]:
    paths = {}
    torch.manual_seed(0)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<unk>": 0, **{f"token{i}": i for i in range(1, TOY_CONFIG["vocab_size"])}}, unk_token="<unk>")
        ),
        unk_token="<unk>",
    )
    for name, model in (
        ("qwen3", Qwen3ForCausalLM(Qwen3Config(**TOY_CONFIG))),
        (
            "qwen3_moe",
            Qwen3MoeForCausalLM(
                Qwen3MoeConfig(moe_intermediate_size=64, num_experts=4, num_experts_per_tok=2, **TOY_CONFIG)
            ),
        ),
        (
            "qwen3_5",
            Qwen3_5ForCausalLM(
                Qwen3_5TextConfig(
                    layer_types=["linear_attention", "full_attention"],
                    linear_num_key_heads=2,
                    linear_num_value_heads=4,
                    linear_key_head_dim=32,
                    linear_value_head_dim=32,
                    **TOY_CONFIG,
                )
            ),
        ),
        (
            "nemotron_h",
            NemotronHForCausalLM(
                NemotronHConfig(
                    **{
                        **TOY_CONFIG,
                        "num_hidden_layers": 4,
                        "layers_block_type": ["mamba", "attention", "moe", "mamba"],
                        "mamba_num_heads": 8,
                        "mamba_head_dim": 32,
                        "ssm_state_size": 16,
                        "n_groups": 2,
                        "n_routed_experts": 4,
                        "num_experts_per_tok": 2,
                        "moe_intermediate_size": 64,
                        "moe_shared_expert_intermediate_size": 128,
                        "num_nextn_predict_layers": 0,
                    }
                )
            ),
        ),
    ):
        paths[name] = tmp_path_factory.mktemp(name)
        model.to(torch.bfloat16).save_pretrained(paths[name])
        tokenizer.save_pretrained(paths[name])
    return paths


def _export(model: Path, out: Path, config: str, *parallelism: str, offline_out: Path | None = None) -> None:
    num_gpus = 1
    for size in parallelism[1::2]:
        num_gpus *= int(size)
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={num_gpus}"]
    # Record the workers' coverage too; CI combines and uploads REPO_ROOT/.coverage.
    cmd += ["-m", "coverage", "run", f"--data-file={REPO_ROOT / '.coverage'}", f"--source={REPO_ROOT}"]
    cmd += ["--parallel-mode", "tests/functional_tests/test_groups/quantization/modelopt_hf_export.py"]
    cmd += ["--hf-model-path", str(model), "--out", str(out), "--config", config, *parallelism]
    if offline_out is not None:
        cmd += ["--offline-out", str(offline_out)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=600)
    assert result.returncode == 0, f"{cmd} failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"


def _load(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    weights = {}
    for shard in path.glob("*.safetensors"):
        with safe_open(shard, framework="pt") as f:
            weights.update({name: f.get_tensor(name) for name in f.keys()})
    return weights, json.loads((path / "config.json").read_text())


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    ("model", "config", "parallelisms"),
    [
        ("qwen3", "nvfp4", [("--tp", "2"), ("--pp", "2")]),
        ("qwen3", "fp8", [("--tp", "2")]),
        ("qwen3_moe", "mixed", [("--ep", "2"), ("--etp", "2"), ("--pp", "2")]),
        ("qwen3_5", "nvfp4", [("--tp", "2"), ("--pp", "2")]),
        ("nemotron_h", "nvfp4", [("--tp", "2"), ("--pp", "2")]),
    ],
)
def test_quantized_export_is_independent_of_parallelism(toy_models, tmp_path, model, config, parallelisms):
    """Every parallelism writes the single-GPU checkpoint bit for bit, with a deployable quantization config."""
    # ModelOpt 0.48 predates its FP8 rounding fix; FP8 bit parity remains deferred above.
    offline_out = tmp_path / "offline" if model == "nemotron_h" and config == "nvfp4" else None
    _export(toy_models[model], tmp_path / "reference", config, offline_out=offline_out)
    weights, hf_config = _load(tmp_path / "reference")

    quantization = hf_config["quantization_config"]
    assert quantization["quant_method"] == "modelopt"
    assert quantization["quant_algo"] == {"nvfp4": "NVFP4", "fp8": "FP8", "mixed": "MIXED_PRECISION"}[config]
    # Only unquantized linear layers are excluded: the embedding, LM head, MoE routers and, in GatedDeltaNet,
    # the conv1d and the beta / alpha gates fused into the quantized in_proj; never norms.
    unquantized = {
        "qwen3_moe": {f"model.layers.{layer}.mlp.gate" for layer in range(2)},
        "qwen3_5": {f"model.layers.0.linear_attn.{name}" for name in ("conv1d", "in_proj_b", "in_proj_a")},
        "nemotron_h": {
            *{f"backbone.layers.{layer}.mixer.conv1d" for layer in (0, 3)},
            "backbone.layers.2.mixer.gate",
        },
    }.get(model, set())
    embedding = "backbone.embeddings" if model == "nemotron_h" else "model.embed_tokens"
    assert set(quantization["ignore"]) == {"lm_head", embedding, *unquantized}
    projection = {
        "qwen3_moe": "model.layers.0.mlp.experts.0.down_proj",
        "qwen3_5": "model.layers.0.linear_attn.in_proj_qkv",
        "nemotron_h": "backbone.layers.0.mixer.in_proj",
    }.get(model, "model.layers.0.mlp.down_proj")
    packed_dtype = torch.float8_e4m3fn if config == "fp8" else torch.uint8
    assert weights[f"{projection}.weight"].dtype == packed_dtype
    assert {f"{projection}.weight_scale", f"{projection}.input_scale"} <= weights.keys()
    assert not any("quantizer" in name or name.endswith("_amax") for name in weights)
    assert not (tmp_path / "reference" / "hf_quant_config.json").exists()

    if offline_out is not None:
        offline_weights, offline_config = _load(offline_out)
        assert offline_config["quantization_config"]["quant_algo"] == quantization["quant_algo"]
        for projection in ("in_proj", "out_proj"):
            prefix = f"backbone.layers.0.mixer.{projection}."
            keys = {name for name in weights if name.startswith(prefix)}
            assert {prefix + "weight", prefix + "weight_scale", prefix + "input_scale"} <= keys
            assert {name for name in offline_weights if name.startswith(prefix)} == keys
            for name in keys:
                assert offline_weights[name].dtype == weights[name].dtype, name
                assert torch.equal(_bits(offline_weights[name]), _bits(weights[name])), name

    for parallelism in parallelisms:
        out = tmp_path / "_".join(parallelism).strip("-")
        _export(toy_models[model], out, config, *parallelism)
        other_weights, other_config = _load(out)
        assert other_config == hf_config, parallelism
        assert other_weights.keys() == weights.keys(), parallelism
        for name, tensor in weights.items():
            assert other_weights[name].dtype == tensor.dtype, (parallelism, name)
            assert torch.equal(_bits(other_weights[name]), _bits(tensor)), (parallelism, name)
