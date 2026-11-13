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

from tests.functional_tests.test_groups.quantization.modelopt_export_toys import save_toy_model


REPO_ROOT = Path(__file__).parents[4]


@pytest.fixture(scope="module")
def toy_models(tmp_path_factory):
    """Build each toy checkpoint the first time a test asks for it."""
    paths = {}

    def toy_model(name: str) -> Path:
        if name not in paths:
            torch.manual_seed(0)
            paths[name] = tmp_path_factory.mktemp(name)
            save_toy_model(name, paths[name])
        return paths[name]

    return toy_model


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


# The HF modules each toy leaves unquantized: embeddings, LM heads, routers, convolutions, the GatedDeltaNet
# beta / alpha gates, vision towers, DSA indexers, DeepSeek-V4 compressors and its einsum-only wo_a, and
# Kimi-K3's attention residuals; never norms. ModelOpt folds some of them into wildcards.
IGNORE = {
    "qwen3": {"lm_head", "model.embed_tokens"},
    "qwen3_5_vl_moe": {
        "lm_head",
        "model.language_model.embed_tokens",
        "model.visual*",
        *{f"model.language_model.layers.0.linear_attn.{name}" for name in ("conv1d", "in_proj_b", "in_proj_a")},
        *{
            f"model.language_model.layers.{layer}.mlp.{gate}"
            for layer in range(2)
            for gate in ("gate", "shared_expert_gate")
        },
    },
    "nemotron_h": {
        "lm_head",
        "backbone.embeddings",
        *{f"backbone.layers.{layer}.mixer.conv1d" for layer in (0, 3)},
        "backbone.layers.2.mixer.gate",
    },
    "glm5_next": {
        "lm_head",
        "model.language_model.embed_tokens",
        "model.visual*",
        *{f"model.language_model.layers.0.self_attn.{name}_conv1d" for name in ("q", "k", "v")},
        "model.language_model.layers.1.mlp.gate",
        "model.language_model.layers.1.self_attn.indexer*",
    },
    "deepseek_v4": {
        "embed",
        "head",
        *{f"layers.{layer}.{name}" for layer in range(3) for name in ("attn.wo_a", "ffn.gate")},
        *{f"layers.{layer}.attn.compressor*" for layer in (1, 2)},
        "layers.1.attn.indexer*",
    },
    "kimi_k3": {
        "language_model.lm_head",
        "language_model.model.embed_tokens",
        "language_model.model.output_attn_res_proj",
        *{
            f"language_model.model.layers.{layer}.{name}_res_proj"
            for layer in range(2)
            for name in ("self_attention", "mlp")
        },
        *{f"language_model.model.layers.0.self_attn.{name}_conv1d" for name in ("q", "k", "v")},
        "language_model.model.layers.1.block_sparse_moe.gate",
    },
}
# A quantized projection per toy, preferably one whose layout the export has to rebuild.
PROJECTION = {
    "qwen3": "model.layers.0.mlp.down_proj",
    "qwen3_5_vl_moe": "model.language_model.layers.0.mlp.experts.0.down_proj",
    "nemotron_h": "backbone.layers.0.mixer.in_proj",
    # The KDA in_proj, split back into q / k / v.
    "glm5_next": "model.language_model.layers.0.self_attn.q_proj",
    "deepseek_v4": "layers.0.ffn.experts.0.w2",
    "kimi_k3": "language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj",
}


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    ("model", "config", "parallelisms"),
    [
        # Nemotron-H covers TP, PP and EP, Qwen3 with Qwen3.5-VL TP, PP and ETP; the other models one or two of them.
        ("qwen3", "fp8", [("--tp", "2")]),
        ("qwen3_5_vl_moe", "mixed", [("--pp", "2"), ("--etp", "2")]),
        ("nemotron_h", "nvfp4", [("--tp", "2"), ("--pp", "2"), ("--ep", "2")]),
        ("glm5_next", "nvfp4", [("--ep", "2")]),
        # DeepSeek-V4 attention supports TP=1 only.
        ("deepseek_v4", "nvfp4", [("--pp", "2")]),
        ("kimi_k3", "nvfp4", [("--tp", "2")]),
    ],
)
def test_quantized_export_is_independent_of_parallelism(toy_models, tmp_path, model, config, parallelisms):
    """Every parallelism writes the single-GPU checkpoint bit for bit, with a deployable quantization config."""
    # ModelOpt 0.48 predates its FP8 rounding fix; FP8 bit parity remains deferred above.
    offline_out = tmp_path / "offline" if model == "nemotron_h" and config == "nvfp4" else None
    _export(toy_models(model), tmp_path / "reference", config, offline_out=offline_out)
    weights, hf_config = _load(tmp_path / "reference")

    quantization = hf_config["quantization_config"]
    assert quantization["quant_method"] == "modelopt"
    assert quantization["quant_algo"] == {"nvfp4": "NVFP4", "fp8": "FP8", "mixed": "MIXED_PRECISION"}[config]
    assert set(quantization["ignore"]) == IGNORE[model]
    projection = PROJECTION[model]
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
        _export(toy_models(model), out, config, *parallelism)
        other_weights, other_config = _load(out)
        assert other_config == hf_config, parallelism
        assert other_weights.keys() == weights.keys(), parallelism
        for name, tensor in weights.items():
            assert other_weights[name].dtype == tensor.dtype, (parallelism, name)
            assert torch.equal(_bits(other_weights[name]), _bits(tensor)), (parallelism, name)
