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

"""ModelOpt fake-quant Hugging Face export: quantizer sidecars at TP / PP / EP and weight-only folding."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from tests.functional_tests.test_groups.quantization.fakequant_hf_export import expected_amax


NUM_LAYERS, NUM_EXPERTS = 2, 4
REPO_ROOT = Path(__file__).parents[4]


@pytest.fixture(scope="module")
def toy_qwen3_moe(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("qwen3_moe_toy")
    config = Qwen3MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        vocab_size=256,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    return path


def _export(model: Path, out: Path, *args: str) -> str:
    num_gpus = 1
    for flag in ("--tp", "--pp", "--ep"):
        if flag in args:
            num_gpus *= int(args[args.index(flag) + 1])
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={num_gpus}"]
    cmd += ["tests/functional_tests/test_groups/quantization/fakequant_hf_export.py"]
    cmd += ["--hf-model-path", str(model), "--out", str(out), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=600)
    assert result.returncode == 0, f"{cmd} failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
    return result.stdout


def _expected_ranges(*, grouped: bool, ep: int) -> dict[str, float]:
    """Every enabled quantizer's HF ``_amax`` name and the range the export script pinned to it."""
    expected = {}
    for layer in range(NUM_LAYERS):
        attention = f"model.layers.{layer}.self_attn"
        for proj in ("q", "k", "v"):
            expected[f"{attention}.{proj}_proj.input_quantizer._amax"] = expected_amax(layer, "linear_qkv")
        expected[f"{attention}.o_proj.input_quantizer._amax"] = expected_amax(layer, "linear_proj")
        for kind in ("k_bmm", "v_bmm"):
            expected[f"{attention}.{kind}_quantizer._amax"] = expected_amax(layer, kind)
        for expert in range(NUM_EXPERTS):
            # Grouped GEMM has one quantizer per EP rank; SequentialMLP one per expert.
            source = expert // (NUM_EXPERTS // ep) if grouped else expert
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            for proj, kind in (("gate", "linear_fc1"), ("up", "linear_fc1"), ("down", "linear_fc2")):
                expected[f"{prefix}.{proj}_proj.input_quantizer._amax"] = expected_amax(layer, kind, source)
    return expected


def _weights(path: Path) -> dict[str, torch.Tensor]:
    weights = {}
    for shard in path.glob("*.safetensors"):
        with safe_open(shard, framework="pt") as f:
            weights.update({name: f.get_tensor(name) for name in f.keys()})
    return weights


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    ("parallelism", "grouped"),
    [
        ((), False),
        (("--tp", "2"), False),
        (("--pp", "2"), False),
        (("--ep", "2"), False),
        (("--ep", "2"), True),
    ],
    ids=["tp1", "tp2", "pp2", "ep2_sequential", "ep2_grouped"],
)
def test_activation_quantizer_sidecars(toy_qwen3_moe, tmp_path, parallelism, grouped):
    """Each enabled quantizer is exported once per HF name, with its range and ModelOpt state."""
    out = tmp_path / "export"
    stdout = _export(toy_qwen3_moe, out, *parallelism, *(["--grouped"] if grouped else []))
    if not parallelism:
        assert "Unexportable quantizer state rejected" in stdout

    ep = int(parallelism[1]) if parallelism and parallelism[0] == "--ep" else 1
    expected = _expected_ranges(grouped=grouped, ep=ep)
    state = torch.load(out / "quantizer_state.pth", weights_only=True)
    assert state.keys() == expected.keys()
    for name, amax in expected.items():
        assert state[name].dtype == torch.float32, name
        assert state[name].item() == pytest.approx(amax, rel=0, abs=1e-6), name

    recipe = yaml.safe_load((out / "quant_recipe.yaml").read_text())
    assert (recipe["format"], recipe["version"]) == ("modelopt_quantizer_recipe", 1)
    quantizers = recipe["quantizers"]
    assert quantizers.keys() == {name.removesuffix("._amax") for name in expected}
    for name, quantizer_state in quantizers.items():
        assert quantizer_state["_disabled"] is False
        assert quantizer_state["_num_bits"] == ([4, 3] if "_bmm_" in name else [2, 1]), name

    assert not any("quantizer" in name for name in _weights(out))
    assert not (out / "modelopt_weights.pt").exists()


@pytest.mark.run_only_on("GPU")
def test_folded_weight_only_export_has_no_sidecars(toy_qwen3_moe, tmp_path):
    """Folded weight-only quantization leaves a plain checkpoint with INT8-grid weights."""
    out = tmp_path / "export"
    _export(toy_qwen3_moe, out, "--mode", "weights")

    assert not (out / "quantizer_state.pth").exists()
    assert not (out / "quant_recipe.yaml").exists()
    weights = _weights(out)
    assert weights.keys() == _weights(toy_qwen3_moe).keys()
    # Per-output-channel INT8 fake quantization leaves at most 255 distinct values per row.
    folded = weights["model.layers.0.self_attn.q_proj.weight"].float()
    assert max(len(row.unique()) for row in folded) <= 255
    assert json.loads((out / "config.json").read_text())["num_experts"] == NUM_EXPERTS
