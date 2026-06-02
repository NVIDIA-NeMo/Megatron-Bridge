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

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from megatron.bridge.models.stepfun.configuration_step35 import Step35Config
from megatron.bridge.models.stepfun.step35_bridge import Step35Bridge


HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
MOE_INTERMEDIATE_SIZE = 64
NUM_ATTENTION_HEADS = 4
NUM_ATTENTION_GROUPS = 2
HEAD_DIM = 32
NUM_EXPERTS = 4
TOP_K = 2
VOCAB_SIZE = 512
NUM_LAYERS = 2


def _randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.bfloat16)


def _layer_common_state(layer_idx: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}"
    return {
        f"{prefix}.input_layernorm.weight": _randn(HIDDEN_SIZE),
        f"{prefix}.post_attention_layernorm.weight": _randn(HIDDEN_SIZE),
        f"{prefix}.self_attn.q_norm.weight": _randn(HEAD_DIM),
        f"{prefix}.self_attn.k_norm.weight": _randn(HEAD_DIM),
        f"{prefix}.self_attn.q_proj.weight": _randn(NUM_ATTENTION_HEADS * HEAD_DIM, HIDDEN_SIZE),
        f"{prefix}.self_attn.k_proj.weight": _randn(NUM_ATTENTION_GROUPS * HEAD_DIM, HIDDEN_SIZE),
        f"{prefix}.self_attn.v_proj.weight": _randn(NUM_ATTENTION_GROUPS * HEAD_DIM, HIDDEN_SIZE),
        f"{prefix}.self_attn.g_proj.weight": _randn(NUM_ATTENTION_HEADS, HIDDEN_SIZE),
        f"{prefix}.self_attn.o_proj.weight": _randn(HIDDEN_SIZE, NUM_ATTENTION_HEADS * HEAD_DIM),
    }


@pytest.fixture(scope="class")
def step35_toy_model_path(tmp_path_factory) -> str:
    """Create a tiny Step-3.5-style HF checkpoint for conversion tests."""
    # Importing Step35Bridge registers the config/model_type with the bridge registry.
    assert Step35Bridge.MODEL_TYPE == "step3p5"

    torch.manual_seed(1234)

    model_dir = tmp_path_factory.mktemp("step35_toy_model") / "step35_toy"
    model_dir.mkdir()

    config = Step35Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_attention_groups=NUM_ATTENTION_GROUPS,
        num_hidden_layers=NUM_LAYERS,
        max_seq_len=1024,
        vocab_size=VOCAB_SIZE,
        rms_norm_eps=1e-5,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
        moe_num_experts=NUM_EXPERTS,
        moe_top_k=TOP_K,
        rope_theta=10000.0,
        max_position_embeddings=1024,
        share_expert_dim=MOE_INTERMEDIATE_SIZE,
        head_dim=HEAD_DIM,
        layer_types=["full_attention", "full_attention"],
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_attention_groups": NUM_ATTENTION_GROUPS,
            "head_dim": HEAD_DIM,
            "true_head_dim": HEAD_DIM,
        },
        moe_layers_enum="1",
        num_nextn_predict_layers=0,
        torch_dtype="bfloat16",
        tie_word_embeddings=False,
        attention_output_gate=True,
        zero_centered=True,
        use_qk_norm=True,
        use_moe_router_bias=True,
        moe_router_activation="sigmoid",
        moe_router_scaling_factor=3.0,
        swiglu_limits=None,
        swiglu_limits_shared=None,
        need_fp32_gate=False,
        partial_rotary_factors=[1.0, 1.0],
    )
    config.architectures = ["Step3p5ForCausalLM"]
    config.save_pretrained(model_dir)

    state = {
        "model.embed_tokens.weight": _randn(VOCAB_SIZE, HIDDEN_SIZE),
        "lm_head.weight": _randn(VOCAB_SIZE, HIDDEN_SIZE),
        "model.norm.weight": _randn(HIDDEN_SIZE),
    }

    state.update(_layer_common_state(0))
    state.update(
        {
            "model.layers.0.mlp.gate_proj.weight": _randn(INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "model.layers.0.mlp.up_proj.weight": _randn(INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "model.layers.0.mlp.down_proj.weight": _randn(HIDDEN_SIZE, INTERMEDIATE_SIZE),
        }
    )

    state.update(_layer_common_state(1))
    state.update(
        {
            "model.layers.1.moe.gate.weight": _randn(NUM_EXPERTS, HIDDEN_SIZE),
            "model.layers.1.moe.router_bias": torch.zeros(NUM_EXPERTS, dtype=torch.float32),
            "model.layers.1.moe.gate_proj.weight": _randn(NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "model.layers.1.moe.up_proj.weight": _randn(NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "model.layers.1.moe.down_proj.weight": _randn(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
            "model.layers.1.share_expert.gate_proj.weight": _randn(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "model.layers.1.share_expert.up_proj.weight": _randn(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "model.layers.1.share_expert.down_proj.weight": _randn(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
        }
    )

    save_file(state, model_dir / "model.safetensors")
    return str(model_dir)


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    "tp,pp,ep,test_name",
    [
        (2, 1, 1, "TP"),
        (1, 2, 1, "PP"),
        (1, 1, 2, "EP"),
    ],
)
def test_step35_conversion_parallelism(step35_toy_model_path, tmp_path, tp, pp, ep, test_name):
    output_dir = tmp_path / f"step35_{test_name}"
    output_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
        "-m",
        "coverage",
        "run",
        f"--data-file={tmp_path / '.coverage'}",
        "--source=/opt/Megatron-Bridge/",
        "--parallel-mode",
        "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
        "--hf-model-id",
        step35_toy_model_path,
        "--output-dir",
        str(output_dir),
        "--tp",
        str(tp),
        "--pp",
        str(pp),
        "--ep",
        str(ep),
        "--skip-save",
    ]

    env = os.environ.copy()
    env["HF_HOME"] = str(tmp_path / "hf_home")
    env["MEGATRON_CONFIG_LOCK_DIR"] = str(tmp_path / "config_locks")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent.parent.parent.parent,
        env=env,
    )
    assert result.returncode == 0, (
        f"Step35 {test_name} conversion failed with return code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
