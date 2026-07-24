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

"""Functional toy-model conversion tests for DeepSeek V4."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


def _has_dsv4_in_transformers() -> bool:
    try:
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM  # noqa: F401

        return True
    except Exception:
        return False


def _has_dsv4_in_mcore() -> bool:
    try:
        return all(
            importlib.util.find_spec(mod) is not None
            for mod in (
                "megatron.core.transformer.hyper_connection",
                "megatron.core.transformer.experimental_attention_variant.csa",
                "megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention",
            )
        )
    except ModuleNotFoundError:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _has_dsv4_in_transformers(),
        reason="transformers does not yet ship DeepseekV4ForCausalLM (HF hub only via trust_remote_code).",
    ),
    pytest.mark.skipif(
        not _has_dsv4_in_mcore(),
        reason="megatron-core does not yet ship DSv4 prerequisites (PRs #3430 / #4458 / #4481 / #4518).",
    ),
]


# Toy config tuned to satisfy DSv4 invariants at minimum size:
#   - len(compress_ratios) == num_hidden_layers + num_nextn_predict_layers
#   - sliding_window <= max_position_embeddings
#   - vocab_size large enough for hash routing and divisible by the DSv4 provider's 1280 vocab padding
#   - n_routed_experts divisible by num_experts_per_tok and the EP sizes we test
HF_DEEPSEEK_V4_TOY_MODEL_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "head_dim": 256,
    "qk_rope_head_dim": 32,
    "intermediate_size": 2048,
    "max_position_embeddings": 4096,
    "moe_intermediate_size": 512,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_attention_heads": 16,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 4,
    "num_nextn_predict_layers": 0,  # disable MTP for the toy
    "q_lora_rank": 256,
    "o_lora_rank": 256,
    "o_groups": 4,
    "compress_ratios": [0, 4, 4, 4],  # 4 entries == num_hidden_layers (mtp=0)
    "sliding_window": 64,
    "index_n_heads": 4,
    "index_head_dim": 32,
    "index_topk": 32,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 4,
    "norm_topk_prob": True,
    "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 1.0,
    "rope_theta": 10000,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "vocab_size": 12800,
    "torch_dtype": "bfloat16",
}


def _hf_to_bridge_state_dict(hf_state_dict: dict, num_layers: int) -> dict:
    """Translate native Transformers DSv4 parameter names to the released checkpoint layout."""
    bridge_state = {}

    def copy(src: str, dst: str) -> None:
        if src in hf_state_dict:
            bridge_state[dst] = hf_state_dict[src].detach().cpu().contiguous()

    copy("model.embed_tokens.weight", "embed.weight")
    copy("lm_head.weight", "head.weight")
    copy("model.norm.weight", "norm.weight")
    copy("model.hc_head.hc_fn", "hc_head_fn")
    copy("model.hc_head.hc_base", "hc_head_base")
    copy("model.hc_head.hc_scale", "hc_head_scale")

    for layer_idx in range(num_layers):
        hf_prefix = f"model.layers.{layer_idx}"
        ckpt_prefix = f"layers.{layer_idx}"

        copy(f"{hf_prefix}.input_layernorm.weight", f"{ckpt_prefix}.attn_norm.weight")
        copy(f"{hf_prefix}.post_attention_layernorm.weight", f"{ckpt_prefix}.ffn_norm.weight")
        copy(f"{hf_prefix}.self_attn.q_a_proj.weight", f"{ckpt_prefix}.attn.wq_a.weight")
        copy(f"{hf_prefix}.self_attn.q_a_norm.weight", f"{ckpt_prefix}.attn.q_norm.weight")
        copy(f"{hf_prefix}.self_attn.q_b_proj.weight", f"{ckpt_prefix}.attn.wq_b.weight")
        copy(f"{hf_prefix}.self_attn.kv_proj.weight", f"{ckpt_prefix}.attn.wkv.weight")
        copy(f"{hf_prefix}.self_attn.kv_norm.weight", f"{ckpt_prefix}.attn.kv_norm.weight")
        copy(f"{hf_prefix}.self_attn.o_a_proj.weight", f"{ckpt_prefix}.attn.wo_a.weight")
        copy(f"{hf_prefix}.self_attn.o_b_proj.weight", f"{ckpt_prefix}.attn.wo_b.weight")
        copy(f"{hf_prefix}.self_attn.sinks", f"{ckpt_prefix}.attn.attn_sink")

        compressor_prefix = f"{hf_prefix}.self_attn.compressor"
        copy(f"{compressor_prefix}.kv_proj.weight", f"{ckpt_prefix}.attn.compressor.wkv.weight")
        copy(f"{compressor_prefix}.gate_proj.weight", f"{ckpt_prefix}.attn.compressor.wgate.weight")
        copy(f"{compressor_prefix}.position_bias", f"{ckpt_prefix}.attn.compressor.ape")
        copy(f"{compressor_prefix}.kv_norm.weight", f"{ckpt_prefix}.attn.compressor.norm.weight")

        indexer_prefix = f"{compressor_prefix}.indexer"
        copy(f"{indexer_prefix}.q_b_proj.weight", f"{ckpt_prefix}.attn.indexer.wq_b.weight")
        # HF transformers nests the indexer score projection under `.scorer.`.
        copy(f"{indexer_prefix}.scorer.weights_proj.weight", f"{ckpt_prefix}.attn.indexer.weights_proj.weight")
        copy(f"{indexer_prefix}.kv_proj.weight", f"{ckpt_prefix}.attn.indexer.compressor.wkv.weight")
        copy(f"{indexer_prefix}.gate_proj.weight", f"{ckpt_prefix}.attn.indexer.compressor.wgate.weight")
        copy(f"{indexer_prefix}.position_bias", f"{ckpt_prefix}.attn.indexer.compressor.ape")
        copy(f"{indexer_prefix}.kv_norm.weight", f"{ckpt_prefix}.attn.indexer.compressor.norm.weight")

        copy(f"{hf_prefix}.mlp.gate.weight", f"{ckpt_prefix}.ffn.gate.weight")
        if f"{hf_prefix}.mlp.gate.e_score_correction_bias" in hf_state_dict:
            copy(f"{hf_prefix}.mlp.gate.e_score_correction_bias", f"{ckpt_prefix}.ffn.gate.bias")
        copy(f"{hf_prefix}.mlp.gate.tid2eid", f"{ckpt_prefix}.ffn.gate.tid2eid")

        gate_up = hf_state_dict[f"{hf_prefix}.mlp.experts.gate_up_proj"].detach().cpu()
        gate, up = gate_up.chunk(2, dim=1)
        down = hf_state_dict[f"{hf_prefix}.mlp.experts.down_proj"].detach().cpu()
        for expert_idx in range(gate.shape[0]):
            bridge_state[f"{ckpt_prefix}.ffn.experts.{expert_idx}.w1.weight"] = gate[expert_idx].contiguous()
            bridge_state[f"{ckpt_prefix}.ffn.experts.{expert_idx}.w3.weight"] = up[expert_idx].contiguous()
            bridge_state[f"{ckpt_prefix}.ffn.experts.{expert_idx}.w2.weight"] = down[expert_idx].contiguous()

        copy(f"{hf_prefix}.mlp.shared_experts.gate_proj.weight", f"{ckpt_prefix}.ffn.shared_experts.w1.weight")
        copy(f"{hf_prefix}.mlp.shared_experts.up_proj.weight", f"{ckpt_prefix}.ffn.shared_experts.w3.weight")
        copy(f"{hf_prefix}.mlp.shared_experts.down_proj.weight", f"{ckpt_prefix}.ffn.shared_experts.w2.weight")
        copy(f"{hf_prefix}.attn_hc.fn", f"{ckpt_prefix}.hc_attn_fn")
        copy(f"{hf_prefix}.attn_hc.base", f"{ckpt_prefix}.hc_attn_base")
        copy(f"{hf_prefix}.attn_hc.scale", f"{ckpt_prefix}.hc_attn_scale")
        copy(f"{hf_prefix}.ffn_hc.fn", f"{ckpt_prefix}.hc_ffn_fn")
        copy(f"{hf_prefix}.ffn_hc.base", f"{ckpt_prefix}.hc_ffn_base")
        copy(f"{hf_prefix}.ffn_hc.scale", f"{ckpt_prefix}.hc_ffn_scale")

    return bridge_state


class TestDeepSeekV4Conversion:
    """Toy HF-to-Megatron roundtrip coverage for DeepSeek V4."""

    @pytest.fixture(scope="class")
    def deepseek_v4_toy_model_path(self, tmp_path_factory):
        import torch
        from safetensors.torch import save_file
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, PreTrainedTokenizerFast

        temp_dir = tmp_path_factory.mktemp("deepseek_v4_toy_model")
        model_dir = temp_dir / "deepseek_v4_toy"
        model_dir.mkdir()

        torch.manual_seed(1234)
        config = DeepseekV4Config(**HF_DEEPSEEK_V4_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16
        model = DeepseekV4ForCausalLM(config).bfloat16()
        bridge_state = _hf_to_bridge_state_dict(model.state_dict(), config.num_hidden_layers)
        for key in list(bridge_state):
            if key.endswith(".tid2eid"):
                bridge_state[key] = bridge_state[key].to(torch.int32)

        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        vocab.update({f"tok_{idx}": idx for idx in range(4, 128)})
        tokenizer_model = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer_model.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_model,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        config.save_pretrained(model_dir)
        with open(model_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        save_file(bridge_state, model_dir / "model.safetensors")
        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    def test_deepseek_v4_roundtrip_ep(self, deepseek_v4_toy_model_path, tmp_path):
        test_output_dir = tmp_path / "deepseek_v4_ep"
        test_output_dir.mkdir(exist_ok=True)

        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            deepseek_v4_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            "1",
            "--pp",
            "1",
            "--ep",
            "2",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent.parent
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        assert result.returncode == 0, f"DeepSeek V4 conversion failed with {result.returncode}"

        converted_dir = test_output_dir / Path(deepseek_v4_toy_model_path).name
        assert (converted_dir / "config.json").exists()
        assert list(converted_dir.glob("*.safetensors"))
