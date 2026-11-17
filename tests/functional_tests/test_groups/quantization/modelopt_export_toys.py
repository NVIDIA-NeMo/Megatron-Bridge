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

"""Tiny Hugging Face checkpoints of the models the ModelOpt export test covers."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
    NemotronHConfig,
    NemotronHForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from tests.functional_tests.test_groups.models.deepseek.test_deepseek_v4_conversion import _hf_to_bridge_state_dict


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
LINEAR_ATTENTION = {
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
}


def save_toy_model(name: str, path: Path) -> None:
    """Save the toy checkpoint ``name`` with a word-level tokenizer to ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    TOY_MODELS[name](path)
    vocab = json.loads((path / "config.json").read_text()).get("vocab_size", TOY_CONFIG["vocab_size"])
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<unk>": 0, **{f"token{i}": i for i in range(1, min(vocab, 256))}}, unk_token="<unk>")
        ),
        unk_token="<unk>",
    ).save_pretrained(path)


def _save_model(model: torch.nn.Module, path: Path) -> None:
    # Some HF initializers leave parameters uninitialized; random values keep calibration deterministic.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() == 1 and "norm" in name:
                param.fill_(1.0)
            else:
                param.copy_(torch.randn(param.shape) * 0.05)
    model.to(torch.bfloat16).save_pretrained(path)


def _save_qwen3(path: Path) -> None:
    _save_model(Qwen3ForCausalLM(Qwen3Config(**TOY_CONFIG)), path)


def _save_qwen3_5_vl_moe(path: Path) -> None:
    """GatedDeltaNet + full attention, routed and shared experts, and a vision tower."""
    config = Qwen3_5MoeConfig(
        tie_word_embeddings=False,
        text_config={
            **TOY_CONFIG,
            **LINEAR_ATTENTION,
            "dtype": "bfloat16",
            "layer_types": ["linear_attention", "full_attention"],
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 2,
            "out_hidden_size": TOY_CONFIG["hidden_size"],
            "num_position_embeddings": 64,
        },
    )
    _save_model(Qwen3_5MoeForConditionalGeneration(config), path)


def _save_nemotron_h(path: Path) -> None:
    config = NemotronHConfig(
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
    _save_model(NemotronHForCausalLM(config), path)


def _save_glm5_next(path: Path) -> None:
    """GLM-5.3-Flash: a KDA layer with a dense MLP, then a DSA (MLA) layer with routed and shared experts."""
    config = Glm5NextConfig(
        text_config={
            "dtype": "bfloat16",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "deepseek_sparse_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "indexer_types": ["full", "full"],
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "q_lora_rank": 64,
            "kv_lora_rank": 64,
            "qk_nope_head_dim": 32,
            "qk_rope_head_dim": 0,
            "v_head_dim": 32,
            "index_head_dim": 32,
            "index_n_heads": 4,
            "index_topk": 16,
            "index_kpool": 4,
            "linear_attn_config": {
                "num_heads": 4,
                "head_dim": 32,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "n_shared_experts": 1,
            "vocab_size": TOY_CONFIG["vocab_size"],
            "max_position_embeddings": TOY_CONFIG["max_position_embeddings"],
            "pad_token_id": 0,
            "scoring_func": "sigmoid",
            "num_nextn_predict_layers": 0,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 128,
            "projection_intermediate_size": 64,
        },
        image_token_id=250,
        video_token_id=251,
        image_start_token_id=252,
        image_end_token_id=253,
        video_start_token_id=254,
        video_end_token_id=255,
        tie_word_embeddings=False,
    )
    _save_model(Glm5NextForConditionalGeneration(config), path)


def _save_deepseek_v4(path: Path) -> None:
    """Window, compressed + indexer and compressed attention; a hash-routed MoE layer, then routed ones."""
    config = DeepseekV4Config(
        hidden_act="silu",
        hidden_size=128,
        head_dim=64,
        qk_rope_head_dim=16,
        max_position_embeddings=4096,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_attention_heads=4,
        num_experts_per_tok=2,
        num_hidden_layers=3,
        num_key_value_heads=1,
        num_nextn_predict_layers=0,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        compress_ratios=[0, 4, 128],
        num_hash_layers=1,
        sliding_window=64,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=32,
        hc_mult=4,
        hc_sinkhorn_iters=4,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        rope_theta=10000,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        # Matches the provider's vocabulary padding.
        vocab_size=1280,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    config.architectures = ["DeepseekV4ForCausalLM"]
    model = DeepseekV4ForCausalLM(config).to(torch.bfloat16)
    # Released checkpoints use their own layout, which the bridge reads.
    state = _hf_to_bridge_state_dict(model.state_dict(), config.num_hidden_layers)
    state = {name: tensor.to(torch.int32) if name.endswith(".tid2eid") else tensor for name, tensor in state.items()}
    (path / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    save_file(state, path / "model.safetensors")


# Kimi-K3 ships its config class with the checkpoint; this is the part the bridge reads.
_KIMI_K3_CONFIG_CLASS = """from transformers import PretrainedConfig


class KimiK3Config(PretrainedConfig):
    model_type = "kimi_k3"

    def __init__(self, text_config=None, **kwargs):
        self.text_config = PretrainedConfig(**text_config) if isinstance(text_config, dict) else text_config
        super().__init__(**kwargs)
"""


def _save_kimi_k3(path: Path) -> None:
    """A KDA layer with a dense MLP, then an MLA layer with a latent MoE, and attention residuals."""
    hidden, vocab, ffn = TOY_CONFIG["hidden_size"], TOY_CONFIG["vocab_size"], TOY_CONFIG["intermediate_size"]
    heads, head_dim, conv = 4, 32, 4
    q_lora, kv_lora, nope, rope, v_dim = 64, 32, 32, 16, 32
    experts, moe_ffn, latent = 4, 64, 64
    text_config = {
        "model_type": "kimi_linear",
        "dtype": "bfloat16",
        "architectures": ["KimiLinearForCausalLM"],
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": ffn,
        "num_hidden_layers": 2,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": TOY_CONFIG["max_position_embeddings"],
        "tie_word_embeddings": False,
        "q_lora_rank": q_lora,
        "kv_lora_rank": kv_lora,
        "qk_nope_head_dim": nope,
        "qk_rope_head_dim": rope,
        "v_head_dim": v_dim,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "num_experts": experts,
        "num_experts_per_token": 2,
        "moe_intermediate_size": moe_ffn,
        "num_shared_experts": 1,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "routed_scaling_factor": 1.0,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "use_grouped_topk": True,
        "num_expert_group": 1,
        "topk_group": 1,
        "routed_expert_hidden_size": latent,
        "latent_moe_use_norm": True,
        "attn_res_block_size": 12,
        "linear_attn_config": {
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": heads,
            "head_dim": head_dim,
            "short_conv_kernel_size": conv,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
    }
    (path / "configuration_kimi_k3.py").write_text(_KIMI_K3_CONFIG_CLASS)
    config = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "auto_map": {"AutoConfig": "configuration_kimi_k3.KimiK3Config"},
        "model_type": "kimi_k3",
        "dtype": "bfloat16",
        "tie_word_embeddings": False,
        "text_config": text_config,
    }
    (path / "config.json").write_text(json.dumps(config, indent=2))

    kda_dim = heads * head_dim
    kda = {
        **{f"self_attn.{name}_proj.weight": (kda_dim, hidden) for name in ("q", "k", "v", "g")},
        **{f"self_attn.{name}_conv1d.weight": (kda_dim, 1, conv) for name in ("q", "k", "v")},
        "self_attn.A_log": (heads,),
        "self_attn.dt_bias": (kda_dim,),
        "self_attn.f_a_proj.weight": (head_dim, hidden),
        "self_attn.f_b_proj.weight": (kda_dim, head_dim),
        "self_attn.b_proj.weight": (heads, hidden),
        "self_attn.o_norm.weight": (head_dim,),
        "self_attn.o_proj.weight": (hidden, kda_dim),
    }
    mla = {
        "self_attn.q_a_proj.weight": (q_lora, hidden),
        "self_attn.q_a_layernorm.weight": (q_lora,),
        "self_attn.q_b_proj.weight": (heads * (nope + rope), q_lora),
        "self_attn.kv_a_proj_with_mqa.weight": (kv_lora + rope, hidden),
        "self_attn.kv_a_layernorm.weight": (kv_lora,),
        "self_attn.kv_b_proj.weight": (heads * (nope + v_dim), kv_lora),
        "self_attn.g_proj.weight": (heads * v_dim, hidden),
        "self_attn.o_proj.weight": (hidden, heads * v_dim),
    }
    residuals = {
        **{f"{name}.weight": (hidden,) for name in ("input_layernorm", "post_attention_layernorm")},
        **{f"{name}_res_norm.weight": (hidden,) for name in ("self_attention", "mlp")},
        **{f"{name}_res_proj.weight": (1, hidden) for name in ("self_attention", "mlp")},
    }
    dense = {
        "mlp.gate_proj.weight": (ffn, hidden),
        "mlp.up_proj.weight": (ffn, hidden),
        "mlp.down_proj.weight": (hidden, ffn),
    }
    moe = {
        "block_sparse_moe.gate.weight": (experts, hidden),
        "block_sparse_moe.gate.e_score_correction_bias": (experts,),
        "block_sparse_moe.routed_expert_down_proj.weight": (latent, hidden),
        "block_sparse_moe.routed_expert_up_proj.weight": (hidden, latent),
        "block_sparse_moe.routed_expert_norm.weight": (latent,),
        "block_sparse_moe.shared_experts.gate_proj.weight": (moe_ffn, hidden),
        "block_sparse_moe.shared_experts.up_proj.weight": (moe_ffn, hidden),
        "block_sparse_moe.shared_experts.down_proj.weight": (hidden, moe_ffn),
        **{
            f"block_sparse_moe.experts.{expert}.{name}.weight": shape
            for expert in range(experts)
            for name, shape in (("w1", (moe_ffn, latent)), ("w3", (moe_ffn, latent)), ("w2", (latent, moe_ffn)))
        },
    }
    prefix = "language_model.model"
    shapes = {
        f"{prefix}.embed_tokens.weight": (vocab, hidden),
        "language_model.lm_head.weight": (vocab, hidden),
        f"{prefix}.norm.weight": (hidden,),
        f"{prefix}.output_attn_res_norm.weight": (hidden,),
        f"{prefix}.output_attn_res_proj.weight": (1, hidden),
        **{f"{prefix}.layers.0.{name}": shape for name, shape in {**residuals, **kda, **dense}.items()},
        **{f"{prefix}.layers.1.{name}": shape for name, shape in {**residuals, **mla, **moe}.items()},
    }
    float32 = ("A_log", "dt_bias", "conv1d.weight", "e_score_correction_bias")
    state = {}
    for name, shape in shapes.items():
        tensor = torch.randn(shape) * 0.05
        if name.endswith("norm.weight"):
            tensor += 1.0
        state[name] = tensor.to(torch.float32 if name.endswith(float32) else torch.bfloat16)
    save_file(state, path / "model.safetensors", metadata={"format": "pt"})


TOY_MODELS = {
    "qwen3": _save_qwen3,
    "qwen3_5_vl_moe": _save_qwen3_5_vl_moe,
    "nemotron_h": _save_nemotron_h,
    "glm5_next": _save_glm5_next,
    "deepseek_v4": _save_deepseek_v4,
    "kimi_k3": _save_kimi_k3,
}
