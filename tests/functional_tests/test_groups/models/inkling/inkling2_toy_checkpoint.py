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

"""Write a randomly initialized toy Inkling2 checkpoint in the released on-disk layout.

Inkling2 has no Hugging Face modeling class, so the toy is written tensor by tensor.
It keeps every component of the released checkpoint at reduced sizes: two AttnRes
blocks of two layers, each ending in a full-attention layer; NVFP4 routed experts
in the first block and MXFP8 routed experts in the second, with their scale
tensors; the FP32 AttnRes queries and router biases; the separate output layer;
and the image encoder with its projection. A small word-level tokenizer stands in
for the released one, so a Megatron import can save its tokenizer assets offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


HIDDEN = 64
NUM_LAYERS = 4
BLOCK_SIZE = 2
FULL_ATTENTION_LAYERS = (1, 3)
NVFP4_LAYERS = (0, 1)
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
REL_DIM = 8
REL_EXTENT = 32
WINDOW = 16
CONV_WIDTH = 4
VOCAB = 512
UNPADDED_VOCAB = 500
EXPERTS = 4
TOP_K = 2
EXPERT_HIDDEN = 64
VISION = {
    "temporal_patch_size": 2,
    "native_patch_size": 4,
    "adapter_merge_size": 2,
    "projection_hidden_dim": 48,
    "projector_bias": False,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "mlp_ratio": 4.0,
    "posemb_grid_size": 4,
    "layer_norm_eps": 1e-05,
}
TEXT_CONFIG = {
    "model_max_length": 4096,
    "torch_dtype": "bfloat16",
    "hidden_size": HIDDEN,
    "num_hidden_layers": NUM_LAYERS,
    "vocab_size": VOCAB,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "d_rel": REL_DIM,
    "rel_extent": REL_EXTENT,
    "q_bias": False,
    "o_bias": False,
    "log_scaling_n_floor": 64,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-06,
    "pre_norm_pre_affine": False,
    "final_norm_pre_affine": False,
    "use_value_norm_no_gamma": True,
    "qk_l2_clip_threshold": 22.627416997969522,
    "rel_l2_clip_threshold": 8.0,
    "local_layer_ids": [i for i in range(NUM_LAYERS) if i not in FULL_ATTENTION_LAYERS],
    "sconv_kernel_size": CONV_WIDTH,
    "unpadded_vocab_size": UNPADDED_VOCAB,
    "final_logit_softcapping": 30.0,
    "swa_head_dim": HEAD_DIM,
    "swa_num_attention_heads": HEADS,
    "swa_num_key_value_heads": KV_HEADS,
    "sliding_window_size": WINDOW,
    "n_layers_per_attn_res_block": BLOCK_SIZE,
    "attn_res_normalize_values": True,
    "n_routed_experts": EXPERTS,
    "num_experts_per_tok": TOP_K,
    "n_shared_experts": 0,
    "intermediate_size": EXPERT_HIDDEN,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "use_gate_bias": True,
}


def _bf16(generator: torch.Generator, *shape: int, std: float = 0.05, mean: float = 0.0) -> torch.Tensor:
    return (torch.randn(*shape, generator=generator) * std + mean).to(torch.bfloat16)


def _fp32(generator: torch.Generator, *shape: int, std: float = 0.05) -> torch.Tensor:
    return torch.randn(*shape, generator=generator) * std


def _e4m3_scales(generator: torch.Generator, *shape: int) -> torch.Tensor:
    """Positive, normal FP8 E4M3 block scales."""
    return torch.exp2(torch.rand(*shape, generator=generator) * 4 - 2).to(torch.float8_e4m3fn)


def _nvfp4_experts(generator: torch.Generator, prefix: str, rows: int, columns: int) -> dict[str, torch.Tensor]:
    """NVFP4 storage: two E2M1 codes per byte, an E4M3 scale per 16 values, and a per-expert amax."""
    return {
        prefix: torch.randint(0, 256, (EXPERTS, rows, columns // 2), generator=generator, dtype=torch.uint8),
        f"{prefix}_scale": _e4m3_scales(generator, EXPERTS, rows, columns // 16),
        f"{prefix}_amax": torch.rand(EXPERTS, generator=generator) + 1.0,
        f"{prefix}_outer_scale_denominator": torch.tensor([6.0 * 448.0]),
    }


def _mxfp8_experts(generator: torch.Generator, prefix: str, rows: int, columns: int) -> dict[str, torch.Tensor]:
    """MXFP8 storage: E4M3 values and a UE8M0 power-of-two scale per 32 values."""
    return {
        prefix: (torch.randn(EXPERTS, rows, columns, generator=generator) * 64)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn),
        f"{prefix}_scale": torch.randint(118, 124, (EXPERTS, rows, columns // 32), generator=generator).to(
            torch.uint8
        ),
    }


def _decoder_layer(generator: torch.Generator, layer: int) -> dict[str, torch.Tensor]:
    full = layer in FULL_ATTENTION_LAYERS
    query_width, kv_width = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    attn = {
        "wq_du.weight": _bf16(generator, query_width, HIDDEN),
        "wk_dv.weight": _bf16(generator, kv_width, HIDDEN),
        "wv_dv.weight": _bf16(generator, kv_width, HIDDEN),
        "wr_du.weight": _bf16(generator, HEADS * REL_DIM, HIDDEN),
        "gate_attn.weight": _bf16(generator, query_width, HIDDEN),
        "attn_sink_proj.weight": _bf16(generator, HEADS, HIDDEN),
        "wo_ud.weight": _bf16(generator, HIDDEN, query_width),
        "q_l2_clip.scale": _bf16(generator, HEADS, HEAD_DIM, mean=1.0),
        "rel_l2_clip.scale": _bf16(generator, HEADS, REL_DIM, mean=1.0),
        "rel_logits_proj.proj": _bf16(generator, REL_DIM, REL_EXTENT if full else WINDOW, std=0.5),
    }
    if full:
        for name in ("k_sconv", "v_sconv"):
            attn[f"{name}.static_w"] = _bf16(generator, CONV_WIDTH, kv_width, std=0.3)
            attn[f"{name}.weight_proj"] = _bf16(generator, KV_HEADS * CONV_WIDTH, HIDDEN)
    else:
        for name in ("k_sconv", "v_sconv"):
            attn[f"{name}.weight"] = _bf16(generator, kv_width, 1, CONV_WIDTH, std=0.3)
    tensors = {f"attn.{name}": tensor for name, tensor in attn.items()}
    if full:
        for name in ("q_input_norm", "k_input_norm", "v_input_norm"):
            tensors[f"{name}.weight"] = _bf16(generator, HIDDEN, mean=1.0)
    else:
        tensors["attn_norm.weight"] = _bf16(generator, HIDDEN, mean=1.0)
    tensors["attn_sconv.weight"] = _bf16(generator, HIDDEN, 1, CONV_WIDTH, std=0.3)
    tensors["mlp_norm.weight"] = _bf16(generator, HIDDEN, mean=1.0)
    tensors["mlp_sconv.weight"] = _bf16(generator, HIDDEN, 1, CONV_WIDTH, std=0.3)
    tensors["mlp.gate.weight"] = _bf16(generator, EXPERTS, HIDDEN)
    tensors["mlp.gate.bias"] = _fp32(generator, EXPERTS)
    # w13 interleaves gate and up rows; w2 is the down projection.
    experts = _nvfp4_experts if layer in NVFP4_LAYERS else _mxfp8_experts
    tensors.update(experts(generator, "mlp.experts.w13_weight", 2 * EXPERT_HIDDEN, HIDDEN))
    tensors.update(experts(generator, "mlp.experts.w2_weight", HIDDEN, EXPERT_HIDDEN))
    return {f"model.llm.layers.{layer}.{name}": tensor for name, tensor in tensors.items()}


def _vision(generator: torch.Generator) -> dict[str, torch.Tensor]:
    width = VISION["hidden_size"]
    # The two temporal copies are averaged before the patch embedding.
    patch = 3 * VISION["native_patch_size"] ** 2
    merged = width * VISION["adapter_merge_size"] ** 2
    tensors = {
        "backbone.embeddings.class_embedding": _bf16(generator, width),
        "backbone.embeddings.cls_position_embedding": _bf16(generator, width),
        "backbone.embeddings.patch_embedding.weight": _bf16(generator, width, patch),
        "backbone.embeddings.position_embedding": _bf16(generator, VISION["posemb_grid_size"] ** 2, width),
        "backbone.ln_pre.weight": _bf16(generator, width, mean=1.0),
        "backbone.ln_pre.bias": _bf16(generator, width),
        "proj_in.weight": _bf16(generator, VISION["projection_hidden_dim"], merged),
        "proj_out.weight": _bf16(generator, HIDDEN, VISION["projection_hidden_dim"]),
    }
    mlp = int(width * VISION["mlp_ratio"])
    for layer in range(VISION["num_hidden_layers"]):
        prefix = f"backbone.layers.{layer}"
        for norm in ("ln_1", "ln_2"):
            tensors[f"{prefix}.{norm}.weight"] = _bf16(generator, width, mean=1.0)
            tensors[f"{prefix}.{norm}.bias"] = _bf16(generator, width)
        for scale in ("ls_1", "ls_2"):
            tensors[f"{prefix}.{scale}.gamma"] = _bf16(generator, width, mean=0.1)
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            tensors[f"{prefix}.self_attn.{projection}.weight"] = _bf16(generator, width, width)
            tensors[f"{prefix}.self_attn.{projection}.bias"] = _bf16(generator, width)
        tensors[f"{prefix}.mlp.c_fc.weight"] = _bf16(generator, mlp, width)
        tensors[f"{prefix}.mlp.c_fc.bias"] = _bf16(generator, mlp)
        tensors[f"{prefix}.mlp.c_proj.weight"] = _bf16(generator, width, mlp)
        tensors[f"{prefix}.mlp.c_proj.bias"] = _bf16(generator, width)
    return {f"model.visual.{name}": tensor for name, tensor in tensors.items()}


def _attn_res_queries(generator: torch.Generator) -> dict[str, torch.Tensor]:
    tensors = {"model.llm.attn_res.output_query": _fp32(generator, HIDDEN, std=0.5)}
    for layer in range(NUM_LAYERS):
        tensors[f"model.llm.attn_res.mlp_queries.{layer}"] = _fp32(generator, HIDDEN, std=0.5)
        if layer > 0:
            # Full-attention layers mix separate query, key, and value inputs.
            shape = (3, HIDDEN) if layer in FULL_ATTENTION_LAYERS else (HIDDEN,)
            tensors[f"model.llm.attn_res.attn_queries.{layer - 1}"] = _fp32(generator, *shape, std=0.5)
    return tensors


def _quant_config(tensors: dict[str, torch.Tensor]) -> dict:
    expert_weights = sorted(name for name in tensors if name.endswith(("w13_weight", "w2_weight")))
    nvfp4 = [name for name in expert_weights if tensors[name].dtype == torch.uint8]
    mxfp8 = [name for name in expert_weights if tensors[name].dtype == torch.float8_e4m3fn]
    fp32 = sorted(name for name in tensors if ".attn_res." in name or name.endswith("mlp.gate.bias"))
    companions = tuple(f"_{suffix}" for suffix in ("scale", "amax", "outer_scale_denominator"))
    bf16 = sorted(
        name
        for name in tensors
        if name not in fp32
        and name not in expert_weights
        and not name.endswith(companions)
        and name != "model.llm.unembed.weight"
    )
    cast = {"required_activation_quantization": None, "weight_block_shape": [], "weight_transform": "cast"}
    return {
        "model": {
            "parameter_groups": [
                {"compute_dtype": "float32", "output_dtype": "float32", "tensors": fp32, **cast},
                {"compute_dtype": "bfloat16", "output_dtype": "bfloat16", "tensors": bf16, **cast},
                {
                    "compute_dtype": "nvfp4",
                    "output_dtype": "uint8",
                    "required_activation_quantization": {
                        "amax_type": "per_token",
                        "block_size": 16,
                        "format": "nvfp4_e2m1",
                        "rounding": "nearest",
                        "scale_dtype": "float8_e4m3fn",
                        "scale_selection_rule": "fouroversix_mae",
                    },
                    "tensors": nvfp4,
                    "weight_block_shape": [],
                    "weight_scale_format": "amax_denominator",
                    "weight_transform": "nvfp4_nearest",
                },
                {
                    "compute_dtype": "mxfp8",
                    "output_dtype": "float8_e4m3fn",
                    "required_activation_quantization": {
                        "block_size": 32,
                        "format": "mxfp8_e4m3",
                        "rounding": "nearest",
                        "scale_dtype": "ue8m0",
                    },
                    "tensors": mxfp8,
                    "weight_block_shape": [],
                    "weight_transform": "mxfp8_nearest",
                },
                {
                    "compute_dtype": "bfloat16",
                    "fwd_output_dtype": "float32",
                    "output_dtype": "bfloat16",
                    "tensors": ["model.llm.unembed.weight"],
                    **cast,
                },
            ]
        },
        "quant_method": "inkling2",
    }


def _save_tokenizer(path: Path) -> None:
    """Write a word-level tokenizer whose end-of-text token has the config's ``eos_token_id`` (6)."""
    tokens = ("<unk>", "<pad>", "<|begin_of_text|>", "hello", "world", "image", "<|endoftext|>")
    tokenizer = Tokenizer(WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<|begin_of_text|>",
        eos_token="<|endoftext|>",
    ).save_pretrained(path)


def write_inkling2_toy_checkpoint(path: str | Path, *, seed: int = 0) -> Path:
    """Write the toy checkpoint to ``path`` and return the directory.

    Args:
        path: Output directory; created if missing.
        seed: Seed for the random weights.

    Returns:
        The checkpoint directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        "model.llm.embed.weight": _bf16(generator, VOCAB, HIDDEN, std=1.0),
        "model.llm.embed_norm.weight": _bf16(generator, HIDDEN, mean=1.0),
        "model.llm.norm.weight": _bf16(generator, HIDDEN, mean=1.0),
        "model.llm.unembed.weight": _bf16(generator, VOCAB, HIDDEN),
    }
    tensors.update(_attn_res_queries(generator))
    for layer in range(NUM_LAYERS):
        tensors.update(_decoder_layer(generator, layer))
    tensors.update(_vision(generator))

    # Shard like the release: the common tensors apart from the decoder layers.
    shards: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        shard = (
            "model-00002-of-00002.safetensors"
            if ".layers." in name and "visual" not in name
            else ("model-00001-of-00002.safetensors")
        )
        shards.setdefault(shard, {})[name] = tensor.contiguous()
    weight_map = {}
    for shard, part in shards.items():
        save_file(part, path / shard, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(part, shard))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, indent=2)
    )
    config = {
        "architectures": ["Inkling2ForConditionalGeneration"],
        "model_type": "inkling2",
        "eos_token_id": 6,
        "image_token_id": UNPADDED_VOCAB,
        "text_config": TEXT_CONFIG,
        "vision_config": VISION,
    }
    (path / "config.json").write_text(json.dumps(config, indent=2))
    (path / "hf_quant_config.json").write_text(json.dumps(_quant_config(tensors), indent=2))
    _save_tokenizer(path)
    return path
