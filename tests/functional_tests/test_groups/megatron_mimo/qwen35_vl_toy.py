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
"""Tiny Qwen3.5-VL checkpoint with one MTP layer, shared by the MegatronMIMO MTP functional tests."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# Special tokens sit inside the toy vocabulary: the MTP block embeds every rolled token ID,
# placeholders included, before masking them out.
IMAGE_TOKEN_ID = 2040
VIDEO_TOKEN_ID = 2041
VISION_START_TOKEN_ID = 2042
VISION_END_TOKEN_ID = 2043
SPATIAL_MERGE_SIZE = 2
PATCH_DIM = 3 * 2 * 14 * 14  # in_channels * temporal_patch_size * patch_size**2

# Three Gated DeltaNet layers, one attention layer and one MTP layer.
HF_TOY_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "torch_dtype": "bfloat16",
    "image_token_id": IMAGE_TOKEN_ID,
    "video_token_id": VIDEO_TOKEN_ID,
    "vision_start_token_id": VISION_START_TOKEN_ID,
    "vision_end_token_id": VISION_END_TOKEN_ID,
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_text",
        "torch_dtype": "bfloat16",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "full_attention_interval": 4,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 32,
        "max_position_embeddings": 4096,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "vocab_size": 2048,
        "rope_parameters": {
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000.0,
            "mrope_section": [3, 3, 2],
        },
    },
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 1,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 256,
        "in_channels": 3,
        "intermediate_size": 512,
        "num_heads": 4,
        "num_position_embeddings": 2304,
        "out_hidden_size": 256,
        "patch_size": 14,
        "spatial_merge_size": SPATIAL_MERGE_SIZE,
        "temporal_patch_size": 2,
    },
}


def hf_toy_config():
    """Return the toy model's Hugging Face config."""
    from transformers import Qwen3_5Config

    return Qwen3_5Config(**HF_TOY_CONFIG)


def save_hf_toy_checkpoint_with_mtp(path: Path) -> None:
    """Save the toy model and add the ``mtp.*`` tensors that Hugging Face does not build."""
    from transformers import Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    config = hf_toy_config()
    Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(path, safe_serialization=True)

    weights_file = path / "model.safetensors"
    tensors = load_file(weights_file)
    # The MTP layer is a full-attention decoder layer; layer 3 is the toy's attention layer.
    attention_layer_prefix = "model.language_model.layers.3."
    for name, tensor in list(tensors.items()):
        if name.startswith(attention_layer_prefix):
            tensors["mtp.layers.0." + name.removeprefix(attention_layer_prefix)] = torch.randn_like(tensor)
    hidden_size = config.text_config.hidden_size
    tensors["mtp.fc.weight"] = torch.randn(hidden_size, 2 * hidden_size, dtype=torch.bfloat16)
    for norm_name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        tensors[f"mtp.{norm_name}.weight"] = torch.randn(hidden_size, dtype=torch.bfloat16)
    save_file(tensors, weights_file, metadata={"format": "pt"})
