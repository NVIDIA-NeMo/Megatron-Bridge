# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Vision encoder (ViT) alignment test: HF ViT vs MG-native ViT.

Compares hidden state outputs from the HuggingFace
Ernie4_5_VLMoeVisionTransformerPretrainedModel and the Megatron-Core native
ErnieVLVisionModel using the same weights and identical input.

Strategy (single GPU, no torchrun required):
    1. Load HF vision model from the ERNIE 4.5 VL checkpoint
    2. Construct MG ViT (ErnieVLVisionModel) with matching config
    3. Transfer weights from HF state_dict to MG state_dict
    4. Generate dummy pixel patches + grid_thw
    5. Forward both models
    6. Compare: cosine similarity, max absolute diff, relative diff

This test validates that the MG-native ViT produces the same output as the
HF ViT, ensuring weight conversion correctness and architectural fidelity.

Usage:
    # With the real 28B model:
    python ernie45_vl_vit_compare.py \
        --hf-model-path /path/to/ERNIE-4.5-VL-28B-A3B-Thinking

    # With a toy model (created by test_ernie45_vl_conversion.py):
    python ernie45_vl_vit_compare.py \
        --hf-model-path /tmp/ernie45_vl_toy

Exit code 0 = PASS, non-zero = FAIL.
"""

import argparse
import logging
import os
import sys


os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn.functional as F


def build_hf_vit(hf_model_path: str, device: torch.device):
    """Load the HF vision encoder from an ERNIE 4.5 VL checkpoint.

    Returns (hf_vit_model, vision_config, hf_config).
    """
    from transformers import AutoConfig
    from transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeVisionTransformerPretrainedModel,
    )

    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

    # Extract vision config
    vision_config = getattr(hf_config, "vision_config", None)
    if vision_config is None:
        raise ValueError("HF config does not have vision_config")

    # Normalize vision config (add missing attributes for compat)
    from megatron.bridge.models.ernie_vl.modeling_ernie45_vl import (
        _normalize_vision_config,
    )

    _normalize_vision_config(vision_config, hf_config=hf_config)

    # Create HF ViT from config (random weights)
    hf_vit = Ernie4_5_VLMoeVisionTransformerPretrainedModel._from_config(vision_config)

    # Load weights from the checkpoint
    # HF ViT weights are prefixed with "vision_model." or "model.vision_model." on disk
    from pathlib import Path

    from safetensors import safe_open

    model_dir = Path(hf_model_path)
    safetensors_files = list(model_dir.glob("*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {hf_model_path}")

    # Determine the on-disk prefix for vision weights
    # Flat config (Thinking): "vision_model."
    # Nested config (Instruct): "model.vision_model."
    text_cfg_attr = getattr(hf_config, "text_config", None)
    is_flat = (text_cfg_attr is None) or (text_cfg_attr is hf_config)
    vision_prefix = "vision_model." if is_flat else "model.vision_model."

    vision_state_dict = {}
    for st_file in safetensors_files:
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(vision_prefix):
                    # Strip the prefix to get the HF in-memory key
                    local_key = key[len(vision_prefix) :]
                    vision_state_dict[local_key] = f.get_tensor(key)

    if not vision_state_dict:
        raise ValueError(f"No vision weights found with prefix '{vision_prefix}' in {hf_model_path}")

    # Load state dict into HF ViT
    missing, unexpected = hf_vit.load_state_dict(vision_state_dict, strict=False)
    if missing:
        logging.warning("Missing keys in HF ViT: %s...", missing[:5])
    if unexpected:
        logging.warning("Unexpected keys in HF ViT: %s...", unexpected[:5])

    hf_vit = hf_vit.to(device=device, dtype=torch.bfloat16).eval()
    print(f"HF ViT loaded: {sum(p.numel() for p in hf_vit.parameters())} params")
    return hf_vit, vision_config, hf_config


def build_mg_vit(vision_config, hf_config, device: torch.device):
    """Construct the MG-native ErnieVLVisionModel.

    Returns the model on the specified device.
    """
    from megatron.bridge.models.ernie_vl.vision_layer_spec import get_ernie_vit_layer_spec
    from megatron.bridge.models.ernie_vl.vision_model import ErnieVLVisionModel
    from megatron.bridge.models.ernie_vl.vision_transformer_config import (
        get_ernie_vision_config,
    )

    # Build transformer config from vision config
    vit_config = get_ernie_vision_config(vision_config)
    vit_layer_spec = get_ernie_vit_layer_spec()

    mg_vit = ErnieVLVisionModel(
        transformer_config=vit_config,
        transformer_layer_spec=vit_layer_spec,
    )

    mg_vit = mg_vit.to(device=device, dtype=torch.bfloat16).eval()
    print(f"MG ViT created: {sum(p.numel() for p in mg_vit.parameters())} params")
    return mg_vit


def _interleave_qkv(hf_qkv, num_heads):
    """Convert HF contiguous QKV layout to Megatron interleaved layout.

    HF stores fused QKV as contiguous blocks:
        [Q0, Q1, ..., Q_{H-1} | K0, K1, ..., K_{H-1} | V0, V1, ..., V_{H-1}]

    Megatron's get_query_key_value_tensors() expects GQA-interleaved layout:
        [Q0, K0, V0 | Q1, K1, V1 | ... | Q_{H-1}, K_{H-1}, V_{H-1}]

    For MHA (num_query_groups == num_heads), each group has exactly one Q, K, V head.

    Args:
        hf_qkv: Tensor of shape [3*hidden, ...] in HF contiguous layout.
                 For weight: [3*hidden, hidden], for bias: [3*hidden].
        num_heads: Number of attention heads.

    Returns:
        Tensor of same shape but with Megatron interleaved layout.
    """
    hidden = hf_qkv.shape[0] // 3
    head_dim = hidden // num_heads

    # Split into Q, K, V
    q_all = hf_qkv[:hidden]
    k_all = hf_qkv[hidden : 2 * hidden]
    v_all = hf_qkv[2 * hidden :]

    # Reshape to per-head: [num_heads, head_dim, ...]
    q_heads = q_all.reshape(num_heads, head_dim, *q_all.shape[1:])
    k_heads = k_all.reshape(num_heads, head_dim, *k_all.shape[1:])
    v_heads = v_all.reshape(num_heads, head_dim, *v_all.shape[1:])

    # Interleave: [Q0, K0, V0, Q1, K1, V1, ...]
    interleaved = torch.stack([q_heads, k_heads, v_heads], dim=1)
    return interleaved.reshape(hf_qkv.shape)


def transfer_weights_hf_to_mg(hf_vit, mg_vit, vision_config):
    """Transfer weights from HF ViT to MG ViT using direct state_dict mapping.

    The HF ViT state_dict keys map to MG ViT keys as follows:
        HF: patch_embed.proj.weight           -> MG: patch_embed.proj.weight
        HF: blocks.{i}.attn.qkv.weight/bias   -> MG: decoder.layers.{i}.self_attention.linear_qkv.weight/bias
        HF: blocks.{i}.attn.proj.weight/bias   -> MG: decoder.layers.{i}.self_attention.linear_proj.weight/bias
        HF: blocks.{i}.norm1.weight/bias       -> MG: decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight/bias
        HF: blocks.{i}.norm2.weight/bias       -> MG: decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight/bias
        HF: blocks.{i}.mlp.fc1.weight/bias     -> MG: decoder.layers.{i}.mlp.linear_fc1.weight/bias
        HF: blocks.{i}.mlp.fc2.weight/bias     -> MG: decoder.layers.{i}.mlp.linear_fc2.weight/bias
        HF: ln.weight/bias                     -> MG: decoder.final_layernorm.weight/bias

    Note: The fused QKV weight/bias must be interleaved from HF's contiguous
    [Q|K|V] format to Megatron's per-head [Q0,K0,V0|Q1,K1,V1|...] format.
    This interleaving is required because Megatron's get_query_key_value_tensors()
    splits the fused QKV output by query groups (= per head for MHA).
    """
    hf_sd = hf_vit.state_dict()
    mg_sd = mg_vit.state_dict()

    num_heads = getattr(vision_config, "num_heads", getattr(vision_config, "num_attention_heads", 16))

    # Build mapping: HF key -> MG key
    key_mapping = {}
    # Track which HF keys need QKV interleaving
    qkv_keys = set()

    # Patch embed
    key_mapping["patch_embed.proj.weight"] = "patch_embed.proj.weight"

    # Final layernorm
    key_mapping["ln.weight"] = "decoder.final_layernorm.weight"
    key_mapping["ln.bias"] = "decoder.final_layernorm.bias"

    # Per-block mappings
    num_layers = getattr(vision_config, "depth", getattr(vision_config, "num_hidden_layers", 32))
    for i in range(num_layers):
        # QKV (fused) - needs interleaving
        qkv_w_key = f"blocks.{i}.attn.qkv.weight"
        qkv_b_key = f"blocks.{i}.attn.qkv.bias"
        key_mapping[qkv_w_key] = f"decoder.layers.{i}.self_attention.linear_qkv.weight"
        key_mapping[qkv_b_key] = f"decoder.layers.{i}.self_attention.linear_qkv.bias"
        qkv_keys.add(qkv_w_key)
        qkv_keys.add(qkv_b_key)
        # Proj
        key_mapping[f"blocks.{i}.attn.proj.weight"] = f"decoder.layers.{i}.self_attention.linear_proj.weight"
        key_mapping[f"blocks.{i}.attn.proj.bias"] = f"decoder.layers.{i}.self_attention.linear_proj.bias"
        # Norm1 -> fused into linear_qkv
        key_mapping[f"blocks.{i}.norm1.weight"] = f"decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight"
        key_mapping[f"blocks.{i}.norm1.bias"] = f"decoder.layers.{i}.self_attention.linear_qkv.layer_norm_bias"
        # Norm2 -> fused into linear_fc1
        key_mapping[f"blocks.{i}.norm2.weight"] = f"decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight"
        key_mapping[f"blocks.{i}.norm2.bias"] = f"decoder.layers.{i}.mlp.linear_fc1.layer_norm_bias"
        # MLP fc1
        key_mapping[f"blocks.{i}.mlp.fc1.weight"] = f"decoder.layers.{i}.mlp.linear_fc1.weight"
        key_mapping[f"blocks.{i}.mlp.fc1.bias"] = f"decoder.layers.{i}.mlp.linear_fc1.bias"
        # MLP fc2
        key_mapping[f"blocks.{i}.mlp.fc2.weight"] = f"decoder.layers.{i}.mlp.linear_fc2.weight"
        key_mapping[f"blocks.{i}.mlp.fc2.bias"] = f"decoder.layers.{i}.mlp.linear_fc2.bias"

    # Transfer weights
    transferred = 0
    for hf_key, mg_key in key_mapping.items():
        if hf_key not in hf_sd:
            print(f"  WARNING: HF key not found: {hf_key}")
            continue
        if mg_key not in mg_sd:
            print(f"  WARNING: MG key not found: {mg_key}")
            continue

        hf_tensor = hf_sd[hf_key]
        mg_tensor = mg_sd[mg_key]

        if hf_tensor.shape != mg_tensor.shape:
            print(f"  WARNING: Shape mismatch for {hf_key} -> {mg_key}: {hf_tensor.shape} vs {mg_tensor.shape}")
            continue

        # Apply QKV interleaving for fused QKV weight and bias
        if hf_key in qkv_keys:
            hf_tensor = _interleave_qkv(hf_tensor, num_heads)

        mg_sd[mg_key] = hf_tensor.to(dtype=mg_tensor.dtype)
        transferred += 1

    # Load the mapped state dict
    mg_vit.load_state_dict(mg_sd, strict=True)
    print(f"Transferred {transferred}/{len(key_mapping)} weight tensors")

    # Check for any MG keys that weren't covered
    mapped_mg_keys = set(key_mapping.values())
    unmapped_mg_keys = [k for k in mg_sd if k not in mapped_mg_keys]
    if unmapped_mg_keys:
        print(f"  WARNING: {len(unmapped_mg_keys)} MG keys not mapped: {unmapped_mg_keys[:5]}...")


def generate_dummy_input(vision_config, device: torch.device, num_images: int = 2):
    """Generate dummy pixel patches and grid_thw for testing.

    Creates random pixel patches that mimic the output of the ERNIE 4.5 VL
    processor (pre-flattened patches of shape [total_patches, C*P*P]).

    Args:
        vision_config: HF vision config with patch_size, spatial_merge_size.
        device: Target device.
        num_images: Number of images to simulate.

    Returns:
        (pixel_values, grid_thw): Dummy inputs.
    """
    patch_size = getattr(vision_config, "patch_size", 14)
    in_channels = getattr(vision_config, "in_channels", 3)
    spatial_merge = getattr(vision_config, "spatial_merge_size", 2)

    # Create images of varying sizes (must be divisible by spatial_merge_size)
    # Use small sizes for efficiency
    grid_thw_list = []
    for i in range(num_images):
        t = 1  # Single frame
        h = spatial_merge * (2 + i)  # e.g., 4, 6 for merge_size=2
        w = spatial_merge * (2 + i)
        grid_thw_list.append([t, h, w])

    grid_thw = torch.tensor(grid_thw_list, dtype=torch.long, device=device)
    total_patches = int(torch.prod(grid_thw, dim=1).sum().item())

    # Generate random pixel patches (simulating processor output)
    # Shape: [total_patches, C * patch_size^2]
    pixel_values = torch.randn(
        total_patches,
        in_channels * patch_size * patch_size,
        dtype=torch.bfloat16,
        device=device,
    )

    return pixel_values, grid_thw


def run_hf_vit_forward(hf_vit, pixel_values, grid_thw):
    """Run HF ViT forward pass and return hidden states."""
    with torch.no_grad():
        output = hf_vit(pixel_values, grid_thw, return_dict=True)
    return output.last_hidden_state


def run_mg_vit_forward(mg_vit, pixel_values, grid_thw):
    """Run MG ViT forward pass and return hidden states."""
    with torch.no_grad():
        output = mg_vit(pixel_values, grid_thw)
    return output


def compare_outputs(hf_out: torch.Tensor, mg_out: torch.Tensor, threshold: float = 0.99):
    """Compare HF and MG ViT outputs and return (passed, stats_dict)."""
    assert hf_out.shape == mg_out.shape, f"Shape mismatch: HF={hf_out.shape} vs MG={mg_out.shape}"

    hf_flat = hf_out.float().flatten()
    mg_flat = mg_out.float().flatten()

    # Cosine similarity
    cos_sim = F.cosine_similarity(hf_flat.unsqueeze(0), mg_flat.unsqueeze(0)).item()

    # Absolute difference
    abs_diff = (hf_flat - mg_flat).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    # Relative difference
    denom = hf_flat.abs().clamp(min=1e-8)
    rel_diff = abs_diff / denom
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    stats = {
        "cosine_similarity": cos_sim,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
    }

    passed = cos_sim >= threshold
    return passed, stats


def main():
    """Run ERNIE 4.5 VL ViT alignment test."""
    parser = argparse.ArgumentParser(description="ERNIE 4.5 VL ViT alignment test")
    parser.add_argument(
        "--hf-model-path",
        type=str,
        required=True,
        help="Path to the HF ERNIE 4.5 VL model directory",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=2,
        help="Number of dummy images to generate (default: 2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Cosine similarity threshold for pass/fail (default: 0.99)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model path: {args.hf_model_path}")

    # Initialize megatron parallel state for MG ViT (single GPU, TP=1)
    # This is needed because TransformerBlock uses parallel_state internally.
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="tcp://127.0.0.1:29500",
            world_size=1,
            rank=0,
        )
    from megatron.core import parallel_state

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    # Step 1: Load HF ViT
    print("\n=== Step 1: Loading HF ViT ===")
    hf_vit, vision_config, hf_config = build_hf_vit(args.hf_model_path, device)

    # Step 2: Build MG ViT
    print("\n=== Step 2: Building MG ViT ===")
    mg_vit = build_mg_vit(vision_config, hf_config, device)

    # Step 3: Transfer weights
    print("\n=== Step 3: Transferring weights HF -> MG ===")
    transfer_weights_hf_to_mg(hf_vit, mg_vit, vision_config)

    # Step 4: Generate dummy input
    print("\n=== Step 4: Generating dummy input ===")
    pixel_values, grid_thw = generate_dummy_input(vision_config, device, num_images=args.num_images)
    print(f"  pixel_values: {pixel_values.shape}, dtype={pixel_values.dtype}")
    print(f"  grid_thw: {grid_thw}")
    print(f"  total_patches: {pixel_values.shape[0]}")

    # Step 5: Forward pass
    print("\n=== Step 5: Running forward passes ===")
    hf_out = run_hf_vit_forward(hf_vit, pixel_values, grid_thw)
    mg_out = run_mg_vit_forward(mg_vit, pixel_values, grid_thw)
    print(f"  HF output: {hf_out.shape}, dtype={hf_out.dtype}")
    print(f"  MG output: {mg_out.shape}, dtype={mg_out.dtype}")

    # Step 6: Compare
    print("\n=== Step 6: Comparing outputs ===")
    passed, stats = compare_outputs(hf_out, mg_out, threshold=args.threshold)

    print(f"  Cosine similarity:  {stats['cosine_similarity']:.8f}")
    print(f"  Max abs diff:       {stats['max_abs_diff']:.6e}")
    print(f"  Mean abs diff:      {stats['mean_abs_diff']:.6e}")
    print(f"  Max rel diff:       {stats['max_rel_diff']:.6e}")
    print(f"  Mean rel diff:      {stats['mean_rel_diff']:.6e}")
    print(f"  Threshold:          {args.threshold}")

    if passed:
        print("\nRESULT: PASS -- MG ViT output matches HF ViT")
    else:
        print("\nRESULT: FAIL -- MG ViT output does NOT match HF ViT")

    # Cleanup
    del hf_vit, mg_vit
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Destroy parallel state
    parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
