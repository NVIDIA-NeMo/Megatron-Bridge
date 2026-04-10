"""
Layer-by-layer ViT alignment debug script.

Compares HF ViT vs MG ViT at each stage:
  1. PatchEmbed output
  2. RoPE computation
  3. After each transformer block
  4. After final LayerNorm

This helps isolate exactly WHERE the output divergence occurs.
"""

import argparse
import sys
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="ERNIE 4.5 VL ViT debug")
    parser.add_argument("--hf-model-path", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize megatron parallel state
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="tcp://127.0.0.1:29501",
            world_size=1, rank=0,
        )
    from megatron.core import parallel_state
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    # =====================================================================
    # Load HF ViT
    # =====================================================================
    from transformers import AutoConfig
    from transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeVisionTransformerPretrainedModel,
    )

    hf_config = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=True)
    vision_config = getattr(hf_config, "vision_config", None)

    from megatron.bridge.models.ernie_vl.modeling_ernie45_vl import _normalize_vision_config
    _normalize_vision_config(vision_config, hf_config=hf_config)

    hf_vit = Ernie4_5_VLMoeVisionTransformerPretrainedModel._from_config(vision_config)

    # Load weights from safetensors
    from safetensors import safe_open
    from pathlib import Path
    model_dir = Path(args.hf_model_path)
    text_cfg_attr = getattr(hf_config, "text_config", None)
    is_flat = (text_cfg_attr is None) or (text_cfg_attr is hf_config)
    vision_prefix = "vision_model." if is_flat else "model.vision_model."

    vision_state_dict = {}
    for st_file in model_dir.glob("*.safetensors"):
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(vision_prefix):
                    local_key = key[len(vision_prefix):]
                    vision_state_dict[local_key] = f.get_tensor(key)

    hf_vit.load_state_dict(vision_state_dict, strict=False)
    hf_vit = hf_vit.to(device=device, dtype=torch.bfloat16).eval()
    print(f"HF ViT loaded: {sum(p.numel() for p in hf_vit.parameters())} params")

    # =====================================================================
    # Build MG ViT
    # =====================================================================
    from megatron.bridge.models.ernie_vl.vision_model import ErnieVLVisionModel
    from megatron.bridge.models.ernie_vl.vision_layer_spec import get_ernie_vit_layer_spec
    from megatron.bridge.models.ernie_vl.vision_transformer_config import get_ernie_vision_config

    vit_config = get_ernie_vision_config(vision_config)
    vit_layer_spec = get_ernie_vit_layer_spec()
    mg_vit = ErnieVLVisionModel(
        transformer_config=vit_config,
        transformer_layer_spec=vit_layer_spec,
    )
    mg_vit = mg_vit.to(device=device, dtype=torch.bfloat16).eval()
    print(f"MG ViT created: {sum(p.numel() for p in mg_vit.parameters())} params")

    # Transfer weights
    hf_sd = hf_vit.state_dict()
    mg_sd = mg_vit.state_dict()

    # Build mapping
    key_mapping = {}
    key_mapping["patch_embed.proj.weight"] = "patch_embed.proj.weight"
    key_mapping["ln.weight"] = "decoder.final_layernorm.weight"
    key_mapping["ln.bias"] = "decoder.final_layernorm.bias"

    num_layers = getattr(vision_config, "depth",
                         getattr(vision_config, "num_hidden_layers", 32))
    for i in range(num_layers):
        key_mapping[f"blocks.{i}.attn.qkv.weight"] = f"decoder.layers.{i}.self_attention.linear_qkv.weight"
        key_mapping[f"blocks.{i}.attn.qkv.bias"] = f"decoder.layers.{i}.self_attention.linear_qkv.bias"
        key_mapping[f"blocks.{i}.attn.proj.weight"] = f"decoder.layers.{i}.self_attention.linear_proj.weight"
        key_mapping[f"blocks.{i}.attn.proj.bias"] = f"decoder.layers.{i}.self_attention.linear_proj.bias"
        key_mapping[f"blocks.{i}.norm1.weight"] = f"decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight"
        key_mapping[f"blocks.{i}.norm1.bias"] = f"decoder.layers.{i}.self_attention.linear_qkv.layer_norm_bias"
        key_mapping[f"blocks.{i}.norm2.weight"] = f"decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight"
        key_mapping[f"blocks.{i}.norm2.bias"] = f"decoder.layers.{i}.mlp.linear_fc1.layer_norm_bias"
        key_mapping[f"blocks.{i}.mlp.fc1.weight"] = f"decoder.layers.{i}.mlp.linear_fc1.weight"
        key_mapping[f"blocks.{i}.mlp.fc1.bias"] = f"decoder.layers.{i}.mlp.linear_fc1.bias"
        key_mapping[f"blocks.{i}.mlp.fc2.weight"] = f"decoder.layers.{i}.mlp.linear_fc2.weight"
        key_mapping[f"blocks.{i}.mlp.fc2.bias"] = f"decoder.layers.{i}.mlp.linear_fc2.bias"

    transferred = 0
    for hf_key, mg_key in key_mapping.items():
        if hf_key in hf_sd and mg_key in mg_sd and hf_sd[hf_key].shape == mg_sd[mg_key].shape:
            mg_sd[mg_key] = hf_sd[hf_key].to(dtype=mg_sd[mg_key].dtype)
            transferred += 1
    mg_vit.load_state_dict(mg_sd, strict=True)
    print(f"Transferred {transferred}/{len(key_mapping)} weight tensors")

    # =====================================================================
    # Generate dummy input
    # =====================================================================
    patch_size = getattr(vision_config, "patch_size", 14)
    in_channels = getattr(vision_config, "in_channels", 3)
    spatial_merge = getattr(vision_config, "spatial_merge_size", 2)

    grid_thw_list = []
    for i in range(args.num_images):
        t, h, w = 1, spatial_merge * (2 + i), spatial_merge * (2 + i)
        grid_thw_list.append([t, h, w])

    grid_thw = torch.tensor(grid_thw_list, dtype=torch.long, device=device)
    total_patches = int(torch.prod(grid_thw, dim=1).sum().item())

    pixel_values = torch.randn(
        total_patches, in_channels * patch_size * patch_size,
        dtype=torch.bfloat16, device=device,
    )
    print(f"\nInput: pixel_values {pixel_values.shape}, grid_thw {grid_thw}")

    # =====================================================================
    # Test 1: PatchEmbed
    # =====================================================================
    print("\n" + "=" * 60)
    print("TEST 1: PatchEmbed")
    print("=" * 60)
    with torch.no_grad():
        hf_patch_out = hf_vit.patch_embed(pixel_values)
        mg_patch_out = mg_vit.patch_embed(pixel_values)
    cos_sim = F.cosine_similarity(hf_patch_out.float().flatten().unsqueeze(0),
                                   mg_patch_out.float().flatten().unsqueeze(0)).item()
    max_diff = (hf_patch_out.float() - mg_patch_out.float()).abs().max().item()
    print(f"  PatchEmbed cosine_sim: {cos_sim:.8f}, max_diff: {max_diff:.6e}")
    assert cos_sim > 0.999, f"PatchEmbed MISMATCH: cos_sim = {cos_sim}"
    print("  PASS")

    # =====================================================================
    # Test 2: RoPE
    # =====================================================================
    print("\n" + "=" * 60)
    print("TEST 2: RoPE")
    print("=" * 60)
    with torch.no_grad():
        # HF RoPE
        hf_rotary = hf_vit.rot_pos_emb(grid_thw)  # [N, 40]
        hf_emb = torch.cat((hf_rotary, hf_rotary), dim=-1)  # [N, 80]
        hf_cos = hf_emb.cos()
        hf_sin = hf_emb.sin()

        # MG RoPE
        mg_rotary = mg_vit.rot_pos_emb(grid_thw)  # [N, 40]
        mg_emb = torch.cat((mg_rotary.reshape(total_patches, 1, 1, -1),
                           mg_rotary.reshape(total_patches, 1, 1, -1)), dim=-1)
        mg_cos = mg_emb.cos().flatten()
        mg_sin = mg_emb.sin().flatten()

    cos_sim_rope = F.cosine_similarity(
        hf_rotary.float().flatten().unsqueeze(0),
        mg_rotary.float().flatten().unsqueeze(0)
    ).item()
    max_diff_rope = (hf_rotary.float() - mg_rotary.float()).abs().max().item()
    print(f"  HF rot_pos_emb shape: {hf_rotary.shape}")
    print(f"  MG rot_pos_emb shape: {mg_rotary.shape}")
    print(f"  RoPE freqs cosine_sim: {cos_sim_rope:.8f}, max_diff: {max_diff_rope:.6e}")

    cos_sim_cos = F.cosine_similarity(
        hf_cos.float().flatten().unsqueeze(0),
        mg_cos.float().flatten().unsqueeze(0)
    ).item()
    print(f"  cos(emb) cosine_sim: {cos_sim_cos:.8f}")

    # =====================================================================
    # Test 3: RoPE application on dummy Q, K
    # =====================================================================
    print("\n" + "=" * 60)
    print("TEST 3: RoPE application")
    print("=" * 60)
    head_dim = 80
    num_heads = 16
    with torch.no_grad():
        # Create dummy Q, K
        q_test = torch.randn(total_patches, num_heads, head_dim,
                              dtype=torch.float32, device=device)

        # HF RoPE application
        from transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import (
            apply_rotary_pos_emb_vision,
        )
        hf_q_rot, _ = apply_rotary_pos_emb_vision(q_test, q_test, hf_cos, hf_sin)

        # MG RoPE application
        from megatron.bridge.models.ernie_vl.vision_attention import apply_rotary_pos_emb_absolute
        mg_freqs = torch.cat((mg_rotary, mg_rotary), dim=-1)  # [N, 80]
        mg_freqs_4d = mg_freqs.reshape(total_patches, 1, 1, -1)  # [N, 1, 1, 80]
        # MG path: bshd format
        mg_q_rot = apply_rotary_pos_emb_absolute(
            q_test[:, None],  # [N, 1, 16, 80]
            mg_freqs_4d,
            config=vit_config,
            cu_seqlens=None,
        ).squeeze(1)  # [N, 16, 80]

    cos_sim_qrot = F.cosine_similarity(
        hf_q_rot.float().flatten().unsqueeze(0),
        mg_q_rot.float().flatten().unsqueeze(0)
    ).item()
    max_diff_qrot = (hf_q_rot.float() - mg_q_rot.float()).abs().max().item()
    print(f"  RoPE-applied Q cosine_sim: {cos_sim_qrot:.8f}, max_diff: {max_diff_qrot:.6e}")

    # =====================================================================
    # Test 4: Single block comparison
    # =====================================================================
    print("\n" + "=" * 60)
    print("TEST 4: First transformer block")
    print("=" * 60)
    with torch.no_grad():
        # HF: first block
        hf_hidden = hf_patch_out.clone()
        hf_position_embeddings = (hf_cos, hf_sin)
        # HF cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hf_block0_out = hf_vit.blocks[0](
            hf_hidden,
            cu_seqlens=cu_seqlens,
            position_embeddings=hf_position_embeddings,
        )

        # MG: first block - need to trace through the TransformerBlock
        mg_hidden = mg_patch_out.clone()
        mg_hidden_3d = mg_hidden[:, None]  # [N, 1, hidden]

        mg_packed_seq = mg_vit.build_packed_seq_params(grid_thw)
        mg_rotary_for_block = torch.cat(
            (mg_rotary.reshape(total_patches, 1, 1, -1),
             mg_rotary.reshape(total_patches, 1, 1, -1)),
            dim=-1,
        )

        # Get first layer from MG decoder
        mg_layer0 = mg_vit.decoder.layers[0]

        # Run MG first block
        mg_block0_out, _ = mg_layer0(
            hidden_states=mg_hidden_3d,
            attention_mask=None,
            rotary_pos_emb=mg_rotary_for_block,
            packed_seq_params=mg_packed_seq,
        )

    cos_sim_block0 = F.cosine_similarity(
        hf_block0_out.float().flatten().unsqueeze(0),
        mg_block0_out.squeeze(1).float().flatten().unsqueeze(0)
    ).item()
    max_diff_block0 = (hf_block0_out.float() - mg_block0_out.squeeze(1).float()).abs().max().item()
    print(f"  Block 0 cosine_sim: {cos_sim_block0:.8f}, max_diff: {max_diff_block0:.6e}")

    # =====================================================================
    # Test 5: Check the LayerNorm + QKV path specifically
    # =====================================================================
    print("\n" + "=" * 60)
    print("TEST 5: LayerNorm + QKV (pre-attention)")
    print("=" * 60)
    with torch.no_grad():
        # HF: norm1 then QKV
        hf_normed = hf_vit.blocks[0].norm1(hf_patch_out)
        hf_qkv = hf_vit.blocks[0].attn.qkv(hf_normed)

        # MG: TELayerNormColumnParallelLinear does fused LN + Linear
        # But we can compare the QKV output by accessing the linear_qkv layer
        mg_linear_qkv = mg_layer0.self_attention.linear_qkv
        # This is TELayerNormColumnParallelLinear - it fuses LN + Linear
        mg_qkv_out, _ = mg_linear_qkv(mg_patch_out[:, None])  # [N, 1, 3*hidden]
        mg_qkv_flat = mg_qkv_out.squeeze(1)

    cos_sim_ln = F.cosine_similarity(
        hf_normed.float().flatten().unsqueeze(0),
        # We can't easily extract just the LN output from the fused module
        # so just compare QKV output
        torch.zeros(1).unsqueeze(0),  # placeholder
    ).item()

    cos_sim_qkv = F.cosine_similarity(
        hf_qkv.float().flatten().unsqueeze(0),
        mg_qkv_flat.float().flatten().unsqueeze(0)
    ).item()
    max_diff_qkv = (hf_qkv.float() - mg_qkv_flat.float()).abs().max().item()
    print(f"  QKV output cosine_sim: {cos_sim_qkv:.8f}, max_diff: {max_diff_qkv:.6e}")

    # =====================================================================
    # Test 6: Full model
    # =====================================================================
    print("\n" + "=" * 60)
    print("TEST 6: Full model comparison")
    print("=" * 60)
    with torch.no_grad():
        hf_out = hf_vit(pixel_values, grid_thw, return_dict=True).last_hidden_state
        mg_out = mg_vit(pixel_values, grid_thw)

    cos_sim_full = F.cosine_similarity(
        hf_out.float().flatten().unsqueeze(0),
        mg_out.float().flatten().unsqueeze(0)
    ).item()
    max_diff_full = (hf_out.float() - mg_out.float()).abs().max().item()
    print(f"  Full model cosine_sim: {cos_sim_full:.8f}, max_diff: {max_diff_full:.6e}")

    if cos_sim_full > 0.99:
        print("\nOVERALL: PASS")
    else:
        print("\nOVERALL: FAIL")

    # Cleanup
    parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    sys.exit(0 if cos_sim_full > 0.99 else 1)


if __name__ == "__main__":
    main()
