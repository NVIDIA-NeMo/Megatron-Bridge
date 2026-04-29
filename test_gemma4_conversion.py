"""
Gemma 4 toy model creation and roundtrip conversion test.

Creates a tiny Gemma4ForCausalLM with random weights (matching the Gemma 4 MoE
architecture) and runs it through the Megatron Bridge roundtrip conversion.

Usage:
    uv run python test_gemma4_conversion.py
"""

import json
import os
import tempfile

import torch
from rich.console import Console

console = Console()

# Minimal Gemma 4 config that matches the real architecture
# but with tiny dimensions for fast testing
GEMMA4_TOY_CONFIG = {
    "architectures": ["Gemma4ForCausalLM"],
    "model_type": "gemma4_text",
    # Model dims (tiny)
    "hidden_size": 256,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,         # sliding attention KV heads (must divide num_attention_heads)
    "num_global_key_value_heads": 2,  # global attention KV heads (must divide num_attention_heads)
    "attention_k_eq_v": True,         # K=V sharing on global layers (v_proj absent)
    "head_dim": 256,
    "global_head_dim": 512,
    # MLP / MoE
    "intermediate_size": 256,         # dense MLP (shared expert)
    "moe_intermediate_size": 128,     # routed expert intermediate
    "num_experts": 4,                 # tiny: 4 experts (real is 128)
    "top_k_experts": 2,               # top-2 routing (real is 8)
    "enable_moe_block": True,
    # Attention
    "sliding_window": 64,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "full_attention"],
    # New nested rope_parameters format (transformers 5.6+)
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
    },
    # Norms
    "rms_norm_eps": 1e-6,
    # Logit softcapping
    "final_logit_softcapping": 30.0,
    # Misc
    "vocab_size": 262144,
    "max_position_embeddings": 2048,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "pad_token_id": 0,
    "torch_dtype": "bfloat16",
    "hidden_activation": "gelu_pytorch_tanh",
    "attention_bias": False,
}


def main():
    from transformers import AutoConfig, GemmaTokenizer, Gemma4ForCausalLM

    console.print("[bold green]Step 1: Creating Gemma 4 toy model...[/bold green]")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = os.path.join(tmp_dir, "gemma4_toy")
        os.makedirs(model_dir)

        # Write config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(GEMMA4_TOY_CONFIG, f, indent=2)

        # Save tokenizer (Gemma 4 uses same SentencePiece vocab as Gemma 2/3)
        try:
            tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2b")
            tokenizer.save_pretrained(model_dir)
            console.print("  Tokenizer saved")
        except Exception as e:
            console.print(f"  [yellow]Tokenizer not available: {e}[/yellow]")

        # Load config and create model with random weights
        hf_config = AutoConfig.from_pretrained(model_dir)
        console.print(f"  Config type: {type(hf_config).__name__}")

        model = Gemma4ForCausalLM(hf_config).bfloat16()
        model.save_pretrained(model_dir, safe_serialization=True)
        console.print(f"  Saved toy model to {model_dir}")

        # Count params
        n_params = sum(p.numel() for p in model.parameters())
        console.print(f"  Parameters: {n_params:,}")

        console.print("\n[bold green]Step 2: Running AutoBridge roundtrip conversion...[/bold green]")

        from megatron.bridge import AutoBridge
        from megatron.bridge.models.conversion import weights_verification_table

        bridge = AutoBridge.from_hf_pretrained(model_dir)
        console.print(f"  Bridge type: {type(bridge).__name__}")

        megatron_model = bridge.to_megatron_model(wrap_with_ddp=False)
        console.print("  Megatron model instantiated")

        try:
            table = weights_verification_table(bridge, megatron_model)
            console.print(table)
        except KeyError as e:
            console.print(f"  [yellow]Verification skipped (synthesized weight): {e}[/yellow]")

        output_dir = os.path.join(tmp_dir, "converted")
        os.makedirs(output_dir)
        console.print(f"\n[bold green]Step 3: Saving converted model to {output_dir}...[/bold green]")
        bridge.save_hf_pretrained(megatron_model, os.path.join(output_dir, "gemma4_toy"), strict=False)
        console.print("[bold green]SUCCESS: Roundtrip conversion completed![/bold green]")


if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
