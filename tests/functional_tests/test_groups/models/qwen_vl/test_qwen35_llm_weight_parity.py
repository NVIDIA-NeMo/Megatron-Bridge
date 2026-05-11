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

"""T2.6: LM weights loaded by qwen35_llm recipe match the HF source weights.

Verifies that with ``add_encoder=False`` the LLM-only loader is not
over-eager: LM parameters in the Megatron model must match the corresponding
HF checkpoint weights (via the bridge mapping).

Two kinds of checks:
  1. DirectMapping byte-identity for params that survive HF→Megatron
     conversion unchanged at TP=1 (embedding, final layernorm).
  2. Fused-weight L2 norm parity for params that the bridge fuses on import
     (QKV from {q,k,v}_proj; gate_up from {gate,up}_proj). The L2 norm is
     invariant under any internal permutation/concatenation order, so it
     catches the failure modes that matter for a smoke test — wrong layer
     loaded, scaling drift, fused param left at random init — without
     re-implementing the fusion layout here.

Run with (single GPU):
    HF_HOME=... HF_PATH=Qwen/Qwen3.5-0.8B \\
    uv run python -m torch.distributed.run --nproc_per_node=1 --master_port=29710 \\
        tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_weight_parity.py
"""

import math
import os
import sys
import tempfile

import torch
import torch.distributed as dist


HF_PATH = os.environ.get("HF_PATH", "Qwen/Qwen3.5-0.8B")

# HF param name → Megatron param name (DirectMapping, unchanged at TP=1)
PARITY_PAIRS = {
    "model.language_model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
    "model.language_model.norm.weight": "language_model.decoder.final_layernorm.weight",
}

# Megatron fused param → list of HF source params whose weights it concatenates.
# We check L2 norm equivalence, which is order/layout-invariant.
FUSED_PAIRS = {
    "language_model.decoder.layers.0.self_attention.linear_qkv.weight": [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
    ],
    "language_model.decoder.layers.0.mlp.linear_fc1.weight": [
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
    ],
}


def main():
    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.qwen_vl import qwen35_llm_800m_sft_config
    from megatron.bridge.training.callbacks import Callback, CallbackContext
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.vlm_step import forward_step

    captured: dict[str, torch.Tensor] = {}
    targets = set(PARITY_PAIRS.values()) | set(FUSED_PAIRS.keys())

    class WeightCapture(Callback):
        def on_train_start(self, context: CallbackContext) -> None:
            from megatron.core.utils import unwrap_model

            for chunk in unwrap_model(context.model):
                for name, param in chunk.named_parameters():
                    if name in targets:
                        captured[name] = param.detach().cpu().float()

    cfg = qwen35_llm_800m_sft_config(hf_path=HF_PATH)
    cfg.train.train_iters = 1
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1

    _ckpt_dir = tempfile.mkdtemp(prefix="t26_parity_ckpt_")
    cfg.checkpoint.save = _ckpt_dir
    cfg.checkpoint.load = _ckpt_dir

    finetune(cfg, forward_step, callbacks=[WeightCapture()])

    if not dist.is_initialized() or dist.get_rank() == 0:
        # Load HF weights lazily (no full model load needed).
        ab = AutoBridge.from_hf_pretrained(HF_PATH)

        print("\n" + "=" * 70)
        print("T2.6 LM Weight Parity Check")
        print("=" * 70)

        failed = []

        # 1. DirectMapping byte-identity checks.
        for hf_name, meg_name in PARITY_PAIRS.items():
            if meg_name not in captured:
                print(f"  SKIP  {meg_name}: not captured (param not on rank-0?)")
                continue

            hf_weight = ab.hf_pretrained.state[hf_name].cpu().float()
            meg_weight = captured[meg_name]

            if hf_weight.shape != meg_weight.shape:
                failed.append(f"Shape mismatch for {meg_name}: HF={hf_weight.shape} Meg={meg_weight.shape}")
                continue

            if not torch.equal(hf_weight, meg_weight):
                max_diff = (hf_weight - meg_weight).abs().max().item()
                failed.append(f"Value mismatch for {meg_name}: max_abs_diff={max_diff:.2e}")
            else:
                print(f"  PASS  {meg_name}  shape={tuple(hf_weight.shape)}")

        # 2. Fused-weight L2-norm parity checks (QKV, gate_up).
        for meg_name, hf_sources in FUSED_PAIRS.items():
            if meg_name not in captured:
                print(f"  SKIP  {meg_name}: not captured (param not on rank-0?)")
                continue

            meg_norm_sq = captured[meg_name].pow(2).sum().item()
            hf_norm_sq = sum(ab.hf_pretrained.state[name].cpu().float().pow(2).sum().item() for name in hf_sources)

            meg_l2 = math.sqrt(meg_norm_sq)
            hf_l2 = math.sqrt(hf_norm_sq)
            rel_err = abs(meg_l2 - hf_l2) / max(hf_l2, 1e-12)
            # bf16 round-trip + summation order tolerance; well above
            # what a random-init or wrong-layer failure would show.
            if rel_err > 1e-3:
                failed.append(
                    f"Fused L2-norm mismatch for {meg_name}: HF={hf_l2:.4f} Meg={meg_l2:.4f} "
                    f"rel_err={rel_err:.2e} (sources: {hf_sources})"
                )
            else:
                print(f"  PASS  {meg_name}  L2={meg_l2:.4f} (HF fused L2={hf_l2:.4f}, rel_err={rel_err:.2e})")

        if failed:
            print("\n  FAIL:")
            for msg in failed:
                print(f"    {msg}")
            sys.exit(1)

        print("\n  PASS — LM weights match HF source (direct + fused)")
        print("=" * 70)


if __name__ == "__main__":
    main()
