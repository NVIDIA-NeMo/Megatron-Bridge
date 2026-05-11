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
over-eager: LM parameters in the Megatron model must be bitwise-identical to
the corresponding HF checkpoint weights (via the bridge mapping).

Checks two DirectMapping params that survive the HF→Megatron conversion
unchanged at TP=1:
  - language_model.embedding.word_embeddings.weight  (embed_tokens.weight)
  - language_model.decoder.final_layernorm.weight    (final_layernorm.weight)

Run with (single GPU):
    HF_HOME=... HF_PATH=Qwen/Qwen3.5-0.8B \
    uv run python -m torch.distributed.run --nproc_per_node=1 --master_port=29710 \
        tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_weight_parity.py
"""

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


def main():
    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.qwen_vl import qwen35_llm_800m_sft_config
    from megatron.bridge.training.callbacks import Callback, CallbackContext
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.vlm_step import forward_step

    captured: dict[str, torch.Tensor] = {}

    class WeightCapture(Callback):
        def on_train_start(self, context: CallbackContext) -> None:
            from megatron.core.utils import unwrap_model

            for chunk in unwrap_model(context.model):
                for name, param in chunk.named_parameters():
                    if name in PARITY_PAIRS.values():
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

        if failed:
            print("\n  FAIL:")
            for msg in failed:
                print(f"    {msg}")
            sys.exit(1)

        print("\n  PASS — LM weights are bitwise-identical to HF source")
        print("=" * 70)


if __name__ == "__main__":
    main()
