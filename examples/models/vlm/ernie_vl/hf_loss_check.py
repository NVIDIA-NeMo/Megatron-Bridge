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

"""Quick HF-side loss check for ERNIE 4.5 VL MoE.

Computes next-token prediction loss using the HF model directly (with its native
softmax routing) to compare against the Megatron model (sigmoid routing).

Usage:
    python hf_loss_check.py --hf-model-path ./ERNIE-4.5-VL-28B-A3B-Thinking \
        --prompt "请介绍一下你自己。"
"""

import argparse
import os


os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Run HF model loss check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model-path", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.hf_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)

    print(f"Loading HF model from {args.hf_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    # Tokenize
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"Prompt: {args.prompt!r}")
    print(f"Tokenized: {len(token_ids)} tokens")
    print(f"Token IDs: {token_ids}")

    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    seq_len = input_ids.size(1)

    # 3D M-RoPE position_ids: [batch, seq_len, 3] — for text-only, all 3 dims identical
    position_ids = (
        torch.arange(seq_len, dtype=torch.long, device=device)
        .unsqueeze(0)  # [1, seq_len]
        .unsqueeze(-1)  # [1, seq_len, 1]
        .expand(1, seq_len, 3)  # [1, seq_len, 3]
        .clone()
    )

    # Next-token prediction labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # ignore last position

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool, device=device),
            position_ids=position_ids,
        )

    # The HF model may not compute loss even when labels are passed,
    # so compute it manually from logits.
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    # logits: [1, seq_len, vocab_size]
    # Shift: logits[:-1] predicts tokens[1:]
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )

    print(f"\nHF model loss (next-token prediction): {loss.item():.6f}")
    print(
        f"ln(vocab_size) = ln({tokenizer.vocab_size}) = {torch.log(torch.tensor(float(tokenizer.vocab_size))).item():.4f}"
    )

    print(f"\nLogits shape: {logits.shape}")

    # Show top-5 predictions for each position
    for pos in range(logits.shape[1] - 1):
        topk_vals, topk_ids = torch.topk(logits[0, pos], k=5)
        actual_next = input_ids[0, pos + 1].item()
        decoded_predictions = [tokenizer.decode([tid]) for tid in topk_ids.tolist()]
        actual_decoded = tokenizer.decode([actual_next])
        print(
            f"  Position {pos}: actual_next='{actual_decoded}'({actual_next}), "
            f"top5={list(zip(decoded_predictions, topk_ids.tolist(), topk_vals.tolist()))}"
        )


if __name__ == "__main__":
    main()
