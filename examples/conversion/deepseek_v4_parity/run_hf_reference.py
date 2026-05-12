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

"""Run a HuggingFace reference forward pass on a fixed prompt set and save the last-position logits.

The output is a `.pt` file consumed by ``compare_logits.py``. The artifact format is shared
with ``run_mbridge.py`` so the comparator can load either side without branching.

DSv4 has native ``DeepseekV4ForCausalLM`` support in transformers, so the default loader
path works without ``trust_remote_code``.

Run example (DSv4-Flash on 4xB200, sharded via accelerate ``device_map="auto"``):

    uv run python examples/conversion/deepseek_v4_parity/run_hf_reference.py \
        --hf_model_path deepseek-ai/DeepSeek-V4-Flash \
        --prompts_file examples/conversion/deepseek_v4_parity/prompts.json \
        --output /chcui/parity/dsv4/logits_hf.pt
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run HF reference forward pass and save logits")
    parser.add_argument("--hf_model_path", type=str, required=True, help="HuggingFace model id or local path")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output .pt file path")
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Device map for from_pretrained ("auto", "cuda", or a JSON dict)',
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to HF loaders")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Truncate any prompt longer than this (in tokens) to keep activations bounded",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Save top-K token ids/logits for sanity inspection")
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=1,
        help="Pad input_ids on the right to this multiple. Match the Megatron TP size so HF/MB "
        "see the same input (otherwise padded-vs-unpadded comparisons drift on bf16 transformers).",
    )
    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=0,
        help="Token id used for right-padding to --pad_to_multiple_of. compare.py uses 0; mirror that.",
    )
    return parser.parse_args()


def load_prompts(prompts_file: str) -> list[dict]:
    """Load the prompts JSON file and return its ``prompts`` list."""
    with open(prompts_file) as f:
        data = json.load(f)
    return data["prompts"]


def main() -> None:
    """Run the HF reference forward pass on each prompt and save artifacts."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    args = parse_args()

    prompts = load_prompts(args.prompts_file)
    logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    logger.info(f"Loading tokenizer from {args.hf_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=args.trust_remote_code)

    logger.info(f"Loading model from {args.hf_model_path} with device_map={args.device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    logger.info(f"Model loaded. Class: {type(model).__name__}")

    results: list[dict] = []

    for i, item in enumerate(prompts):
        prompt_id = item.get("id", str(i))
        prompt_text = item["text"]

        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=args.max_seq_len, add_special_tokens=True
        )
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        orig_seq_len = int(input_ids.shape[1])

        # Pad to Megatron's TP multiple so HF sees the same input layout MB does. We treat the
        # padded positions as *real* tokens (attention_mask=1 for them) — this matches the
        # Megatron side which uses a pure causal mask (no padding-awareness). With
        # attention_mask=[1]*orig+[0]*pad, HF subtly differs from MB at the last real position
        # due to mask-aware normalization paths; ones_like avoids that.
        if args.pad_to_multiple_of > 1 and orig_seq_len % args.pad_to_multiple_of != 0:
            pad_len = args.pad_to_multiple_of - (orig_seq_len % args.pad_to_multiple_of)
            pad_ids = torch.full((1, pad_len), args.pad_token_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=1)
        padded_seq_len = int(input_ids.shape[1])
        # Use a fully-active mask on the padded sequence to mirror MB's causal-only treatment.
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        device = next(model.parameters()).device
        input_ids_dev = input_ids.to(device)
        attention_mask_dev = attention_mask.to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids_dev, attention_mask=attention_mask_dev)

        # Save logits at both the last real (semantically meaningful) and last padded position
        # so the comparator can use either. compare.py samples at the last padded position; that
        # is the position where BF16 cosine is tightest. We keep last_pos_logits = last_real for
        # back-compat with the existing comparator field name.
        last_real_logits = out.logits[0, orig_seq_len - 1, :].float().cpu()
        last_padded_logits = out.logits[0, -1, :].float().cpu()
        last_pos_logits = last_real_logits
        vocab_size = int(last_pos_logits.shape[0])

        top_vals, top_ids = torch.topk(last_real_logits, k=args.top_k)
        top_token_strs = [tokenizer.decode([tid.item()]) for tid in top_ids]
        top_padded_vals, top_padded_ids = torch.topk(last_padded_logits, k=args.top_k)
        top_padded_strs = [tokenizer.decode([tid.item()]) for tid in top_padded_ids]
        top1_id = int(top_ids[0].item())
        top1_str = top_token_strs[0]

        logger.info(
            f"[{i + 1}/{len(prompts)}] id={prompt_id} seq_len={orig_seq_len} padded={padded_seq_len} "
            f"real_top1={top1_id} ({top1_str!r}) padded_top1={int(top_padded_ids[0].item())} "
            f"real_mean={last_real_logits.mean():.4f} padded_mean={last_padded_logits.mean():.4f}"
        )

        results.append(
            {
                "id": prompt_id,
                "prompt": prompt_text,
                "input_ids": input_ids[0].cpu(),
                "seq_len": orig_seq_len,
                "padded_seq_len": padded_seq_len,
                "last_pos_logits": last_pos_logits,
                "last_real_logits": last_real_logits,
                "last_padded_logits": last_padded_logits,
                "vocab_size": vocab_size,
                "top_k": args.top_k,
                "top_ids": top_ids.cpu(),
                "top_logits": top_vals.float().cpu(),
                "top_strs": top_token_strs,
                "top_padded_ids": top_padded_ids.cpu(),
                "top_padded_strs": top_padded_strs,
            }
        )

    artifact = {
        "source": "hf_transformers",
        "model_path": args.hf_model_path,
        "dtype": "bfloat16",
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)
    logger.info(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
