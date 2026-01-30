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
Collation utilities for building LLM training batches from conversation examples.

This module mirrors the VLM collation behavior but is text-only and relies on a
HuggingFace tokenizer (no processor).
"""

from typing import Any

import torch

from megatron.bridge.data.datasets.utils import create_multiturn_loss_mask_by_search
from megatron.bridge.data.vlm_datasets.token_utils import extract_skipped_token_ids


def default_collate_fn(examples: list, tokenizer: Any) -> dict[str, torch.Tensor]:
    """Default collate function for text-only chat models using a tokenizer.

    The tokenizer must implement `apply_chat_template` compatible with HuggingFace
    chat templates.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must not be None for LLM collation")

    skipped_tokens = extract_skipped_token_ids(tokenizer)

    # Normalize content to strings to satisfy HF chat template expectations
    def _to_hf_chat(ex: dict) -> list[dict[str, Any]]:
        msgs = []
        for turn in ex.get("conversation", []):
            role = turn.get("role") or turn.get("sender") or "user"
            content = turn.get("content", "")
            if isinstance(content, list):
                buf = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                        buf.append(p["text"])
                    elif isinstance(p, str):
                        buf.append(p)
                content = "".join(buf)
            elif not isinstance(content, str):
                content = str(content)
            msgs.append({"role": role, "content": content})
        return msgs

    batch = tokenizer.apply_chat_template(
        [_to_hf_chat(example) for example in examples],
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    )
    # We force "attention_mask" to None and allow TE to generate the mask on the fly.
    batch["attention_mask"] = None
    batch["tokens"] = batch["input_ids"]

    # Build position ids when absent
    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
        )

    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    # Always mask out added/pad/special tokens
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels

    # Structured search-based masking using the example content
    loss_masks = [
        create_multiturn_loss_mask_by_search(example, input_ids, tokenizer, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])  # type: ignore[arg-type]
    ]
    loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
    # Align with next-token labels
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, -100)
    batch["loss_mask"] = loss_mask_t

    return batch


COLLATE_FNS = {
    "default": default_collate_fn,
}
