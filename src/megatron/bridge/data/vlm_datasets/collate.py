"""
Collation utilities for building VLM training batches from conversation examples.
"""

from PIL import Image  # noqa: F401  # may be used downstream by processors
import torch.nn.functional as F

import torch

from .token_utils import extract_skipped_token_ids


# Local message used when optional qwen_vl_utils dependency is missing
MISSING_QWEN_VL_UTILS_MSG = (
    "qwen_vl_utils is required for Qwen2.5 VL processing. Please `pip install qwen-vl-utils` or"
    " provide compatible vision preprocessing."
)

try:
    from qwen_vl_utils import process_vision_info

    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False


def create_loss_mask_with_start_of_response_token(input_ids, processor, start_of_response_token=None):
    r"""
    Create loss mask by finding start of turn token positions, similar to squad.py approach.

    Args:
        input_ids: List or tensor of token IDs for a single example
        processor: Processor/tokenizer to convert token string to ID
        start_of_response_token: String token that marks the start of turns (e.g., "<start_of_turn>model\n")

    Returns:
        loss_mask: List of 0/1 flags where 0 = masked (prompt), 1 = unmasked (response)
    """

    def find_sequence_in_list(input_ids, target_sequence):
        """Find the starting index of target_sequence in input_ids"""
        if not target_sequence:
            return -1
        for i in range(len(input_ids) - len(target_sequence) + 1):
            if input_ids[i : i + len(target_sequence)] == target_sequence:
                return i
        return -1

    tokenizer = getattr(processor, "tokenizer", processor)
    input_ids = input_ids.tolist()

    if start_of_response_token is None:
        return [1] * len(input_ids)

    if isinstance(start_of_response_token, str):
        start_of_response_token_ids = tokenizer(start_of_response_token, add_special_tokens=False)["input_ids"]
        first_occurrence = find_sequence_in_list(input_ids, start_of_response_token_ids)
        response_start = first_occurrence if first_occurrence >= 0 else 0
    else:
        response_start = 0

    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0
    loss_mask = [0] * response_start + [1] * (len(input_ids) - response_start)

    for i, token_id in enumerate(input_ids):
        if token_id == pad_token_id:
            loss_mask[i] = 0

    return loss_mask


def _gather_assistant_text_segments(example: dict) -> list[str]:
    """Extract assistant text segments from the structured conversation example.

    The example schema is expected to be {"conversation": [{"role": ..., "content": [...]} ...]} where
    content is a list of items like {"type": "text"|"image"|..., "text": "..."}.
    Returns a list of concatenated text strings, one per assistant turn.
    """
    texts: list[str] = []
    for turn in example.get("conversation", []):
        if turn.get("role") != "assistant":
            continue
        parts = turn.get("content", [])
        buf = []
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                    buf.append(p["text"])
        elif isinstance(parts, str):
            buf.append(parts)
        if buf:
            texts.append("".join(buf))
    return texts


def create_multiturn_loss_mask_by_search(example: dict, input_ids, processor, skipped_tokens: torch.Tensor) -> list[int]:
    """Tokenizer-agnostic masking via substring search of assistant texts.

    - Tokenize full conversation with processor already done -> input_ids
    - Extract assistant text strings from the structured example
    - For each assistant text, tokenize without special tokens and search sequentially
    - On success, unmask that span; otherwise leave masked
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    ids = input_ids.tolist()
    mask = [0] * len(ids)

    def try_mark(span_text: str, start_from: int) -> int:
        """Tokenize a span and mark its occurrence if found. Returns new search start index."""
        variants = [span_text, span_text + "\n"]
        for text in variants:
            span_tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not span_tokens:
                continue
            # naive sequential search from start_from
            for i in range(start_from, len(ids) - len(span_tokens) + 1):
                if ids[i : i + len(span_tokens)] == span_tokens:
                    for j in range(i, i + len(span_tokens)):
                        mask[j] = 1
                    return i + len(span_tokens)
        return start_from

    search_start = 0
    for asst_text in _gather_assistant_text_segments(example):
        search_start = try_mark(asst_text, search_start)

    # Ensure pad/skipped tokens are masked
    ids_t = torch.tensor(ids)
    for k, t in enumerate(ids_t):
        if t in skipped_tokens:
            mask[k] = 0
    return mask


def phi4_mm_collate_fn(examples, processor):
    """Collate function for Phi-4 MM model audio input"""

    # Extract conversations and audio data
    conversations = [example["conversation"] for example in examples]
    audios = [example["audio"] for example in examples]
    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
    audio_inputs = [(audio["array"], audio["sampling_rate"]) if isinstance(audio, dict) else audio for audio in audios]
    batch = processor(
        text=texts, audios=audio_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024
    )
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)

    loss_masks = []
    for i, conversation in enumerate(conversations):
        input_ids = batch["input_ids"][i].tolist()

        assistant_content = conversation[1]["content"]
        assistant_tokens = processor.tokenizer(assistant_content, add_special_tokens=False)["input_ids"]

        loss_mask = [0] * len(input_ids)
        for start_idx in range(len(input_ids) - len(assistant_tokens) + 1):
            if input_ids[start_idx : start_idx + len(assistant_tokens)] == assistant_tokens:
                for j in range(len(assistant_tokens)):
                    loss_mask[start_idx + j] = 1
                break
        loss_masks.append(loss_mask)

    max_len = max(len(mask) for mask in loss_masks)
    padded_loss_masks = [mask + [0] * (max_len - len(mask)) for mask in loss_masks]
    batch["loss_mask"] = torch.tensor(padded_loss_masks, dtype=torch.float)

    labels[batch["loss_mask"] == 0] = -100
    batch["labels"] = labels

    # Remove specified batch features if present
    for key in ["input_image_embeds", "image_sizes", "image_attention_mask"]:
        if key in batch:
            del batch[key]
    return batch


def qwen2_5_collate_fn(
    examples: list, processor, start_of_response_token: str = "<|im_start|>assistant\n"
) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2.5 VL model."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    skipped_tokens = extract_skipped_token_ids(processor)

    texts = [processor.apply_chat_template(example["conversation"], tokenize=False) for example in examples]
    # Build per-example images (list) and split by presence
    per_example_images = []
    has_images = []
    for example in examples:
        imgs = process_vision_info(example["conversation"])[0]
        if imgs is None:
            imgs = []
        elif not isinstance(imgs, list):
            imgs = [imgs]
        per_example_images.append(imgs)
        has_images.append(len(imgs) > 0)

    idx_with = [i for i, h in enumerate(has_images) if h]
    idx_without = [i for i, h in enumerate(has_images) if not h]

    batch_with = None
    batch_without = None

    if idx_with:
        texts_with = [texts[i] for i in idx_with]
        images_with = [per_example_images[i] for i in idx_with]
        batch_with = processor(
            text=texts_with,
            images=images_with,
            padding=True,
            return_tensors="pt",
        )

    if idx_without:
        texts_without = [texts[i] for i in idx_without]
        batch_without = processor(
            text=texts_without,
            padding=True,
            return_tensors="pt",
        )

    # Merge batches back to original order
    if batch_with is not None and batch_without is None:
        batch = batch_with
    elif batch_with is None and batch_without is not None:
        batch = batch_without
    else:
        # Both exist: pad to common max length and interleave rows
        pad_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0
        in_with = batch_with["input_ids"]
        in_without = batch_without["input_ids"]
        max_len = max(in_with.shape[1], in_without.shape[1])

        def pad_to(x, tgt_len):
            if x.shape[1] == tgt_len:
                return x
            pad_len = tgt_len - x.shape[1]
            return F.pad(x, (0, pad_len), value=pad_id)

        in_with = pad_to(in_with, max_len)
        in_without = pad_to(in_without, max_len)

        input_ids = torch.full((len(examples), max_len), pad_id, dtype=in_with.dtype)
        # Place rows
        for row, i in enumerate(idx_with):
            input_ids[i] = in_with[row]
        for row, i in enumerate(idx_without):
            input_ids[i] = in_without[row]

        batch = {"input_ids": input_ids}
        # Carry over vision tensors if present
        if "pixel_values" in batch_with:
            batch["pixel_values"] = batch_with["pixel_values"]
        if "image_grid_thw" in batch_with:
            batch["image_grid_thw"] = batch_with["image_grid_thw"]

    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels
    # Ensure position_ids exist for the model
    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
        )
    # Prefer general search-based masking using structured example content (not template-specific)
    loss_masks = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])  # type: ignore[arg-type]
    ]
    loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
    # Shift loss mask to align with next-token labels timeline
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    # Enforce label masking to match shifted loss_mask
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, -100)
    batch["loss_mask"] = loss_mask_t
    return batch


def nemotron_nano_v2_vl_collate_fn(examples: list, processor, start_of_response_token=None) -> dict[str, torch.Tensor]:
    """Collate function for Nemotron Nano V2 VL model."""
    skipped_tokens = extract_skipped_token_ids(processor)

    batch = processor.apply_chat_template(
        [example["conversation"] for example in examples],
        tokenize=True,
        padding=processor.tokenizer.pad_token is not None,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    )

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
        )

    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels
    loss_masks = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])  # type: ignore[arg-type]
    ]
    loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
    # Shift loss mask to align with next-token labels timeline
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, -100)
    batch["loss_mask"] = loss_mask_t
    return batch


def default_collate_fn(examples: list, processor, start_of_response_token=None) -> dict[str, torch.Tensor]:
    """Default collate function for VLM models."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    skipped_tokens = extract_skipped_token_ids(processor)

    batch = processor.apply_chat_template(
        [example["conversation"] for example in examples],
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    )

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
        )

    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels
    loss_masks = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])  # type: ignore[arg-type]
    ]
    loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
    # Shift loss mask to align with next-token labels timeline
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, -100)
    batch["loss_mask"] = loss_mask_t
    return batch


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "NemotronNanoVLV2Processor": nemotron_nano_v2_vl_collate_fn,
    "default": default_collate_fn,
}


