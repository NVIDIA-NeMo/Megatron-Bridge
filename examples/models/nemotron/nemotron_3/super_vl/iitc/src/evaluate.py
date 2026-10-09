# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matched HF generation and IITC scoring after native Bridge LoRA merge/export."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from rouge import Rouge
from transformers import AutoModelForImageTextToText, AutoProcessor


PROMPT_HEAD = (
    "Answer the question in English using the following text and images. Give a concise factual answer. "
    "Identify the single picture that supports your answer and cite it exactly once using [Picture N], "
    "replacing N with its number in the context. Do not repeat the citation. If uncertain, give your best "
    "answer and picture choice. Known content: "
)
RESUME_FIELDS = (
    "input_ids_sha256",
    "prompt_sha256",
    "image_paths",
    "image_sha256",
    "image_token_runs",
    "pixel_shapes",
    "gold",
)
TEST_SHA256 = "11700fdd10a90c7583099b3eef11e4cdd79f1281a50d4593386c5b061c8db809"


def score(response: str, row: dict) -> dict:
    """Use VEGA's citation extraction and rouge==1.0.1 ROUGE-L F1."""
    numbers = re.findall(r"\[Picture (\d+)\]", response)
    if not numbers:
        numbers = re.findall(r"\[Picture(\d+)\]", response)
    numbers = list(map(int, numbers)) if numbers else [0]
    correct = len(numbers) == 1 and numbers[0] == int(row["truth_fig_idx"]) + 1
    try:
        rouge_l = Rouge().get_scores(response, row["answer"])[0]["rouge-l"]["f"]
    except (ValueError, ZeroDivisionError):
        rouge_l = 0.0
    return {"picture_accuracy": float(correct), "rouge_l": rouge_l, "extracted_picture_numbers": numbers}


def prepare(row: dict, proc: Any, images_root: Path) -> tuple[dict, dict]:
    """Preserve every native text segment and image position, without truncation."""
    paths = re.findall(r"<img>(.*?)</img>", row["context"])
    assert paths == row["image_paths"]
    context = re.sub(r"<img>.*?</img>", "<image>", row["context"])
    query = PROMPT_HEAD + context + "\nQuestion:" + row["question"]
    prompt = proc.tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    images = []
    try:
        for path in paths:
            with Image.open(images_root / path) as image:
                images.append(image.convert("RGB"))
        encoded = proc(text=[prompt], images=images, return_tensors=None, padding=False, truncation=False)
    finally:
        for image in images:
            image.close()
    ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
    values = [torch.as_tensor(p) for p in encoded["pixel_values"]]
    values = [p.unsqueeze(0) if p.ndim == 3 else p for p in values]
    image_id = proc.tokenizer.convert_tokens_to_ids("<image>")
    assert image_id == 18
    positions = (ids[0] == image_id).nonzero().flatten().tolist()
    runs = []
    for position in positions:
        if runs and position == runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([position, position + 1])
    assert len(runs) == len(paths), (row["id"], len(runs), len(paths))
    audit = {
        "id": row["id"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "prompt_tokens": ids.shape[1],
        "image_paths": paths,
        "image_sha256": [hashlib.sha256((images_root / path).read_bytes()).hexdigest() for path in paths],
        "image_token_runs": runs,
        "image_tokens": len(positions),
        "pixel_shapes": [list(p.shape) for p in values],
        "truncated": False,
    }
    inputs = {
        "input_ids": ids.to("cuda:0"),
        "attention_mask": torch.as_tensor(encoded["attention_mask"], dtype=torch.long).to("cuda:0"),
        "pixel_values": [p.to("cuda:0") for p in values],
    }
    return inputs, audit


def read(path: Path) -> list[dict]:
    """Read JSONL, preserving published row order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_model(path: Path) -> torch.nn.Module:
    """Reload standalone HF weights, with the same compatibility shims for both models."""
    model, info = AutoModelForImageTextToText.from_pretrained(
        str(path),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={i: "70GiB" for i in range(8)},
        attn_implementation="flash_attention_2",
        output_loading_info=True,
    )
    assert not info.get("missing_keys") and not info.get("mismatched_keys") and not info.get("error_msgs"), info
    assert set(info.get("unexpected_keys", [])) <= {"backbone.embeddings.weight", "lm_head.weight"}, info
    assert not any("lora_" in name or ".adapter." in name for name, _ in model.named_parameters())
    project = model.vision_projector.forward

    def project_ragged(pixel_values, vision_model):
        features = [project(pv, vision_model).reshape(-1, model.config.llm_config.hidden_size) for pv in pixel_values]
        merged = torch.cat(features, dim=0).unsqueeze(0)
        assert merged.shape[1] == model.expected_image_tokens
        return merged

    language_forward = model.language_model.forward

    def selected_logits_forward(*args, **kwargs):
        kwargs["logits_to_keep"] = 1
        return language_forward(*args, **kwargs)

    model.vision_projector.forward = project_ragged
    model.language_model.forward = selected_logits_forward
    model.eval().requires_grad_(False)
    return model


def main() -> None:
    """Generate and score all 658 IITC-8K test responses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--processor", type=Path, required=True, help="Use the original processor for every checkpoint"
    )
    parser.add_argument("--source", type=Path, required=True, help="Published IITC_8k_test.json (JSONL)")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    assert importlib.metadata.version("rouge") == "1.0.1"
    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == TEST_SHA256, "Wrong published test file"
    rows = read(args.source)
    assert len(rows) == 658
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "model": str(args.model.resolve()),
        "processor": str(args.processor.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256((args.model / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "prompt_head": PROMPT_HEAD,
        "max_new_tokens": 512,
        "do_sample": False,
        "enable_thinking": False,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "rouge")},
        "prompt_protocol": "custom English single-citation; not the published prompt",
    }
    path = args.output / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Use a new output directory for a changed protocol/model"
    path.write_text(json.dumps(protocol, indent=2))
    target = args.output / "predictions.jsonl"
    records = read(target) if target.exists() else []
    done = {row["row_index"]: row for row in records}
    assert len(done) == len(records) and done.keys() <= set(range(658))
    model = load_model(args.model)
    proc = AutoProcessor.from_pretrained(args.processor, trust_remote_code=True)
    for index, row in enumerate(rows):
        inputs, audit = prepare(row, proc, args.images)
        if index in done:
            assert all(done[index][key] == audit[key] for key in RESUME_FIELDS if key != "gold")
            assert done[index]["gold"] == row["answer"]
            continue
        model.expected_image_tokens = audit["image_tokens"]
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=512, use_cache=True)
        ids = output[0, audit["prompt_tokens"] :].tolist()
        response = proc.tokenizer.decode(ids, skip_special_tokens=True)
        record = {
            **audit,
            "row_index": index,
            "title": row["title"],
            "prediction": response,
            "gold": row["answer"],
            "generated_token_ids": ids,
            "capped": len(ids) == 512,
            **score(response, row),
        }
        with target.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        done[index] = record
        logging.info("Completed %d/658", len(done))
    result = {metric: sum(row[metric] for row in done.values()) / 658 for metric in ("picture_accuracy", "rouge_l")}
    result.update(n=658, capped=sum(row["capped"] for row in done.values()))
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    (args.output / "COMPLETE").touch()
    logging.info("Complete: %s", json.dumps(result))


if __name__ == "__main__":
    main()
