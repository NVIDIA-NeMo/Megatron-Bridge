# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Run the DiffusionGemma SFT recipe on synthetic long-context multimodal JSONL.

The script renders a large synthetic image, writes explicit train and
validation JSONL splits with a long text prompt and a multi-block completion,
and runs real ``pretrain()`` steps through the public processor,
``DiffusionGemmaJSONLDataset``, and the 26B EP=8 recipe. It reports per-step
losses, gradient norms, and peak memory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from training_loop_smoke import TrainingReport

from megatron.bridge.recipes.diffusion_gemma import diffusion_gemma_26b_sft_config
from megatron.bridge.training.diffusion_gemma_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def _write_image(path: Path, items: int) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((120, 100), "SYNTHETIC CATALOG", fill="black")
    for index in range(items):
        y = 200 + index * 42
        draw.rectangle((120, y, 150, y + 30), outline="black", fill=(40 * (index % 6), 90, 160))
        draw.text((180, y + 8), f"Item {index:03d}: shape={index % 5} size={index + 1}", fill="black")
    image.save(path)


def _row(image_name: str, items: int) -> dict:
    instructions = " ".join(
        f"Constraint {index}: keep field group {index} in the requested order." for index in range(80)
    )
    completion = {
        "title": "SYNTHETIC CATALOG",
        "items": [{"id": index, "shape": index % 5, "size": index + 1} for index in range(items)],
    }
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe the image as JSON. {instructions}"},
                    {"type": "image", "image": image_name},
                ],
            }
        ],
        "completion": json.dumps(completion),
    }


def main() -> None:
    """Create synthetic splits and execute the recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--items", type=int, default=12, help="Rows drawn in the image and listed in the completion.")
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--disable-recompute", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    _write_image(args.work_dir / "image.png", args.items)
    for split, items in (("train", args.items), ("validation", max(1, args.items // 2))):
        # Keep at least one example per data-parallel rank for validation.
        rows = [_row("image.png", items) for _ in range(8)]
        (args.work_dir / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    cfg = diffusion_gemma_26b_sft_config(
        train_path=str(args.work_dir / "train.jsonl"),
        validation_path=str(args.work_dir / "validation.jsonl"),
        hf_model_path=args.model_path,
    )
    cfg.train.train_iters = args.train_steps
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_decay_iters = args.train_steps
    cfg.validation.eval_interval = 2
    cfg.validation.eval_iters = 1
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.logger.tensorboard_dir = None
    cfg.logger.log_throughput = False
    cfg.rng.seed = 1234
    if args.disable_recompute:
        cfg.model.recompute_granularity = None
        cfg.model.recompute_method = None
        cfg.model.recompute_num_layers = None
    pretrain(cfg, forward_step, callbacks=[TrainingReport(args.report, record_performance=True)])


if __name__ == "__main__":
    main()
