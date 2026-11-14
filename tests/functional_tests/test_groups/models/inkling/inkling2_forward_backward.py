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

"""Forward and backward passes of the toy Inkling2 checkpoint at one parallel layout.

Runs a plain batch, a packed batch of two documents, and an image batch through the
pipeline schedule with gradients, then applies LoRA to the attention projections and
runs the plain batch again. Every loss must be finite and every step must produce a
finite, nonzero gradient; the image batch must also reach the image encoder. The last
pipeline stage writes the losses as JSON.

Launch with ``python -m torch.distributed.run --nproc_per_node=2``.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel import get_forward_backward_func

from megatron.bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA


SEQ = 32
IMAGE_ROWS, IMAGE_COLUMNS = 2, 3
LORA_TARGETS = ["linear_q", "linear_k", "linear_v", "linear_proj"]


def _batches(config: dict, device: torch.device) -> dict[str, dict]:
    """A plain, a packed, and an image batch with next-token labels, identical on every rank."""
    generator = torch.Generator().manual_seed(1234)
    vocab = config["text_config"]["unpadded_vocab_size"]
    image_token = config["image_token_id"]

    def tokens(count: int) -> torch.Tensor:
        return torch.randint(0, vocab, (count,), generator=generator)

    def batch(ids: torch.Tensor, loss_mask: torch.Tensor, **extra) -> dict:
        return {
            "input_ids": ids[:-1].unsqueeze(0).to(device),
            "position_ids": extra.pop("position_ids", torch.arange(SEQ).unsqueeze(0)).to(device),
            "attention_mask": None,
            "labels": ids[1:].unsqueeze(0).to(device),
            "loss_mask": loss_mask.unsqueeze(0).to(device),
            **extra,
        }

    plain = batch(tokens(SEQ + 1), torch.ones(SEQ))

    # Two documents of 12 and 20 tokens in one pack, in the THD format of the packed SFT recipes.
    lengths = (12, 20)
    cu_seqlens = torch.tensor([0, lengths[0], SEQ], dtype=torch.int32, device=device)
    packed = batch(
        tokens(SEQ + 1),
        torch.ones(SEQ),
        position_ids=torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0),
        packed_seq_params=PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        ),
    )

    # One image of 2 x 3 merged patches, one placeholder token each, after five text tokens.
    vision = config["vision_config"]
    side = vision["native_patch_size"] * vision["adapter_merge_size"]
    patches = IMAGE_ROWS * IMAGE_COLUMNS
    ids = torch.cat([tokens(5), torch.full((patches,), image_token), tokens(SEQ + 1 - 5 - patches)])
    pixels = torch.rand((patches, vision["temporal_patch_size"], side, side, 3), generator=generator) * 2 - 1
    image = batch(
        ids,
        (ids[1:] != image_token).float(),
        pixel_values=pixels.to(device, torch.bfloat16),
        image_grid_thw=torch.tensor([[1, IMAGE_ROWS, IMAGE_COLUMNS]]),
    )
    # The toy's image token is its first padded vocabulary row, whose logit is -inf, so a placeholder label would
    # give an infinite loss that the zero loss mask turns into NaN. Label those masked positions with token 0.
    image["labels"] = image["labels"].masked_fill(image["labels"] == image_token, 0)
    return {"plain": plain, "packed": packed, "image": image}


def _forward_step(data_iterator, model, **unused):
    batch = dict(next(data_iterator))
    loss_mask = batch.pop("loss_mask")
    output = model(**batch)

    def loss_func(output_tensor):
        mask = loss_mask.float().reshape(-1)
        loss = (output_tensor.float().reshape(-1) * mask).sum() / mask.sum()
        # The schedule scales the returned loss in place, so report a copy.
        return loss, {"loss": loss.detach().clone()}

    return output, loss_func


def _train_step(models: list, batch: dict, step: str) -> float | None:
    """Run one forward and backward pass; return the loss on the last stage and ``None`` elsewhere."""
    for model in models:
        model.zero_grad(set_to_none=True)
    outputs = get_forward_backward_func()(
        forward_step_func=_forward_step,
        data_iterator=iter([batch]),
        model=models,
        num_microbatches=1,
        seq_length=SEQ,
        micro_batch_size=1,
        forward_only=False,
    )
    if not parallel_state.is_pipeline_last_stage():
        return None
    loss = float(outputs[0]["loss"])
    if not math.isfinite(loss):
        raise RuntimeError(f"{step}: non-finite loss {loss}")
    return loss


def _check_gradients(models: list, step: str, *, expect_image_encoder: bool = False) -> None:
    """Require a finite, nonzero gradient norm over all ranks, and image encoder gradients if asked."""
    trainable, received = 0, []
    squares = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float64)
    for model in models:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            trainable += 1
            if parameter.grad is not None:
                received.append(name)
                squares += parameter.grad.double().square().sum()
    dist.all_reduce(squares)
    norm = math.sqrt(float(squares))
    if not received or not math.isfinite(norm) or norm == 0.0:
        raise RuntimeError(f"{step}: {len(received)} of {trainable} parameters got gradients, global norm {norm}")
    if (
        expect_image_encoder
        and parallel_state.is_pipeline_first_stage()
        and not any(".vision." in name for name in received)
    ):
        raise RuntimeError(f"{step}: the image encoder got no gradients")


def main() -> None:
    """Train each batch once at the requested layout, then with LoRA, and write the losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Toy Inkling2 checkpoint directory")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--pattern", default=None, help="Hybrid layer pattern whose '|' marks the stage boundaries")
    parser.add_argument("--output", required=True, help="JSON file for the losses")
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    bridge = AutoBridge.from_hf_pretrained(args.model, torch_dtype=torch.bfloat16)
    config = bridge.get_model_config()
    config.tensor_model_parallel_size = args.tp
    config.pipeline_model_parallel_size = args.pp
    config.expert_model_parallel_size = 1
    config.expert_tensor_parallel_size = 1
    config.sequence_parallel = args.tp > 1
    config.pipeline_dtype = torch.bfloat16
    # Without a DDP wrapper there is no main_grad buffer for fused weight-gradient accumulation.
    config.gradient_accumulation_fusion = False
    if args.pattern is not None:
        config.hybrid_layer_pattern = args.pattern
    models = bridge.get_model(config, load_weights=True, wrap_with_ddp=False)
    for model in models:
        model.train()

    hf_config = json.loads((Path(args.model) / "config.json").read_text())
    batches = _batches(hf_config, torch.device("cuda", torch.cuda.current_device()))
    losses = {}
    for name, batch in batches.items():
        losses[name] = _train_step(models, batch, name)
        _check_gradients(models, name, expect_image_encoder=name == "image")

    # LoRA starts with zero output projections, so the loss must not change; only the adapters train.
    models = LoRA(target_modules=LORA_TARGETS, dim=8, alpha=16)(models, training=True)
    trainable = [name for model in models for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(".adapter." not in name for name in trainable):
        raise RuntimeError(f"LoRA should train only its adapters, got {trainable[:8]}")
    lora_loss = _train_step(models, batches["plain"], "lora")
    _check_gradients(models, "lora")

    if (
        parallel_state.is_pipeline_last_stage()
        and parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_data_parallel_rank() == 0
    ):
        Path(args.output).write_text(json.dumps({"losses": losses, "lora_loss": lora_loss}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
