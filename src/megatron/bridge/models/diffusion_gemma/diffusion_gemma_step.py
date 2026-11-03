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

"""Megatron forward/loss step for DiffusionGemma uniform-state diffusion SFT.

Batch contract (prompt left-padded, canvas right-padded)::

    input_ids          [B, P]  encoder prompt, including image placeholder tokens
    attention_mask     [B, P]  1 for real prompt tokens, 0 for padding
    canvas_ids         [B, L]  clean target tokens x0, EOS-terminated
    canvas_mask        [B, L]  1 for scored target tokens, 0 for padding
    position_ids       [B, P]  optional
    pixel_values, image_position_ids, mm_token_type_ids  optional image inputs

Optional deterministic overrides, mainly for parity tests:
``diffusion_t`` [B], ``noisy_canvas_ids`` [B, L], ``self_conditioning_mask`` [B].
"""

from __future__ import annotations

from functools import partial
from typing import Any, Iterable

import torch
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_model_config

from megatron.bridge.models.diffusion_gemma.loss import diffusion_sft_loss, normalize_variant
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection


_BATCH_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "pixel_values",
    "image_position_ids",
    "mm_token_type_ids",
    "canvas_ids",
    "canvas_mask",
    "diffusion_t",
    "noisy_canvas_ids",
    "self_conditioning_mask",
)


def corrupt_canvas(
    x0: torch.Tensor,
    canvas_mask: torch.Tensor,
    *,
    vocab_size: int,
    noise_min: float = 1e-3,
    noise_max: float = 1.0 - 1e-3,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one time per example and replace valid tokens uniformly with probability ``t``."""
    if not 0.0 <= noise_min <= noise_max <= 1.0:
        raise ValueError("Expected 0 <= diffusion_noise_min <= diffusion_noise_max <= 1")
    t = torch.empty((x0.shape[0],), device=x0.device, dtype=torch.float32)
    t.uniform_(noise_min, noise_max, generator=generator)
    valid = canvas_mask.to(device=x0.device, dtype=torch.bool)
    corrupt = (torch.rand(x0.shape, device=x0.device, generator=generator) < t[:, None]) & valid
    random_tokens = torch.randint(0, vocab_size, x0.shape, device=x0.device, generator=generator)
    return torch.where(corrupt, random_tokens, x0), t


def diffusion_loss_function(
    output_tensor: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    variant: str,
    vocab_size: int,
    diffusion_weight_clip: float | None = None,
    check_for_nan_in_loss: bool = True,
    check_for_spiky_loss: bool = False,
    decoder_loss_weight: float = 1.0,
    encoder_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return Megatron's ``(sum_loss, denominator, metrics)`` contract.

    DiffGemma averages loss per example, so the denominator is the number of
    examples. Megatron divides by it (or reduces it globally when
    ``calculate_per_token_loss`` is enabled), preserving the reference mean.
    """
    logits, x0, xt, t, loss_mask = output_tensor[:5]
    per_example = diffusion_sft_loss(
        logits[..., :vocab_size],
        x0,
        xt,
        t,
        loss_mask,
        variant=variant,
        vocab_size=vocab_size,
        diffusion_weight_clip=diffusion_weight_clip,
    )
    decoder_sum = per_example.sum()
    loss = decoder_loss_weight * decoder_sum
    encoder_loss = output_tensor[5] if len(output_tensor) > 5 else None
    if encoder_loss is not None:
        loss = loss + encoder_loss_weight * encoder_loss * per_example.numel()
    rerun_state_machine = get_rerun_state_machine()
    if check_for_nan_in_loss:
        for rejection_func, label in ((torch.isnan, "NaN"), (torch.isinf, "Inf")):
            rerun_state_machine.validate_result(
                result=loss,
                rejection_func=rejection_func,
                message=f"found {label} in local DiffusionGemma loss calculation",
                tolerance=0.0,
                fatal=True,
            )
    if check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(rerun_state_machine.is_unexpectedly_large, threshold=10.0, context="loss"),
            message="Spiky DiffusionGemma loss",
            tolerance=0.0,
            fatal=False,
        )
    num_examples = torch.tensor(per_example.numel(), device=loss.device, dtype=torch.int)
    reporting = torch.cat((loss.detach().reshape(1), num_examples.reshape(1).to(loss.dtype)))
    metrics = {"lm loss": reporting}
    if encoder_loss is not None:
        metrics["diffusion loss"] = torch.stack((decoder_sum.detach(), num_examples.to(loss.dtype)))
        metrics["encoder loss"] = torch.stack((encoder_loss.detach() * num_examples, num_examples.to(loss.dtype)))
    return loss, num_examples, metrics


def get_batch_from_iterator(
    data_iterator: Iterable,
    *,
    is_first_pp_stage: bool = True,
    is_last_pp_stage: bool = True,
) -> dict[str, Any]:
    """Load a DiffusionGemma batch and move tensors to the current CUDA device."""
    if not (is_first_pp_stage and is_last_pp_stage):
        raise NotImplementedError("DiffusionGemma training currently requires pipeline_model_parallel_size=1")
    batch = next(data_iterator)
    missing = [key for key in ("input_ids", "attention_mask", "canvas_ids", "canvas_mask") if key not in batch]
    if missing:
        raise KeyError(f"DiffusionGemma batch is missing required keys: {missing}")
    return {
        key: batch[key].cuda(non_blocking=True) if isinstance(batch.get(key), torch.Tensor) else batch.get(key)
        for key in _BATCH_KEYS
        if key in batch
    }


_MICROBATCH_COUNTERS: dict[str, dict[str, int | None]] = {
    "train": {"step": None, "index": 0},
    "eval": {"step": None, "index": 0},
}


def next_microbatch_index(mode: str, step: int) -> int:
    """Return this phase's microbatch index, restarting whenever the step changes.

    Train and eval keep separate counters. A validation pass inside step ``N``
    must not shift the noise that training step ``N`` later draws, otherwise a
    run resumed from a checkpoint (fresh counters) would diverge from the
    uninterrupted run.
    """
    if mode not in _MICROBATCH_COUNTERS:
        raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
    counter = _MICROBATCH_COUNTERS[mode]
    if counter["step"] != step:
        counter["step"] = step
        counter["index"] = 0
    index = int(counter["index"])
    counter["index"] = index + 1
    return index


def noise_seed(*, seed: int, mode: str, step: int, microbatch: int, dp_rank: int) -> int:
    """Derive a deterministic 63-bit RNG seed for one microbatch on one DP rank."""
    phase = 0 if mode == "train" else 1
    value = seed * 1_000_003 + step * 10_007 + microbatch * 101 + dp_rank * 7_919 + phase * 104_729
    return value % (2**63 - 1)


def _noise_generator(state: GlobalState, model: Any) -> torch.Generator:
    """Return a CUDA RNG keyed by (seed, phase, step, microbatch, DP rank).

    TP/EP peers in one data-parallel replica draw identical noise; DP replicas
    draw distinct noise. Keying by training step makes resumed runs reproduce the
    original corruption without checkpointing generator state.
    """
    mode = "train" if getattr(model, "training", True) else "eval"
    step = int(getattr(getattr(state, "train_state", None), "step", 0))
    microbatch = next_microbatch_index(mode, step)
    pg = get_pg_collection(model)
    dp_rank = torch.distributed.get_rank(pg.dp) if torch.distributed.is_initialized() else 0
    generator = torch.Generator(device=f"cuda:{torch.cuda.current_device()}")
    generator.manual_seed(
        noise_seed(seed=int(state.cfg.rng.seed), mode=mode, step=step, microbatch=microbatch, dp_rank=dp_rank)
    )
    return generator


def _diffusion_setting(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def forward_step(state: GlobalState, data_iterator: Iterable, model: Any) -> tuple[tuple[torch.Tensor, ...], partial]:
    """Corrupt the target canvas, run one DiffusionGemma forward, and return its loss."""
    pg = get_pg_collection(model)
    timers = state.timers
    timers("batch-generator", log_level=2).start()
    batch = get_batch_from_iterator(
        data_iterator,
        is_first_pp_stage=is_pp_first_stage(pg.pp),
        is_last_pp_stage=is_pp_last_stage(pg.pp),
    )
    timers("batch-generator").stop()

    config = get_model_config(model)
    if getattr(config, "context_parallel_size", 1) != 1:
        raise NotImplementedError("DiffusionGemma training does not support context parallelism yet")
    if getattr(config, "sequence_parallel", False):
        tp_size = int(config.tensor_model_parallel_size)
        for key in ("input_ids", "canvas_ids"):
            if batch[key].shape[1] % tp_size:
                raise ValueError(
                    f"DiffusionGemma sequence parallelism requires {key} length divisible by TP={tp_size}; "
                    "left-pad prompts and right-pad canvases in the collator."
                )

    x0 = batch["canvas_ids"]
    canvas_mask = batch["canvas_mask"]
    vocab_size = int(getattr(config, "vocab_size", 0) or config.text_config.vocab_size)
    generator = None
    if batch.get("noisy_canvas_ids") is not None:
        xt = batch["noisy_canvas_ids"]
        t = batch["diffusion_t"].float()
    else:
        generator = _noise_generator(state, model)
        xt, t = corrupt_canvas(
            x0,
            canvas_mask,
            vocab_size=vocab_size,
            noise_min=float(_diffusion_setting(config, "diffusion_noise_min", 1e-3)),
            noise_max=float(_diffusion_setting(config, "diffusion_noise_max", 1.0 - 1e-3)),
            generator=generator,
        )

    self_conditioning_prob = float(_diffusion_setting(config, "diffusion_self_conditioning_prob", 0.0))
    self_conditioning_mask = batch.get("self_conditioning_mask")
    if self_conditioning_mask is None and self_conditioning_prob > 0.0:
        if generator is None:
            generator = _noise_generator(state, model)
        self_conditioning_mask = (
            torch.rand((x0.shape[0],), device=x0.device, generator=generator) < self_conditioning_prob
        )
    pixel_values = batch.get("pixel_values")
    if pixel_values is not None and torch.is_floating_point(pixel_values):
        pixel_values = pixel_values.to(config.params_dtype)

    forward_args = {
        "input_ids": batch["input_ids"],
        "decoder_input_ids": xt,
        "position_ids": batch.get("position_ids"),
        "pixel_values": pixel_values,
        "image_position_ids": batch.get("image_position_ids"),
        "mm_token_type_ids": batch.get("mm_token_type_ids"),
        "encoder_attention_mask": batch["attention_mask"],
        "self_conditioning_mask": self_conditioning_mask,
    }
    encoder_weight = float(_diffusion_setting(config, "diffusion_encoder_loss_weight", 0.0))
    with state.straggler_timer:
        if self_conditioning_mask is not None:
            # DiffusionGemma's first pass is intentionally detached.  Passing
            # an explicit mask to the second pass preserves zero signal for
            # unselected examples; zeroing logits would produce a uniform
            # (non-zero) mean embedding. Do not skip this pass when a local mask
            # is all-False: MoE expert-parallel all-to-all requires every rank to
            # execute the same number of forwards.
            with torch.no_grad():
                first = model(
                    **{
                        key: value
                        for key, value in forward_args.items()
                        if key != "self_conditioning_mask" and value is not None
                    }
                )
            forward_args["self_conditioning_logits"] = first.logits.detach()
        if encoder_weight > 0.0:
            forward_args["encoder_target_ids"] = x0
            forward_args["encoder_target_mask"] = canvas_mask
        output = model(**{key: value for key, value in forward_args.items() if value is not None})

    loss_function = partial(
        diffusion_loss_function,
        variant=normalize_variant(str(_diffusion_setting(config, "diffusion_loss_variant", "base-sft"))),
        vocab_size=vocab_size,
        diffusion_weight_clip=_diffusion_setting(config, "diffusion_weight_clip", None),
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
        decoder_loss_weight=float(_diffusion_setting(config, "diffusion_decoder_loss_weight", 1.0)),
        encoder_loss_weight=encoder_weight,
    )
    return (output.logits, x0, xt, t, canvas_mask, output.encoder_loss), loss_function
