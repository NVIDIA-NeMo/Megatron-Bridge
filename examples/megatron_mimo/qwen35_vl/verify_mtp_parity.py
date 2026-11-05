#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Verify that MegatronMIMO computes Qwen3.5-VL MTP exactly like the standard path.

One Hugging Face checkpoint is loaded twice:

- ``standard`` loads it into the standard Qwen3.5-VL model on one GPU.
- ``mimo`` loads it into MegatronMIMO with the language model on rank 0 and the vision encoder
  on rank 1, then runs the training pipeline schedule and MegatronMIMO forward step.

Both run one forward and backward pass on the same synthetic image-text batch with the loss on
every token. Each saves fingerprints of the loaded weights, the per-token language-model loss, the
MTP loss and all gradients to ``--output-dir``. ``compare`` checks them against each other and
exits with status 1 on a mismatch.

MegatronMIMO does not condition MTP on image-placeholder tokens (``mtp_input_mask``); the standard
path does. The standard step therefore applies that mask to the run it compares, and also reports
the MTP loss without it.

Both models run with tensor and pipeline parallel size 1, so the model must fit on one GPU:

    uv run python -m torch.distributed.run --nproc_per_node=1 \\
        examples/megatron_mimo/qwen35_vl/verify_mtp_parity.py standard \\
        --hf-model Qwen/Qwen3.5-0.8B --output-dir /tmp/qwen35_vl_mtp_parity
    uv run python -m torch.distributed.run --nproc_per_node=2 \\
        examples/megatron_mimo/qwen35_vl/verify_mtp_parity.py mimo \\
        --hf-model Qwen/Qwen3.5-0.8B --output-dir /tmp/qwen35_vl_mtp_parity
    uv run python examples/megatron_mimo/qwen35_vl/verify_mtp_parity.py compare \\
        --output-dir /tmp/qwen35_vl_mtp_parity

The saved gradients take about twice the model size in bfloat16.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)

IMAGES = "images"
ENCODER = "qwen_visual"
LANGUAGE = "language"


def build_batch(hf_config: Any, *, seq_length: int, image_grid_thw: tuple[int, int, int], seed: int) -> dict:
    """Build one synthetic single-image micro-batch in the MegatronMIMO format.

    Every next token is supervised, so the MTP loss also covers positions conditioned on image tokens.

    Args:
        hf_config: Hugging Face Qwen3.5-VL config.
        seq_length: Total tokens, image placeholders included.
        image_grid_thw: Image grid in patches before the spatial merge.
        seed: Seed for the random text tokens and pixel patches.

    Returns:
        A batch with ``input_ids``, MRoPE ``position_ids``, ``labels``, ``loss_mask`` and the
        vision encoder inputs under ``modality_inputs``.
    """
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index

    vision = hf_config.vision_config
    merge = vision.spatial_merge_size
    frames, height, width = image_grid_thw
    if height % merge or width % merge:
        raise ValueError(f"Image grid height and width must be multiples of spatial_merge_size={merge}.")
    num_patches = frames * height * width
    num_image_tokens = num_patches // merge**2
    num_text_tokens = seq_length - num_image_tokens - 2
    if num_text_tokens < 2:
        raise ValueError(f"--seq-length {seq_length} leaves no room for text around {num_image_tokens} image tokens.")

    special_token_ids = (
        hf_config.image_token_id,
        hf_config.video_token_id,
        hf_config.vision_start_token_id,
        hf_config.vision_end_token_id,
    )
    generator = torch.Generator().manual_seed(seed)
    text = torch.randint(0, min(special_token_ids), (num_text_tokens,), generator=generator)
    prompt_length = num_text_tokens // 4
    input_ids = torch.cat(
        [
            text[:prompt_length],
            torch.tensor([hf_config.vision_start_token_id]),
            torch.full((num_image_tokens,), hf_config.image_token_id),
            torch.tensor([hf_config.vision_end_token_id]),
            text[prompt_length:],
        ]
    ).unsqueeze(0)
    grid_thw = torch.tensor([image_grid_thw])
    position_ids, _ = get_rope_index(
        merge,
        hf_config.image_token_id,
        hf_config.video_token_id,
        hf_config.vision_start_token_id,
        input_ids=input_ids,
        image_grid_thw=grid_thw,
        attention_mask=torch.ones_like(input_ids),
    )
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = 0
    loss_mask = torch.ones(input_ids.shape, dtype=torch.float)
    loss_mask[:, -1] = 0.0
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": None,
        "labels": labels,
        "loss_mask": loss_mask,
        "modality_inputs": {
            IMAGES: {
                ENCODER: {
                    "hidden_states": torch.randn(num_patches, patch_dim, generator=generator),
                    "grid_thw": grid_thw,
                }
            }
        },
    }


def _weight_fingerprints(module: torch.nn.Module) -> dict[str, str]:
    """Return dtype, shape and a SHA-256 of every tensor in the module's state dict."""
    fingerprints = {}
    for name, tensor in module.state_dict().items():
        if not isinstance(tensor, torch.Tensor) or name.endswith("_extra_state"):
            continue
        data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        fingerprints[name] = f"{tensor.dtype}:{tuple(tensor.shape)}:{hashlib.sha256(data).hexdigest()}"
    return fingerprints


def _gradients(module: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if param.grad is None else param.grad.detach().cpu() for name, param in module.named_parameters()
    }


def _pop_mtp_losses() -> list[float]:
    """Return this step's MTP loss per depth and clear MCore's tracker."""
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    losses = MTPLossLoggingHelper.tracker["loss_values"].tolist()
    MTPLossLoggingHelper.clean_metrics_in_tracker()
    return losses


def _require_mtp(provider: Any, hf_model: str) -> None:
    if not getattr(provider, "mtp_num_layers", None):
        raise SystemExit(f"{hf_model} has no MTP layers to verify.")


def _load_batch(args: argparse.Namespace) -> tuple[Any, dict]:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(args.hf_model)
    batch = build_batch(hf_config, seq_length=args.seq_length, image_grid_thw=args.image_grid_thw, seed=args.seed)
    return hf_config, batch


def _init_distributed(world_size: int, step: str) -> None:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    if dist.get_world_size() != world_size:
        raise SystemExit(f"The {step} step needs exactly {world_size} rank(s) (--nproc_per_node={world_size}).")


def run_standard(args: argparse.Namespace) -> None:
    """Run the standard Qwen3.5-VL model with and without MegatronMIMO's MTP input mask."""
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
    from megatron.core.utils import unwrap_model

    from megatron.bridge import AutoBridge

    hf_config, batch = _load_batch(args)
    _init_distributed(1, "standard")
    provider = AutoBridge.from_hf_pretrained(args.hf_model).to_megatron_provider(load_weights=True)
    _require_mtp(provider, args.hf_model)
    # Without DDP buffers, Transformer Engine must write plain .grad tensors.
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    provider.initialize_model_parallel(seed=args.seed)
    (wrapped,) = provider.provide_distributed_model(wrap_with_ddp=False)
    wrapped.train()
    model = unwrap_model(wrapped)

    batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    encoder_inputs = batch["modality_inputs"][IMAGES][ENCODER]
    result: dict[str, Any] = {
        "weights": {
            LANGUAGE: _weight_fingerprints(model.language_model),
            IMAGES: _weight_fingerprints(model.vision_model),
        }
    }
    for variant, mtp_input_mask in (("masked", batch["input_ids"] != hf_config.image_token_id), ("unmasked", None)):
        wrapped.zero_grad(set_to_none=True)
        # The pipeline schedule sets this scale for one micro-batch without loss scaling.
        MTPLossAutoScaler.set_loss_scale(torch.ones(1, device="cuda"))
        per_token = wrapped(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
            pixel_values=encoder_inputs["hidden_states"].cuda(),
            image_grid_thw=encoder_inputs["grid_thw"].cuda(),
            mtp_input_mask=mtp_input_mask,
        )
        lm_loss = (per_token.float() * batch["loss_mask"]).sum() / batch["loss_mask"].sum()
        lm_loss.backward()
        result[variant] = {
            "per_token": per_token.detach().float().cpu(),
            "lm_loss": lm_loss.item(),
            "mtp_losses": _pop_mtp_losses(),
        }
        if variant == "masked":
            result[variant]["grads"] = {
                LANGUAGE: _gradients(model.language_model),
                IMAGES: _gradients(model.vision_model),
            }
        logger.info(f"standard ({variant}): lm loss {lm_loss.item():.6f}, mtp losses {result[variant]['mtp_losses']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_dir / "standard.pt")
    dist.destroy_process_group()


def run_mimo(args: argparse.Namespace) -> None:
    """Run MegatronMIMO through the training pipeline schedule and MIMO forward step."""
    from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
    from megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_without_interleaving
    from megatron.core.utils import get_model_config

    from megatron.bridge.models.megatron_mimo import build_megatron_mimo_model
    from megatron.bridge.models.megatron_mimo.conversion import MegatronMIMOBridge
    from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
        MegatronMIMOParallelismConfig,
        ModuleParallelismConfig,
    )
    from megatron.bridge.training.megatron_mimo_parallel_utils import build_pg_collection_for_schedule
    from megatron.bridge.training.megatron_mimo_step import forward_step

    _, batch = _load_batch(args)
    _init_distributed(2, "mimo")
    parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms={
            name: ModuleParallelismConfig(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                data_parallel_size=1,
                rank_offset=rank_offset,
            )
            for name, rank_offset in ((LANGUAGE, 0), (IMAGES, 1))
        }
    )
    bridge = MegatronMIMOBridge.from_hf_pretrained(args.hf_model, parallelism_config=parallelism_config)
    provider = bridge.to_megatron_mimo_provider()
    _require_mtp(provider.standard_provider, args.hf_model)
    # Without DDP buffers, Transformer Engine must write plain .grad tensors.
    provider.standard_provider.gradient_accumulation_fusion = False
    model, infra = build_megatron_mimo_model(
        provider, bf16=True, seed=args.seed, wrap_with_ddp=False, data_parallel_random_init=False
    )
    bridge.load_hf_weights(model, infra=infra)
    model.train()

    config = get_model_config(model)
    config.pipeline_dtype = torch.bfloat16
    communicator = MultiModulePipelineCommunicator(
        infra.module_to_grid_map,
        infra.topology,
        config,
        dim_mapping={"s": 0, "b": 1, "h": 2},
        module_output_ndim=infra.module_output_ndim,
    )
    # forward_step reads only these training settings from the global state.
    state = SimpleNamespace(
        cfg=SimpleNamespace(
            dataset=SimpleNamespace(enable_in_batch_packing=False, megatron_mimo_scalable_dp=False),
            rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=False, check_for_spiky_loss=False),
        )
    )
    per_token_losses = []

    def recording_forward_step(data_iterator, mimo_model):
        output, loss_func = forward_step(state, data_iterator, mimo_model)
        if loss_func is not None:
            per_token_losses.append(output.detach().float().cpu())
        return output, loss_func

    losses = forward_backward_pipelining_without_interleaving(
        forward_step_func=recording_forward_step,
        data_iterator=iter([batch]),
        model=[model],
        num_microbatches=1,
        seq_length=args.seq_length,
        micro_batch_size=1,
        forward_only=False,
        p2p_communicator=communicator,
        pg_collection=build_pg_collection_for_schedule(infra),
    )

    if model.language_model is not None:
        part, module = LANGUAGE, model.language_model
    else:
        part, module = IMAGES, model.modality_submodules[IMAGES].encoders[ENCODER]
    result: dict[str, Any] = {"weights": _weight_fingerprints(module), "grads": _gradients(module)}
    if part == LANGUAGE:
        loss_sum, num_tokens = losses[0]["lm loss"].view(-1).tolist()
        result.update(per_token=per_token_losses[0], lm_loss=loss_sum / num_tokens, mtp_losses=_pop_mtp_losses())
        logger.info(f"MegatronMIMO: lm loss {result['lm_loss']:.6f}, mtp losses {result['mtp_losses']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_dir / f"mimo_{part}.pt")
    dist.barrier()
    dist.destroy_process_group()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def _relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def _is_finite(value: float | torch.Tensor) -> bool:
    return bool(torch.isfinite(torch.as_tensor(value)).all())


def compare_results(output_dir: Path, *, min_cosine: float, loss_rtol: float) -> list[str]:
    """Compare the standard and MegatronMIMO outputs saved in ``output_dir``.

    Every compared loss and gradient must also be finite: NaN or infinity fails the
    check even when both paths produce the same value.

    Args:
        output_dir: Directory written by the standard and mimo steps.
        min_cosine: Minimum cosine similarity for per-token losses and every gradient.
        loss_rtol: Maximum relative difference of the language-model and MTP losses.

    Returns:
        A description of each failed check; empty when the paths match.
    """
    standard = torch.load(output_dir / "standard.pt", weights_only=False)
    mimo = {part: torch.load(output_dir / f"mimo_{part}.pt", weights_only=False) for part in (LANGUAGE, IMAGES)}
    masked = standard["masked"]
    failures = []

    for part in (LANGUAGE, IMAGES):
        expected, actual = standard["weights"][part], mimo[part]["weights"]
        differing = sorted(name for name in expected.keys() | actual.keys() if expected.get(name) != actual.get(name))
        logger.info(f"{part} weights: {len(expected)} tensors, {len(differing)} differ")
        if differing:
            failures.append(f"{part} weights differ: {differing[:5]}")

    per_token_cosine = _cosine(masked["per_token"], mimo[LANGUAGE]["per_token"])
    lm_difference = _relative_difference(masked["lm_loss"], mimo[LANGUAGE]["lm_loss"])
    logger.info(
        f"lm loss: standard {masked['lm_loss']:.6f}, MegatronMIMO {mimo[LANGUAGE]['lm_loss']:.6f}; "
        f"per-token cosine {per_token_cosine:.8f}"
    )
    lm_values = (masked["lm_loss"], mimo[LANGUAGE]["lm_loss"], masked["per_token"], mimo[LANGUAGE]["per_token"])
    if not all(_is_finite(value) for value in lm_values):
        failures.append("lm loss is not finite")
    elif per_token_cosine < min_cosine or lm_difference > loss_rtol:
        failures.append(f"lm loss differs: per-token cosine {per_token_cosine:.6f}, relative {lm_difference:.2e}")
    for depth, (expected, actual) in enumerate(zip(masked["mtp_losses"], mimo[LANGUAGE]["mtp_losses"]), start=1):
        logger.info(f"mtp_{depth} loss: standard {expected:.6f}, MegatronMIMO {actual:.6f}")
        if not (_is_finite(expected) and _is_finite(actual)):
            failures.append(f"mtp_{depth} loss is not finite: {expected} vs {actual}")
        elif _relative_difference(expected, actual) > loss_rtol:
            failures.append(f"mtp_{depth} loss differs: {expected:.6f} vs {actual:.6f}")
    if len(masked["mtp_losses"]) != len(mimo[LANGUAGE]["mtp_losses"]):
        failures.append(f"MTP depth differs: {len(masked['mtp_losses'])} vs {len(mimo[LANGUAGE]['mtp_losses'])}")

    for part in (LANGUAGE, IMAGES):
        expected, actual = masked["grads"][part], mimo[part]["grads"]
        identical, worst = 0, (1.0, "")
        for name in sorted(expected.keys() | actual.keys()):
            a, b = expected.get(name), actual.get(name)
            if a is None and b is None:
                continue
            if a is None or b is None:
                failures.append(f"{part} gradient {name} exists in only one path")
                continue
            # torch.equal treats matching infinities as equal, so check finiteness first.
            if not (_is_finite(a) and _is_finite(b)):
                failures.append(f"{part} gradient {name} is not finite")
                continue
            if torch.equal(a, b):
                identical += 1
                continue
            cosine = 0.0 if min(a.float().norm(), b.float().norm()) == 0 else _cosine(a, b)
            worst = min(worst, (cosine, name))
            if cosine < min_cosine:
                failures.append(f"{part} gradient {name} differs: cosine {cosine:.6f}")
        logger.info(
            f"{part} gradients: {len(expected)} parameters, {identical} bitwise identical, "
            f"lowest cosine {worst[0]:.8f} {worst[1]}"
        )

    unmasked = standard["unmasked"]["mtp_losses"]
    logger.info(
        f"For reference, the standard path without the MTP input mask gives mtp losses {unmasked} "
        f"(MegatronMIMO {mimo[LANGUAGE]['mtp_losses']})."
    )
    return failures


def _image_grid(value: str) -> tuple[int, int, int]:
    frames, height, width = (int(part) for part in value.split(","))
    return frames, height, width


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="step", required=True)
    for step in ("standard", "mimo"):
        subparser = subparsers.add_parser(step)
        subparser.add_argument("--hf-model", required=True, help="Hugging Face Qwen3.5-VL model id or local path")
        subparser.add_argument("--output-dir", required=True)
        subparser.add_argument("--seq-length", type=int, default=64)
        subparser.add_argument(
            "--image-grid-thw", type=_image_grid, default=(1, 4, 4), help="Image grid in patches, as T,H,W"
        )
        subparser.add_argument("--seed", type=int, default=1234)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--min-cosine", type=float, default=0.999)
    compare.add_argument("--loss-rtol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    """Run one verification step."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    if args.step == "standard":
        run_standard(args)
    elif args.step == "mimo":
        run_mimo(args)
    else:
        failures = compare_results(Path(args.output_dir), min_cosine=args.min_cosine, loss_rtol=args.loss_rtol)
        for failure in failures:
            logger.error(f"FAIL: {failure}")
        logger.info("FAIL" if failures else "PASS: MegatronMIMO matches the standard path")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
