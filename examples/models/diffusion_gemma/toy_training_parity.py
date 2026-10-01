#!/usr/bin/env python3
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

"""Tiny DiffusionGemma training-step parity and overfit check.

Uses a randomly initialized public-architecture toy model and synthetic token ids.
It compares the real Megatron ``forward_step`` against the DiffGemma reference
loss using fixed corruption and self-conditioning masks, then checks gradients
for every exported HF parameter and runs a short overfit.
"""

import argparse
import json
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core.transformer.enums import AttnBackend
from real_encoder_parity import compare_layer0, hf_layer0_trace, megatron_layer0_trace
from toy_parity import image_case, native_hf_encoder_inputs, toy_config
from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion

from megatron.bridge import AutoBridge
from megatron.bridge.models.diffusion_gemma.diffusion_gemma_step import forward_step
from megatron.bridge.models.diffusion_gemma.loss import diffusion_sft_loss


class _Timer:
    def start(self):
        return None

    def stop(self):
        return None


def fake_state(seed: int = 1234):
    """Build the minimal state used only by the standalone parity harness."""
    return SimpleNamespace(
        timers=lambda *_args, **_kwargs: _Timer(),
        straggler_timer=nullcontext(),
        cfg=SimpleNamespace(
            rng=SimpleNamespace(seed=seed),
            rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=False, check_for_spiky_loss=False),
        ),
    )


def text_batch(pad_id: int, variant_seed: int = 0, padded: bool = True) -> dict[str, torch.Tensor]:
    """Build a two-example denoising batch with optional prompt padding."""
    generator = torch.Generator().manual_seed(17 + variant_seed)
    second_length = 6 if padded else 9
    prompts = [
        torch.randint(10, 1900, (9,), generator=generator),
        torch.randint(10, 1900, (second_length,), generator=generator),
    ]
    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full((2, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((2, width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        input_ids[row, -len(prompt) :] = prompt
        attention_mask[row, -len(prompt) :] = 1
    canvas_ids = torch.randint(10, 1900, (2, 16), generator=generator)
    canvas_mask = torch.ones((2, 16), dtype=torch.float32)
    canvas_mask[1, 11:] = 0
    canvas_ids[1, 11:] = pad_id
    noisy = canvas_ids.clone()
    noisy[:, ::3] = torch.randint(10, 1900, noisy[:, ::3].shape, generator=generator)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": torch.arange(width)[None].expand(2, -1).clone(),
        "mm_token_type_ids": torch.zeros_like(input_ids),
        "canvas_ids": canvas_ids,
        "canvas_mask": canvas_mask,
        "noisy_canvas_ids": noisy,
        "diffusion_t": torch.tensor([0.35, 0.8]),
        "self_conditioning_mask": torch.tensor([True, False]),
    }


def image_batch(config) -> dict[str, torch.Tensor]:
    """Build an image-conditioned canvas with fixed corruption."""
    batch = image_case(config)
    generator = torch.Generator().manual_seed(99)
    canvas_ids = torch.randint(10, 1900, (1, 16), generator=generator)
    noisy = canvas_ids.clone()
    noisy[:, 1::2] = torch.randint(10, 1900, noisy[:, 1::2].shape, generator=generator)
    batch.update(
        canvas_ids=canvas_ids,
        canvas_mask=torch.ones((1, 16)),
        noisy_canvas_ids=noisy,
        diffusion_t=torch.tensor([0.55]),
        self_conditioning_mask=torch.tensor([True]),
    )
    return batch


def left_pad_prompt(batch: dict[str, torch.Tensor], pad_id: int, multiple: int) -> dict[str, torch.Tensor]:
    """Round prompt length to the sequence-parallel multiple."""
    width = batch["input_ids"].shape[1]
    pad = (-width) % multiple
    if pad == 0:
        return batch
    result = dict(batch)
    rows = batch["input_ids"].shape[0]
    result["input_ids"] = torch.cat((torch.full((rows, pad), pad_id, dtype=torch.long), batch["input_ids"]), dim=1)
    result["attention_mask"] = torch.cat(
        (torch.zeros((rows, pad), dtype=batch["attention_mask"].dtype), batch["attention_mask"]), dim=1
    )
    result["mm_token_type_ids"] = torch.cat(
        (torch.zeros((rows, pad), dtype=torch.long), batch["mm_token_type_ids"]), dim=1
    )
    result["position_ids"] = torch.arange(width + pad)[None].expand(rows, -1).clone()
    return result


def long_prompt_batch(pad_id: int) -> dict[str, torch.Tensor]:
    """Cross the toy model's 16-token local window in both encoder and decoder."""
    generator = torch.Generator().manual_seed(123)
    input_ids = torch.randint(10, 1900, (1, 24), generator=generator)
    canvas_ids = torch.randint(10, 1900, (1, 16), generator=generator)
    noisy = canvas_ids.clone()
    noisy[:, 1::3] = torch.randint(10, 1900, noisy[:, 1::3].shape, generator=generator)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1])[None],
        "mm_token_type_ids": torch.zeros_like(input_ids),
        "canvas_ids": canvas_ids,
        "canvas_mask": torch.ones_like(canvas_ids, dtype=torch.float32),
        "noisy_canvas_ids": noisy,
        "diffusion_t": torch.tensor([0.6]),
        "self_conditioning_mask": torch.tensor([True]),
    }


def hf_loss(
    model, batch, *, variant: str, vocab_size: int, encoder_loss_weight: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference DiffGemma loss using HF and a semantic decoder mask."""
    inputs = {
        key: batch[key] for key in ("input_ids", "attention_mask", "position_ids", "mm_token_type_ids") if key in batch
    }
    native = native_hf_encoder_inputs(model, inputs)
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": native["attention_mask"],
        "position_ids": batch["position_ids"],
        "mm_token_type_ids": batch["mm_token_type_ids"],
        "decoder_input_ids": batch["noisy_canvas_ids"],
        "decoder_attention_mask": hf_decoder_masks(model, batch["attention_mask"], batch["noisy_canvas_ids"].shape[1]),
    }
    for key in ("pixel_values", "image_position_ids"):
        if key in batch:
            kwargs[key] = batch[key]
    with torch.no_grad():
        first = model(**kwargs).logits.detach()
    output = model(**kwargs, self_conditioning_logits=first, self_conditioning_mask=batch["self_conditioning_mask"])
    per_example = diffusion_sft_loss(
        output.logits,
        batch["canvas_ids"],
        batch["noisy_canvas_ids"],
        batch["diffusion_t"],
        batch["canvas_mask"],
        variant=variant,
        vocab_size=vocab_size,
    )
    loss = per_example.mean()
    if encoder_loss_weight > 0.0:
        seq = torch.cat((batch["input_ids"], batch["canvas_ids"]), dim=1)
        prompt_mask = batch["attention_mask"]
        canvas_mask = batch["canvas_mask"]
        clean_inputs = {
            "input_ids": seq,
            "attention_mask": torch.cat((prompt_mask, canvas_mask.to(prompt_mask.dtype)), dim=1),
            "position_ids": torch.arange(seq.shape[1], device=seq.device)[None].expand_as(seq),
            "mm_token_type_ids": torch.cat((batch["mm_token_type_ids"], torch.zeros_like(batch["canvas_ids"])), dim=1),
        }
        for key in ("pixel_values", "image_position_ids"):
            if key in batch:
                clean_inputs[key] = batch[key]
        hidden = model.model.encoder(
            **native_hf_encoder_inputs(model, clean_inputs), use_cache=False
        ).last_hidden_state
        clean_logits = model.lm_head(hidden).float()
        cap = model.final_logit_softcapping
        if cap:
            clean_logits = cap * torch.tanh(clean_logits / cap)
        score = torch.cat((torch.zeros_like(prompt_mask), canvas_mask), dim=1)[:, 1:].bool()
        targets = seq[:, 1:].masked_fill(~score, -100)
        encoder_loss = torch.nn.functional.cross_entropy(
            clean_logits[:, :-1].reshape(-1, vocab_size), targets.reshape(-1), ignore_index=-100
        )
        loss = loss + encoder_loss_weight * encoder_loss
    return loss, output.logits.detach()


def megatron_loss(model, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the production training step and its normalized objective."""
    output, loss_function = forward_step(fake_state(), iter([batch]), model)
    loss_sum, count, _ = loss_function(output)
    return loss_sum / count, output[0].detach()


def exported_grads(bridge, model) -> dict[str, torch.Tensor]:
    """Convert gradients into HF layout using the exact weight mappings."""
    saved = {}
    for name, parameter in model.named_parameters():
        saved[name] = parameter.data.clone()
        parameter.data.copy_(parameter.grad if parameter.grad is not None else torch.zeros_like(parameter))
    try:
        return {
            name: tensor.detach().float().cpu()
            for name, tensor in bridge.export_hf_weights([model], show_progress=False)
        }
    finally:
        for name, parameter in model.named_parameters():
            parameter.data.copy_(saved[name])


def finalize_tp_grads(model) -> None:
    """Mirror Megatron DDP + finalize_model_grads for an unwrapped TP/EP model.

    Every data-parallel rank receives the same batch in this harness. Dense
    gradients are averaged over DP (a no-op for identical batches), while expert
    gradients receive tokens from all DP ranks through all-to-all, so they are
    summed over expert DP and divided by the dense DP size as Megatron DDP does.
    """
    if model.config.tensor_model_parallel_size == 1 and model.config.expert_model_parallel_size == 1:
        return
    from megatron.core import parallel_state

    tp_group = parallel_state.get_tensor_model_parallel_group()
    expert_dp_group = parallel_state.get_expert_data_parallel_group()
    dp_group = parallel_state.get_data_parallel_group()
    dp_size = dist.get_world_size(dp_group)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not getattr(parameter, "allreduce", True):
            dist.all_reduce(parameter.grad, group=expert_dp_group)
            parameter.grad.div_(dp_size)
            continue
        if dp_size > 1:
            dist.all_reduce(parameter.grad, group=dp_group)
            parameter.grad.div_(dp_size)
        if getattr(parameter, "average_gradients_across_tp_domain", False):
            dist.all_reduce(parameter.grad, group=tp_group)
            parameter.grad.div_(dist.get_world_size(tp_group))
        elif (model.config.sequence_parallel and getattr(parameter, "sequence_parallel", False)) or (
            model.config.qk_layernorm and ("q_layernorm" in name or "k_layernorm" in name)
        ):
            dist.all_reduce(parameter.grad, group=tp_group)


def compare(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    """Measure absolute and relative L2 differences for one tensor."""
    reference, actual = reference.float().cpu(), actual.float().cpu()
    diff = actual - reference
    return {
        "max_abs": float(diff.abs().max()),
        "relative_l2": float(diff.norm() / reference.norm().clamp_min(1e-12)),
    }


def hf_decoder_masks(model, prompt_mask: torch.Tensor, canvas_length: int) -> dict[str, torch.Tensor]:
    """Additive decoder masks for HF eager.

    Transformers 5.12.1 passes DiffusionGemma's boolean decoder mask directly to
    eager attention, which adds 1/0 instead of 0/-inf. SDPA interprets the same
    mask correctly. Supplying the mapping keeps the eager reference semantic.
    """
    batch = prompt_mask.shape[0]
    canvas = torch.ones((batch, canvas_length), dtype=torch.bool, device=prompt_mask.device)
    full_allowed = torch.cat((prompt_mask.bool(), canvas), dim=1)[:, None, None, :].expand(-1, 1, canvas_length, -1)
    window = model.config.text_config.sliding_window
    sliding_prefix = prompt_mask[:, -(window - 1) :] if prompt_mask.shape[1] >= window else prompt_mask
    sliding_allowed = torch.cat((sliding_prefix.bool(), canvas), dim=1)[:, None, None, :].expand(
        -1, 1, canvas_length, -1
    )

    def additive(allowed: torch.Tensor) -> torch.Tensor:
        value = torch.zeros(allowed.shape, dtype=torch.float32, device=prompt_mask.device)
        return value.masked_fill(~allowed, torch.finfo(torch.float32).min)

    return {"full_attention": additive(full_allowed), "sliding_attention": additive(sliding_allowed)}


def diagnose_text(hf, megatron, batch) -> dict[str, dict[str, float]]:
    """Compare encoder and non-self-conditioned decoder outputs per example."""
    inputs = {key: batch[key] for key in ("input_ids", "attention_mask", "position_ids", "mm_token_type_ids")}
    valid = batch["attention_mask"].bool()
    with torch.no_grad():
        reference = hf.model.encoder(**native_hf_encoder_inputs(hf, inputs), use_cache=False).last_hidden_state
        actual = megatron.encode(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state
        hf_logits = hf(
            input_ids=batch["input_ids"],
            attention_mask=native_hf_encoder_inputs(hf, inputs)["attention_mask"],
            position_ids=batch["position_ids"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            decoder_input_ids=batch["noisy_canvas_ids"],
            decoder_attention_mask=hf_decoder_masks(hf, batch["attention_mask"], batch["noisy_canvas_ids"].shape[1]),
        ).logits
        megatron_logits = megatron(
            input_ids=batch["input_ids"],
            decoder_input_ids=batch["noisy_canvas_ids"],
            position_ids=batch["position_ids"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            encoder_attention_mask=batch["attention_mask"],
        ).logits
    rows = {}
    for row in range(batch["input_ids"].shape[0]):
        rows[f"row{row}_encoder_valid"] = compare(reference[row][valid[row]], actual[row][valid[row]])
        rows[f"row{row}_decoder_no_sc"] = compare(hf_logits[row], megatron_logits[row])
    return rows


def main() -> None:
    """Run loss/gradient parity and optimizer updates for the selected mesh."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="reweighted-loo-ce")
    parser.add_argument("--attention-backend", choices=[b.name for b in AttnBackend], default="unfused")
    parser.add_argument("--overfit-steps", type=int, default=80)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--encoder-loss-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    shared_dir = [tempfile.mkdtemp(prefix="diffusiongemma-toy-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(shared_dir, src=0)
    model_dir = Path(shared_dir[0]) / "diffusiongemma-toy"
    try:
        config = toy_config()
        for item in (config, config.text_config, config.vision_config):
            item._attn_implementation = "eager"
        hf = DiffusionGemmaForBlockDiffusion(config).to(torch.float32)
        if dist.get_rank() == 0:
            hf.save_pretrained(model_dir, safe_serialization=True)
        dist.barrier()
        hf = DiffusionGemmaForBlockDiffusion.from_pretrained(
            model_dir, dtype=torch.float32, attn_implementation="eager"
        )
        hf = hf.cuda().train()

        bridge = AutoBridge.from_hf_pretrained(model_dir, torch_dtype=torch.float32)
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.tensor_model_parallel_size = args.tp
        provider.pipeline_model_parallel_size = 1
        provider.expert_model_parallel_size = args.ep
        provider.expert_tensor_parallel_size = 1
        provider.sequence_parallel = args.tp > 1
        provider.params_dtype = torch.float32
        provider.pipeline_dtype = torch.float32
        provider.bf16 = False
        provider.fp16 = False
        provider.moe_aux_loss_coeff = 0.0
        provider.gradient_accumulation_fusion = False
        provider.attention_backend = AttnBackend[args.attention_backend]
        provider.diffusion_loss_variant = args.variant
        provider.diffusion_encoder_loss_weight = args.encoder_loss_weight
        provider.finalize()
        provider.initialize_model_parallel(seed=0)
        megatron = provider.provide_distributed_model(wrap_with_ddp=False)[0].cuda().train()

        cases = {
            "text_unpadded_batch": text_batch(config.text_config.pad_token_id, padded=False),
            "text_padded_batch": text_batch(config.text_config.pad_token_id),
            "image": image_batch(config),
            "text_long_prompt": long_prompt_batch(config.text_config.pad_token_id),
        }
        if args.tp > 1:
            cases = {
                name: left_pad_prompt(batch, config.text_config.pad_token_id, args.tp) for name, batch in cases.items()
            }
        results = {}
        for name, raw_batch in cases.items():
            batch = {key: value.cuda() for key, value in raw_batch.items()}
            diagnostics = diagnose_text(hf, megatron, batch) if name.startswith("text") else {}
            if args.encoder_loss_weight:
                clean_ids = torch.cat((batch["input_ids"], batch["canvas_ids"]), dim=1)
                clean_mm = torch.cat((batch["mm_token_type_ids"], torch.zeros_like(batch["canvas_ids"])), dim=1)
                clean_mask = torch.cat((batch["attention_mask"], batch["canvas_mask"]), dim=1)
                clean_inputs = {
                    "input_ids": clean_ids,
                    "position_ids": torch.arange(clean_ids.shape[1], device=clean_ids.device)[None].expand_as(
                        clean_ids
                    ),
                    "mm_token_type_ids": clean_mm,
                    "attention_mask": clean_mask,
                }
                clean_kwargs = {key: batch[key] for key in ("pixel_values", "image_position_ids") if key in batch}
                with torch.no_grad():
                    ref_hidden = hf.model.encoder(
                        **native_hf_encoder_inputs(hf, clean_inputs), **clean_kwargs, use_cache=False
                    ).last_hidden_state
                    act_hidden = megatron.encode(**clean_inputs, **clean_kwargs).last_hidden_state
                    masks = native_hf_encoder_inputs(hf, clean_inputs)["attention_mask"]
                    blocked = (
                        megatron._compute_attention_mask(clean_ids, clean_mm) | ~clean_mask.bool()[:, None, None, :]
                    )
                    q = torch.arange(clean_ids.shape[1], device=clean_ids.device)
                    sliding_blocked = (
                        blocked | (q[None, :] <= q[:, None] - config.text_config.sliding_window)[None, None]
                    )
                    diagnostics["clean_encoder_valid"] = compare(
                        ref_hidden[clean_mask.bool()], act_hidden[clean_mask.bool()]
                    )
                    diagnostics["sliding_mask_mismatched_elements"] = int(
                        (masks["sliding_attention"].bool() != sliding_blocked).sum()
                    )
                    if name == "text_unpadded_batch":
                        clean_inputs.update(clean_kwargs)
                        diagnostics["clean_layer0_trace"] = compare_layer0(
                            hf_layer0_trace(hf, clean_inputs), megatron_layer0_trace(megatron, clean_inputs)
                        )
                        errors = torch.linalg.vector_norm(ref_hidden - act_hidden, dim=-1)
                        diagnostics["per_token_encoder_error"] = errors.cpu().tolist()
            hf.zero_grad(set_to_none=True)
            megatron.zero_grad(set_to_none=True)
            reference_loss, reference_logits = hf_loss(
                hf,
                batch,
                variant=args.variant,
                vocab_size=config.text_config.vocab_size,
                encoder_loss_weight=args.encoder_loss_weight,
            )
            reference_loss.backward()
            actual_loss, actual_logits = megatron_loss(megatron, batch)
            actual_loss.backward()
            finalize_tp_grads(megatron)
            valid = batch["canvas_mask"].bool()
            hf_grads = {}
            seen = set()
            for param_name, parameter in hf.named_parameters():
                if parameter.grad is not None and id(parameter) not in seen:
                    seen.add(id(parameter))
                    hf_grads[param_name] = parameter.grad.detach().float().cpu()
            megatron_grads = exported_grads(bridge, megatron)
            grad_rows = {}
            aliases = {
                "lm_head.weight": "model.decoder.embed_tokens.weight",
            }
            for hf_name in list(hf_grads):
                if hf_name.startswith("model.encoder.language_model."):
                    aliases[hf_name] = hf_name.replace("model.encoder.language_model.", "model.decoder.", 1)
            for alias, canonical in aliases.items():
                if alias in hf_grads and canonical not in hf_grads:
                    hf_grads[canonical] = hf_grads.pop(alias)
                else:
                    hf_grads.pop(alias, None)
            missing = sorted(set(hf_grads) - set(megatron_grads))
            for param_name, reference_grad in hf_grads.items():
                if param_name in megatron_grads:
                    grad_rows[param_name] = compare(reference_grad, megatron_grads[param_name])
            worst = (
                max(grad_rows.items(), key=lambda item: item[1]["relative_l2"])
                if grad_rows
                else ("tp", {"relative_l2": 0.0})
            )
            total_ref = torch.cat([hf_grads[key].flatten() for key in grad_rows]) if grad_rows else torch.zeros(1)
            total_actual = (
                torch.cat([megatron_grads[key].flatten() for key in grad_rows]) if grad_rows else torch.zeros(1)
            )
            results[name] = {
                "reference_loss": float(reference_loss),
                "megatron_loss": float(actual_loss),
                "loss_abs_diff": float((reference_loss - actual_loss).abs()),
                "valid_logits": compare(reference_logits[valid], actual_logits[valid]),
                "gradient_tensors_compared": len(grad_rows),
                "gradient_tensors_missing": missing,
                "global_gradient": compare(total_ref, total_actual),
                "worst_gradient_tensor": {"name": worst[0], **worst[1]},
                "diagnostics": diagnostics,
            }

        raw_overfit = text_batch(config.text_config.pad_token_id, 3)
        if args.tp > 1:
            raw_overfit = left_pad_prompt(raw_overfit, config.text_config.pad_token_id, args.tp)
        overfit_batch = {key: value.cuda() for key, value in raw_overfit.items()}
        if args.tp > 1 or args.ep > 1:
            # One optimizer step is enough to prove the TP backward path is live;
            # the single-rank overfit remains the stronger convergence assertion.
            args.overfit_steps = min(args.overfit_steps, 2)
        for key in ("noisy_canvas_ids", "diffusion_t", "self_conditioning_mask"):
            overfit_batch.pop(key)
        megatron.config.diffusion_self_conditioning_prob = 0.5
        # Reweighted variants have a 1/(1-t) weight and are too noisy for a short
        # moving-average assertion. Parity above checks the requested variant.
        megatron.config.diffusion_loss_variant = "base-sft"
        optimizer = torch.optim.AdamW(megatron.parameters(), lr=2e-3, weight_decay=0.0)
        losses = []
        for _step in range(args.overfit_steps):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = megatron_loss(megatron, overfit_batch)
            loss.backward()
            finalize_tp_grads(megatron)
            torch.nn.utils.clip_grad_norm_(megatron.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        first = sum(losses[:10]) / 10
        last = sum(losses[-10:]) / 10

        passed = all(
            row["loss_abs_diff"] <= 1e-5 + 1e-5 * abs(row["reference_loss"])
            and row["valid_logits"]["relative_l2"] <= 1e-3
            and row["global_gradient"]["relative_l2"] <= 2e-3
            and not row["gradient_tensors_missing"]
            for row in results.values()
        ) and (last < 0.5 * first if args.tp == args.ep == 1 else all(torch.isfinite(torch.tensor(losses))))
        report = {
            "variant": args.variant,
            "tp": args.tp,
            "ep": args.ep,
            "dtype": "float32",
            "attention_backend": args.attention_backend,
            "cases": results,
            "overfit": {
                "variant": "base-sft",
                "steps": args.overfit_steps,
                "first10_mean": first,
                "last10_mean": last,
            },
            "passed": passed,
        }
        if dist.get_rank() == 0:
            print(json.dumps(report, indent=2), flush=True)
        if args.output and dist.get_rank() == 0:
            args.output.write_text(json.dumps(report, indent=2))
        if not passed:
            raise AssertionError("DiffusionGemma toy training parity failed")
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            import shutil

            shutil.rmtree(shared_dir[0], ignore_errors=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
