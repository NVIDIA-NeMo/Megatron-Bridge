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

"""DGPTStep: forward step for sbd_block_diff diffusion language model training."""

import logging
from functools import partial
from typing import Iterable, Tuple

import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_model_config, unwrap_model

from megatron.bridge.diffusion.common.cp_utils import all_gather_seq_cp, local_zigzag_mask, zigzag_slice
from megatron.bridge.diffusion.common.dllm import forward_process_simple_masking
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import _DEFAULT_SPIKY_LOSS_FACTOR as SPIKY_LOSS_FACTOR
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def get_batch_from_iterator(
    data_iterator: Iterable,
    use_mtp: bool = False,
    skip_getting_attention_mask_from_dataset: bool = True,
) -> dict[str, torch.Tensor]:
    """Get a batch of data from the iterator."""
    batch = next(data_iterator)

    required_device_keys = set()
    required_host_keys = set()

    if not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    if parallel_state.is_pipeline_first_stage() or use_mtp:
        required_device_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu() if val is not None else None
        else:
            _batch_required_keys[key] = None

    return _batch_required_keys


def get_batch(
    data_iterator: Iterable, cfg: ConfigContainer, use_mtp: bool = False
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Generate a batch."""
    if (not parallel_state.is_pipeline_first_stage()) and (not parallel_state.is_pipeline_last_stage()):
        return None, None, None, None, None, None, None, None

    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
    )
    # NOTE: do NOT shard across CP here. The sbd_block_diff sequence is doubled to
    # [xt | x0] in DGPTStep._apply_noise; CP sharding must happen on the full 2L
    # sequence (see DGPTStep.__call__), not on the pre-doubled L.

    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
        batch.get("cu_seqlens"),
        batch.get("cu_seqlens_argmin"),
        batch.get("max_seqlen"),
    )


# ---------------------------------------------------------------------------
# DGPTStep (sbd_block_diff, no KD, simple masking only)
# ---------------------------------------------------------------------------


class DGPTStep:
    """Forward training step for sbd_block_diff diffusion LM."""

    def __init__(self, seed: int = 1234):
        self.seed = seed
        self._noise_generator = None
        self._current_microbatch = 0
        self._first_call = True

    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model: GPTModel,
        return_schedule_plan: bool = False,
    ) -> tuple[torch.Tensor, partial]:
        if self._first_call:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            self._first_call = False

        timers = state.timers
        straggler_timer = state.straggler_timer

        config = get_model_config(model)
        use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0
        self.config = config

        # Per-DP-rank noise generator
        if self.config.different_seed_per_dp and self._noise_generator is None:
            # Seed per DP replica WITHOUT context parallel: all CP ranks of a replica
            # share the same global batch and must apply the SAME diffusion masking,
            # otherwise the CP all-gather in attention reconstructs an inconsistently
            # masked sequence. (with_context_parallel=True would desync masks per CP rank.)
            noise_seed = self.seed + 100 * parallel_state.get_data_parallel_rank(with_context_parallel=False)
            self._noise_generator = torch.Generator(device="cuda")
            self._noise_generator.manual_seed(noise_seed)

        timers("batch-generator", log_level=2).start()
        with straggler_timer(bdata=True):
            tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, cu_seqlens_argmin, max_seqlen = (
                get_batch(data_iterator, state.cfg, use_mtp)
            )

        # For diffusion LM: labels are the clean tokens themselves (not shifted)
        labels_causal = labels.clone()
        labels = tokens.clone()

        timers("batch-generator").stop()

        # Build the doubled [xt | x0] sequence on the FULL (un-sharded) batch.
        (noisy_tokens, labels, loss_mask, attention_mask, position_ids, masked_indices, p_mask, input_ids_len) = (
            self._apply_noise(tokens, labels, loss_mask, attention_mask, position_ids)
        )
        if cu_seqlens is not None:
            raise ValueError("Packed sequence support is not currently implemented for DGPTStep")

        # Context-parallel sharding of the FULL 2L sequence (load-balanced zigzag,
        # matching what TEDotProductAttention assumes). cp_size == 1 -> no-op.
        cp_size = self.config.context_parallel_size
        cp_group = parallel_state.get_context_parallel_group() if cp_size > 1 else None
        cp_rank = parallel_state.get_context_parallel_rank() if cp_size > 1 else 0
        if cp_size > 1:
            input_ids_local = zigzag_slice(noisy_tokens, cp_rank, cp_size, seq_dim=1)
        else:
            input_ids_local = noisy_tokens

        # position_ids are shape-only here: this model uses position_embedding_type="none",
        # and the attention recomputes RoPE from an internal arange (per half of the doubled
        # sequence, after the CP all-gather), so the *values* passed in are never consumed.
        # We only need a tensor whose shape matches input_ids_local.
        position_ids_local = torch.zeros_like(input_ids_local)

        forward_args = {
            "input_ids": input_ids_local,
            "position_ids": position_ids_local,
            "attention_mask": attention_mask,
        }

        check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
        check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss

        with straggler_timer:
            if return_schedule_plan:
                assert config.overlap_moe_expert_parallel_comm
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
                return schedule_plan, loss_function
            else:
                output = model(**forward_args)
                logits = output[0] if isinstance(output, tuple) else output

        # Gather the CP-sharded logits [b, 2L/cp, v] back to the full doubled
        # sequence [b, 2L, v] so the [xt | x0] split is well-defined (identical on
        # all CP ranks). Loss is then restricted to this rank's owned positions.
        if cp_size > 1:
            logits = all_gather_seq_cp(logits, cp_group, seq_dim=1)

        # Split logits: first half = DLM logits over xt, second half = AR logits over x0.
        # The two halves are views over shared storage, and vocab-parallel cross-entropy
        # (used by both DLM and AR losses under calculate_per_token_loss) mutates its
        # input in place -> clone the slices so each loss has independent storage. Clone
        # is differentiable; grad flows back through the slice (and the CP all-gather).
        causal_logits = logits[:, input_ids_len:].clone()
        logits = logits[:, :input_ids_len].clone()

        core_model = unwrap_model(model)
        if hasattr(core_model, "language_model"):
            core_model = core_model.language_model

        # Restrict the loss to this CP rank's owned (zigzag) positions so that
        # summing loss/num_tokens across the CP group recovers the global totals
        # (keeps standard Megatron CP loss reduction valid). No-op at cp_size == 1.
        x0_owned = None
        if cp_size > 1:
            owned_2l = local_zigzag_mask(2 * input_ids_len, cp_rank, cp_size, device=logits.device)
            xt_owned = owned_2l[:input_ids_len]  # [L]
            x0_owned = owned_2l[input_ids_len:]  # [L]
            masked_indices = masked_indices & xt_owned.unsqueeze(0)

        # DLM cross-entropy on masked tokens, scaled by 1/p_mask
        output_tensor = core_model.compute_language_model_loss(labels, logits.transpose(0, 1).contiguous())
        output_tensor = output_tensor[masked_indices] / p_mask[masked_indices]
        loss_mask = masked_indices

        # AR cross-entropy on causal logits
        ar_loss_weight = getattr(self.config, "ar_loss_weight", 1.0)
        dlm_loss_weight = getattr(self.config, "dlm_loss_weight", 1.0)

        ar_output_tensor = core_model.compute_language_model_loss(
            labels_causal, causal_logits.transpose(0, 1).contiguous()
        )
        if x0_owned is not None:
            ar_output_tensor = ar_output_tensor[x0_owned.unsqueeze(0).expand_as(ar_output_tensor)]
            num_tokens_ar = labels_causal.shape[0] * int(x0_owned.sum().item())
        else:
            num_tokens_ar = labels_causal.numel()

        output_tensor = (output_tensor, ar_output_tensor, num_tokens_ar)
        loss_function = _create_loss_function_sbd(
            loss_mask,
            check_for_nan_in_loss,
            check_for_spiky_loss,
            dlm_loss_weight,
            ar_loss_weight,
        )

        self._current_microbatch += 1
        if self._current_microbatch >= get_num_microbatches():
            self._current_microbatch = 0
        return output_tensor, loss_function

    def _apply_noise(self, tokens, labels, loss_mask, attention_mask, position_ids):
        """Apply simple uniform masking and concatenate [noisy | clean] for sbd_block_diff."""
        noisy_inputs, masked_indices, p_mask = forward_process_simple_masking(
            tokens,
            mask_token_id=self.config.mask_token_id,
            generator=self._noise_generator,
            loss_mask=loss_mask,
        )
        input_ids_len = noisy_inputs.shape[1]
        # sbd_block_diff doubles the sequence: [xt | x0]
        noisy_inputs = torch.cat([noisy_inputs, tokens], dim=1)
        return noisy_inputs, labels, loss_mask, attention_mask, position_ids, masked_indices, p_mask, input_ids_len


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss):
    return partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )


def _create_loss_function_sbd(
    loss_mask,
    check_for_nan_in_loss,
    check_for_spiky_loss,
    dlm_loss_weight=1.0,
    ar_loss_weight=1.0,
):
    return partial(
        _masked_loss_sbd_block_diff,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
        dlm_loss_weight=dlm_loss_weight,
        ar_loss_weight=ar_loss_weight,
    )


def _masked_loss_sbd_block_diff(
    loss_mask: torch.Tensor,
    output_tensor: Tuple[torch.Tensor, ...],
    check_for_nan_in_loss: bool = True,
    check_for_spiky_loss: bool = False,
    dlm_loss_weight: float = 1.0,
    ar_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Combined DLM + AR loss for sbd_block_diff training."""
    dlm_losses, ar_losses, num_tokens_ar = output_tensor
    dlm_loss = torch.sum(dlm_losses)
    ar_loss = torch.sum(ar_losses)

    rerun_state_machine = get_rerun_state_machine()
    if check_for_nan_in_loss:
        for loss_val, name in ((dlm_loss, "dlm"), (ar_loss, "ar")):
            rerun_state_machine.validate_result(
                result=loss_val,
                rejection_func=torch.isnan,
                message=f"found NaN in {name} loss",
                tolerance=0.0,
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss_val,
                rejection_func=torch.isinf,
                message=f"found Inf in {name} loss",
                tolerance=0.0,
                fatal=True,
            )
    if check_for_spiky_loss:
        for loss_val, name in ((dlm_loss, "dlm loss"), (ar_loss, "ar loss")):
            rerun_state_machine.validate_result(
                result=loss_val,
                rejection_func=partial(
                    rerun_state_machine.is_unexpectedly_large,
                    threshold=SPIKY_LOSS_FACTOR,
                    context=name,
                ),
                message="Spiky loss",
                tolerance=0.0,
                fatal=False,
            )

    num_tokens_dlm = loss_mask.sum().clone().detach().to(torch.int)
    num_tokens_ar = torch.tensor(num_tokens_ar, device=loss_mask.device, dtype=torch.int)

    loss = dlm_loss * dlm_loss_weight + ar_loss * ar_loss_weight
    num_tokens = num_tokens_dlm + num_tokens_ar
    num_tokens = num_tokens.detach().to(torch.int)

    reporting_loss_dlm = torch.cat([dlm_loss.clone().detach().view(1), num_tokens_dlm.view(1)])
    reporting_loss_ar = torch.cat([ar_loss.clone().detach().view(1), num_tokens_ar.view(1)])
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    # The training-log reducer interprets each report entry as a [numerator,
    # denominator] pair and shows numerator/denominator (summed over microbatches
    # and all-reduced over the DP-CP group). A 1-element entry degenerates to 0,
    # so report num_tokens_dlm as [dlm_tokens, num_samples] -> the displayed value
    # is the (global) average number of DLM/masked tokens per sample.
    # Divide num_samples by the CP world size: all CP ranks of a replica hold the
    # same batch (num_samples is replicated, not sharded, across CP), so without
    # this the DP-CP all-reduce would count each sample cp_size times.
    cp_world_size = parallel_state.get_context_parallel_world_size()
    num_samples_dlm = torch.tensor([float(loss_mask.shape[0]) / cp_world_size], device=loss_mask.device)
    report_dict = {
        "lm loss": reporting_loss,
        "ar loss": reporting_loss_ar,
        "dlm loss": reporting_loss_dlm,
        "num_tokens_dlm": torch.cat([num_tokens_dlm.view(1).to(torch.float), num_samples_dlm]),
    }

    return loss, num_tokens, report_dict
