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

"""DGPTStep: forward step for sbd_block_diff diffusion language model training."""

import logging
from functools import partial
from typing import Iterable, Tuple

import torch
import torch.distributed
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import _DEFAULT_SPIKY_LOSS_FACTOR as SPIKY_LOSS_FACTOR, masked_next_token_loss
from megatron.bridge.training.state import GlobalState
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config, unwrap_model

from megatron.bridge.diffusion.common.dllm import forward_process_simple_masking

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
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Generate a batch."""
    if (not parallel_state.is_pipeline_first_stage()) and (not parallel_state.is_pipeline_last_stage()):
        return None, None, None, None, None, None, None, None

    batch = get_batch_from_iterator(
        data_iterator, use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
    )
    batch = get_batch_on_this_cp_rank(batch)

    return (
        batch["tokens"], batch["labels"], batch["loss_mask"],
        batch["attention_mask"], batch["position_ids"],
        batch.get("cu_seqlens"), batch.get("cu_seqlens_argmin"), batch.get("max_seqlen"),
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
        self, state: GlobalState, data_iterator: Iterable, model: GPTModel,
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
            noise_seed = self.seed + 100 * parallel_state.get_data_parallel_rank(with_context_parallel=True)
            self._noise_generator = torch.Generator(device='cuda')
            self._noise_generator.manual_seed(noise_seed)

        timers("batch-generator", log_level=2).start()
        with straggler_timer(bdata=True):
            tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, cu_seqlens_argmin, max_seqlen = get_batch(
                data_iterator, state.cfg, use_mtp
            )

        # For diffusion LM: labels are the clean tokens themselves (not shifted)
        labels_causal = labels.clone()
        labels = tokens.clone()

        timers("batch-generator").stop()

        (noisy_tokens, labels, loss_mask, attention_mask, position_ids,
         masked_indices, p_mask, input_ids_len) = self._apply_noise(
            tokens, labels, loss_mask, attention_mask, position_ids
        )

        forward_args = {
            "input_ids": noisy_tokens,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

        if cu_seqlens is not None:
            raise ValueError("Packed sequence support is not currently implemented for DGPTStep")

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

        # Split logits: first half = DLM logits over xt, second half = AR logits over x0
        causal_logits = logits[:, input_ids_len:]
        logits = logits[:, :input_ids_len]

        core_model = unwrap_model(model)
        if hasattr(core_model, "language_model"):
            core_model = core_model.language_model

        # DLM cross-entropy on masked tokens, scaled by 1/p_mask
        output_tensor = core_model.compute_language_model_loss(
            labels, logits.transpose(0, 1).contiguous()
        )
        output_tensor = output_tensor[masked_indices] / p_mask[masked_indices]
        loss_mask = masked_indices

        # AR cross-entropy on causal logits
        ar_loss_weight = getattr(self.config, 'ar_loss_weight', 1.0)
        dlm_loss_weight = getattr(self.config, 'dlm_loss_weight', 1.0)

        ar_output_tensor = core_model.compute_language_model_loss(
            labels_causal, causal_logits.transpose(0, 1).contiguous()
        )
        num_tokens_ar = labels_causal.numel()

        output_tensor = (output_tensor, ar_output_tensor, num_tokens_ar)
        loss_function = _create_loss_function_sbd(
            loss_mask, check_for_nan_in_loss, check_for_spiky_loss,
            dlm_loss_weight, ar_loss_weight,
            divide_by_masked_tokens=getattr(self.config, 'divide_by_masked_tokens', True),
        )

        self._current_microbatch += 1
        if self._current_microbatch >= get_num_microbatches():
            self._current_microbatch = 0
        return output_tensor, loss_function

    def _apply_noise(self, tokens, labels, loss_mask, attention_mask, position_ids):
        """Apply simple uniform masking and concatenate [noisy | clean] for sbd_block_diff."""
        noisy_inputs, masked_indices, p_mask = forward_process_simple_masking(
            tokens, mask_token_id=self.config.mask_token_id, generator=self._noise_generator,
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
        masked_next_token_loss, loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )


def _create_loss_function_sbd(
    loss_mask, check_for_nan_in_loss, check_for_spiky_loss,
    dlm_loss_weight=1.0, ar_loss_weight=1.0, divide_by_masked_tokens=True,
):
    return partial(
        _masked_loss_sbd_block_diff, loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
        dlm_loss_weight=dlm_loss_weight,
        ar_loss_weight=ar_loss_weight,
        divide_by_masked_tokens=divide_by_masked_tokens,
    )


def _masked_loss_sbd_block_diff(
    loss_mask: torch.Tensor,
    output_tensor: Tuple[torch.Tensor, ...],
    check_for_nan_in_loss: bool = True,
    check_for_spiky_loss: bool = False,
    dlm_loss_weight: float = 1.0,
    ar_loss_weight: float = 1.0,
    divide_by_masked_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Combined DLM + AR loss for sbd_block_diff training."""
    dlm_losses, ar_losses, num_tokens_ar = output_tensor
    dlm_loss = torch.sum(dlm_losses)
    ar_loss = torch.sum(ar_losses)

    rerun_state_machine = get_rerun_state_machine()
    if check_for_nan_in_loss:
        rerun_state_machine.validate_result(
            result=dlm_loss, rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0, fatal=True,
        )
        rerun_state_machine.validate_result(
            result=dlm_loss, rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0, fatal=True,
        )
    if check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=dlm_loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR, context="dlm loss",
            ),
            message="Spiky loss", tolerance=0.0, fatal=False,
        )

    num_tokens_dlm = loss_mask.sum().clone().detach().to(torch.int)
    num_tokens_ar = torch.tensor(num_tokens_ar, device=loss_mask.device, dtype=torch.int)

    loss = dlm_loss * dlm_loss_weight + ar_loss * ar_loss_weight
    num_tokens = num_tokens_dlm + num_tokens_ar
    num_tokens = num_tokens.detach().to(torch.int)

    reporting_loss_dlm = torch.cat([dlm_loss.clone().detach().view(1), num_tokens_dlm.view(1)])
    reporting_loss_ar = torch.cat([ar_loss.clone().detach().view(1), num_tokens_ar.view(1)])
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    report_dict = {
        "lm loss": reporting_loss,
        "ar loss": reporting_loss_ar,
        "dlm loss": reporting_loss_dlm,
        "num_tokens_dlm": torch.cat([num_tokens_dlm.view(1).to(torch.float)]),
    }

    return loss, num_tokens, report_dict
