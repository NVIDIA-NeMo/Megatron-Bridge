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

"""CTCStep: forward step for CTC-based diffusion language model training."""

import logging
from functools import partial
from typing import Iterable, Tuple

import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config, unwrap_model

from megatron.bridge.diffusion.models.common.dgpt_step import get_batch
from megatron.bridge.training.losses import _DEFAULT_SPIKY_LOSS_FACTOR as SPIKY_LOSS_FACTOR
from megatron.bridge.training.state import GlobalState


logger = logging.getLogger(__name__)


class CTCStep:
    """Forward training step for CTC-based dLLM.

    Layout: [xt | x0] where xt = blank tokens, x0 = clean tokens.
    xt blocks (size=block_size, e.g. 128) predict target blocks (size=ctc_target_block_size, e.g. 64)
    from x0 using CTC loss. The AR loss on x0 provides an auxiliary causal LM signal.
    """

    def __init__(self, seed: int = 1234):
        self.seed = seed
        self._current_microbatch = 0
        self._first_call = True

    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model,
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

        timers("batch-generator", log_level=2).start()
        with straggler_timer(bdata=True):
            tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, _, _ = (
                get_batch(data_iterator, state.cfg, use_mtp)
            )

        labels_causal = labels.clone()
        timers("batch-generator").stop()

        blank_token_id = self.config.mask_token_id  # token 100

        # Build input: [blank_tokens | clean_tokens]
        input_ids_len = tokens.shape[1]
        blank_input = torch.full_like(tokens, blank_token_id)
        ctc_input = torch.cat([blank_input, tokens], dim=1)

        if cu_seqlens is not None:
            raise ValueError("Packed sequence support is not implemented for CTCStep")

        check_for_nan = state.cfg.rerun_state_machine.check_for_nan_in_loss
        check_for_spiky = state.cfg.rerun_state_machine.check_for_spiky_loss

        with straggler_timer:
            output = model(input_ids=ctc_input, position_ids=position_ids, attention_mask=attention_mask)
            logits = output[0] if isinstance(output, tuple) else output

        # Split: first half = CTC logits, second half = AR logits
        ctc_logits = logits[:, :input_ids_len]
        causal_logits = logits[:, input_ids_len:]

        # CTC loss per block
        xt_block_size = self.config.block_size  # 128
        target_block_size = getattr(self.config, "ctc_target_block_size", xt_block_size // 2)  # 64
        b, seq_len, vocab_size = ctc_logits.shape
        n_blocks = seq_len // xt_block_size

        # Reshape CTC logits into blocks: [b*n_blocks, xt_block_size, vocab]
        ctc_logits_blocks = ctc_logits[:, :n_blocks * xt_block_size].reshape(
            b * n_blocks, xt_block_size, vocab_size
        )
        # CTC expects [T, N, C]
        ctc_log_probs = ctc_logits_blocks.transpose(0, 1).log_softmax(dim=2)

        # Extract targets: block i target = tokens[i*target_block_size : (i+1)*target_block_size]
        n_target_tokens = n_blocks * target_block_size
        targets = tokens[:, :n_target_tokens].reshape(b, n_blocks, target_block_size)
        targets = targets.reshape(b * n_blocks, target_block_size)

        input_lengths = torch.full((b * n_blocks,), xt_block_size, dtype=torch.long, device=tokens.device)
        target_lengths = torch.full((b * n_blocks,), target_block_size, dtype=torch.long, device=tokens.device)

        ctc_loss = torch.nn.functional.ctc_loss(
            ctc_log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank_token_id,
            reduction="sum",
            zero_infinity=True,
        )

        # AR loss on clean context
        core_model = unwrap_model(model)
        if hasattr(core_model, "language_model"):
            core_model = core_model.language_model

        ar_output_tensor = core_model.compute_language_model_loss(
            labels_causal, causal_logits.transpose(0, 1).contiguous()
        )

        ar_loss_weight = getattr(self.config, "ar_loss_weight", 1.0)
        ctc_loss_weight = getattr(self.config, "dlm_loss_weight", 1.0)
        num_tokens_ar = labels_causal.numel()

        output_tensor = (ctc_loss, ar_output_tensor, num_tokens_ar, b * n_blocks)
        loss_function = partial(
            _ctc_ar_combined_loss,
            check_for_nan_in_loss=check_for_nan,
            check_for_spiky_loss=check_for_spiky,
            ctc_loss_weight=ctc_loss_weight,
            ar_loss_weight=ar_loss_weight,
        )

        self._current_microbatch += 1
        if self._current_microbatch >= get_num_microbatches():
            self._current_microbatch = 0
        return output_tensor, loss_function


def _ctc_ar_combined_loss(
    output_tensor: Tuple[torch.Tensor, ...],
    check_for_nan_in_loss: bool = True,
    check_for_spiky_loss: bool = False,
    ctc_loss_weight: float = 1.0,
    ar_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Combined CTC + AR loss."""
    ctc_loss, ar_losses, num_tokens_ar, num_ctc_blocks = output_tensor
    ar_loss = torch.sum(ar_losses)

    rerun_state_machine = get_rerun_state_machine()
    if check_for_nan_in_loss:
        for loss_val, name in ((ctc_loss, "ctc"), (ar_loss, "ar")):
            rerun_state_machine.validate_result(
                result=loss_val, rejection_func=torch.isnan,
                message=f"found NaN in {name} loss", tolerance=0.0, fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss_val, rejection_func=torch.isinf,
                message=f"found Inf in {name} loss", tolerance=0.0, fatal=True,
            )
    if check_for_spiky_loss:
        for loss_val, name in ((ctc_loss, "ctc loss"), (ar_loss, "ar loss")):
            rerun_state_machine.validate_result(
                result=loss_val,
                rejection_func=partial(
                    rerun_state_machine.is_unexpectedly_large,
                    threshold=SPIKY_LOSS_FACTOR, context=name,
                ),
                message="Spiky loss", tolerance=0.0, fatal=False,
            )

    num_tokens_ar = torch.tensor(num_tokens_ar, device=ctc_loss.device, dtype=torch.int)
    num_ctc_blocks_t = torch.tensor(num_ctc_blocks, device=ctc_loss.device, dtype=torch.int)

    loss = ctc_loss * ctc_loss_weight + ar_loss * ar_loss_weight
    num_tokens = (num_ctc_blocks_t + num_tokens_ar).detach().to(torch.int)

    reporting_ctc = torch.cat([ctc_loss.clone().detach().view(1), num_ctc_blocks_t.view(1).to(torch.float)])
    reporting_ar = torch.cat([ar_loss.clone().detach().view(1), num_tokens_ar.view(1).to(torch.float)])
    reporting_total = torch.cat([loss.clone().detach().view(1), num_tokens.view(1).to(torch.float)])

    return loss, num_tokens, {
        "lm loss": reporting_total,
        "ar loss": reporting_ar,
        "ctc loss": reporting_ctc,
    }
