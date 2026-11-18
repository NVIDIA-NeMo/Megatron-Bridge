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

"""ERNIE 5.5 per-row MTP loss: every MTP depth normalized by the row's supervised tokens over the CP group.

With per-row loss (``ernie_per_row_loss``) the reference implementation normalizes each MTP depth like the LM
loss (:func:`~megatron.bridge.models.ernie.ernie55_context_parallel.ernie55_per_row_lm_loss`): the row is the
micro-batch, ``S`` is the depth's summed token loss on this context-parallel rank and ``N`` the depth's supervised
tokens summed over the CP group (clamped at 1), after the per-document label and mask roll. The depth injects
``mtp_loss_scaling_factor / mtp_num_layers * CP / N`` times its token losses into the backward
(Megatron-Core's ``MTPLossAutoScaler``, whose scale the schedule sets to ``1 / num_microbatches``), and DDP
averages the gradients over DP x CP, so the gradient is that of the mean over micro-batches of each depth's token
mean over all of its CP ranks' tokens. The tracker records ``CP * S / N`` per micro-batch and averages it over
DP x CP, which logs the mean row MTP loss.

Megatron-Core's ``process_mtp_loss`` divides by this rank's own tokens instead (``S / n_local``): the same value at
CP = 1, but wrong at CP > 1, where a row's supervised tokens are spread unevenly over the ranks. Megatron-Core has
no hook for the normalization, so :class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel`
runs :func:`ernie55_process_mtp_loss` in its place for the duration of its forward
(:func:`per_row_mtp_loss_in`). It reuses Megatron-Core's pieces: the document-aware (and CP-aware) label and mask
roll with the micro-batch's prefetched roll context, the output layer (the multimax head), the language-model
loss, the MTP loss tracker, the acceptance counts and ``MTPLossAutoScaler``.

Only the plain cross-entropy MTP objective is supported: labels given (SFT), no MTP input mask, no fused linear
cross-entropy (multimax needs the logits), no logit scaling and no early MTP backward. It needs Megatron-Core dev's
grouped ``roll_tensor`` (:data:`PER_ROW_MTP_LOSS_SUPPORTED`); on other builds the model keeps Megatron-Core's own
loss, which is the same at CP = 1, and the ERNIE 5.5 config validation rejects per-row MTP at CP > 1
(:func:`~megatron.bridge.models.ernie.modeling_ernie55.context_parallel.validate_ernie55_context_parallel_config`).
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable, Iterator
from typing import Any

import torch
from megatron.core import parallel_state
from megatron.core.transformer import multi_token_prediction as mcore_mtp
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper, roll_tensor
from megatron.core.utils import get_pg_size
from torch import Tensor


__all__ = ["PER_ROW_MTP_LOSS_SUPPORTED", "ernie55_process_mtp_loss", "per_row_mtp_loss_in"]

PER_ROW_MTP_LOSS_SUPPORTED = (
    "roll_context" in inspect.signature(roll_tensor).parameters
    and "num_tokens" in inspect.signature(MTPLossLoggingHelper.save_loss_to_tracker).parameters
)
"""Whether this Megatron-Core build has the grouped ``roll_tensor`` and token-aware MTP tracker (dev) the per-row
MTP loss calls."""


def ernie55_process_mtp_loss(
    hidden_states: Tensor,
    labels: Tensor | None,
    loss_mask: Tensor | None,
    output_layer: Callable,
    output_weight: Tensor | None,
    is_training: bool,
    compute_language_model_loss: Callable,
    config: Any,
    cp_group: torch.distributed.ProcessGroup | None = None,
    tp_group: torch.distributed.ProcessGroup | None = None,
    packed_seq_params: Any = None,
    scale_logits_fn: Callable[[Tensor], Tensor] | None = None,
    input_ids: Tensor | None = None,
    sequence_roll_context: Any = None,
    mtp_input_mask: Tensor | None = None,
    metric_avg_group: torch.distributed.ProcessGroup | None = None,
    main_hidden_states: Tensor | None = None,
    dp_cp_group: torch.distributed.ProcessGroup | None = None,
    runtime_gather_output: bool | None = None,
) -> Tensor:
    """Per-row MTP loss of one micro-batch; Megatron-Core ``process_mtp_loss``'s arguments and return value.

    Args:
        hidden_states: The decoder output and every MTP depth's output, concatenated on the sequence axis.
        labels: ``[b, s]`` targets (already aligned with the inputs).
        loss_mask: ``[b, s]`` 1 for supervised tokens; ``None`` supervises every token.
        output_layer: The model's output layer.
        output_weight: Shared output weight, or ``None`` (the output layer's own).
        is_training: Whether to record the loss and acceptance metrics.
        compute_language_model_loss: Per-token cross entropy ``(labels, logits) -> [b, s]``.
        config: Transformer config (``mtp_num_layers``, ``mtp_loss_scaling_factor``, ``attention_cp_layout``).
        cp_group: Context-parallel group of the MTP layout.
        tp_group: Tensor-parallel group (acceptance counts).
        packed_seq_params: Packed metadata of the MTP layout (document-aware rolls).
        scale_logits_fn: Must be ``None``.
        input_ids: Unused (labels are given).
        sequence_roll_context: The micro-batch's prefetched roll context.
        mtp_input_mask: Must be ``None``.
        metric_avg_group: Group the logged loss is averaged over (DP x CP).
        main_hidden_states: The decoder output to return (``None``: the first chunk of ``hidden_states``).
        dp_cp_group: Fallback for ``metric_avg_group``.
        runtime_gather_output: Unused; MTP logits stay vocab-sharded.

    Returns:
        The decoder output, carrying every depth's loss for the backward.

    Raises:
        ValueError: Without labels, or with an MTP input mask, logit scaling, fused linear cross-entropy, early
            MTP backward or an MTP loss type other than cross entropy.
    """
    del input_ids, runtime_gather_output
    unsupported = [
        name
        for name, active in (
            ("labels=None (labels derived from input_ids)", labels is None),
            ("mtp_input_mask", mtp_input_mask is not None),
            ("logit scaling (muP)", scale_logits_fn is not None),
            (
                "mtp_loss_type other than 'cross_entropy'",
                getattr(config, "mtp_loss_type", "cross_entropy") != "cross_entropy",
            ),
            (
                "fused linear cross entropy",
                getattr(config, "cross_entropy_loss_fusion", False)
                and getattr(config, "cross_entropy_fusion_impl", None) == "linear",
            ),
            ("mtp_loss_early_backward", getattr(config, "mtp_loss_early_backward", False)),
        )
        if active
    ]
    if unsupported:
        raise ValueError(f"ERNIE 5.5 per-row MTP loss does not support {', '.join(unsupported)}.")

    num_depths = config.mtp_num_layers
    chunks = torch.chunk(hidden_states, 1 + num_depths, dim=0)
    output = chunks[0] if main_hidden_states is None else main_hidden_states
    if loss_mask is None:
        loss_mask = torch.ones_like(labels)
    if config.mtp_detach_heads:
        output_weight = (output_weight if output_weight is not None else output_layer.weight).detach()
    if metric_avg_group is None:
        metric_avg_group = dp_cp_group
    if is_training and metric_avg_group is None:
        metric_avg_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

    cp_size = get_pg_size(cp_group)
    depth_scale = config.mtp_loss_scaling_factor / num_depths
    mtp_labels = labels
    for depth in range(num_depths):
        mtp_labels, loss_mask = roll_tensor(
            [mtp_labels, loss_mask],
            shifts=-1,
            dims=-1,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            cp_layout=config.attention_cp_layout,
            roll_context=sequence_roll_context,
            sequence_fields=["labels", "loss_mask"],
            roll_depth=depth,
        )[:2]
        logits, _ = output_layer(chunks[depth + 1], weight=output_weight, runtime_gather_output=False)
        token_losses = compute_language_model_loss(mtp_labels, logits) * loss_mask
        row_tokens = loss_mask.sum().detach().float()
        if cp_size > 1:
            torch.distributed.all_reduce(row_tokens, group=cp_group)
        row_scale = cp_size / row_tokens.clamp(min=1.0)
        if is_training:
            correct = total = None
            acceptance_counts = getattr(mcore_mtp, "_compute_mtp_acceptance_counts", None)
            if acceptance_counts is not None and MTPLossLoggingHelper.should_collect_acceptance():
                correct, total = acceptance_counts(logits, mtp_labels, loss_mask, tp_group=tp_group)
            # One token per micro-batch: the tracker keeps CP * S / N as is and averages it over DP x CP.
            MTPLossLoggingHelper.save_loss_to_tracker(
                token_losses.detach().sum() * row_scale,
                torch.ones((), device=row_scale.device),
                depth,
                num_depths,
                correct=correct,
                total=total,
                avg_group=metric_avg_group,
                calculate_per_token_loss=False,
            )
        output = MTPLossAutoScaler.apply(output, depth_scale * token_losses * row_scale)
    return output


@contextlib.contextmanager
def per_row_mtp_loss_in(module: Any) -> Iterator[None]:
    """Make ``module.process_mtp_loss`` the ERNIE 5.5 per-row MTP loss inside the ``with`` block.

    ``module`` is the Megatron-Core module whose forward calls ``process_mtp_loss`` by its global name
    (``megatron.core.models.hybrid.hybrid_model``). The previous binding is restored on exit, also on error.

    Args:
        module: The module to patch.
    """
    previous = module.process_mtp_loss
    module.process_mtp_loss = ernie55_process_mtp_loss
    try:
        yield
    finally:
        module.process_mtp_loss = previous
