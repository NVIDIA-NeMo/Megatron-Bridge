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

"""
FLUX Forward Step with Automodel Pipeline Integration.

This is a prototype showing how to integrate the automodel FlowMatchingPipeline
into Megatron's training flow, reusing the well-tested flow matching logic.
"""

import logging
from functools import partial
from typing import Iterable

import torch
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.utils import get_model_config

# Import automodel pipeline components
from dfm.src.automodel.flow_matching.flow_matching_pipeline import FlowMatchingPipeline

# Import MegatronFluxAdapter from flow_matching module
from megatron.bridge.diffusion.models.flux.flow_matching.flux_adapter import MegatronFluxAdapter


logger = logging.getLogger(__name__)


# =============================================================================
# Megatron Forward Step with Automodel Pipeline
# =============================================================================


def flux_data_step(dataloader_iter, store_in_state=False):
    """Process batch data for FLUX model.

    Args:
        dataloader_iter: Iterator over the dataloader.
        store_in_state: If True, store the batch in GlobalState for callbacks.

    Returns:
        Processed batch dictionary with tensors moved to CUDA.
    """
    batch = next(dataloader_iter)
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    _batch = {k: v.to(device="cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in _batch.items()}

    if "loss_mask" not in _batch or _batch["loss_mask"] is None:
        _batch["loss_mask"] = torch.ones(1, device="cuda")

    # Store batch in state for callbacks (e.g., validation image generation)
    if store_in_state:
        try:
            from megatron.bridge.training.pretrain import get_current_state

            state = get_current_state()
            state._last_validation_batch = _batch
        except:
            pass  # If state access fails, silently continue

    return _batch


class FluxForwardStepWithAutomodel:
    """
    Forward step for FLUX using the automodel FlowMatchingPipeline.

    This class demonstrates how to integrate the well-tested automodel pipeline
    into Megatron's training flow, gaining benefits like:
    - Unified flow matching implementation
    - Better logging and debugging
    - Consistent timestep sampling across models
    - Easier maintenance

    Args:
        timestep_sampling: Method for sampling timesteps ("logit_normal", "uniform", "mode").
        logit_mean: Mean for logit-normal sampling.
        logit_std: Standard deviation for logit-normal sampling.
        flow_shift: Shift parameter for timestep transformation (default: 1.0 for FLUX).
        scheduler_steps: Number of scheduler training steps.
        guidance_scale: Guidance scale for FLUX-dev models.
        use_loss_weighting: Whether to apply flow-based loss weighting.
    """

    def __init__(
        self,
        timestep_sampling: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        flow_shift: float = 1.0,  # FLUX uses shift=1.0 typically
        scheduler_steps: int = 1000,
        guidance_scale: float = 3.5,
        use_loss_weighting: bool = False,  # FLUX typically doesn't use loss weighting
    ):
        self.autocast_dtype = torch.bfloat16

        # Create the automodel pipeline with Megatron adapter
        adapter = MegatronFluxAdapter(guidance_scale=guidance_scale)

        self.pipeline = FlowMatchingPipeline(
            model_adapter=adapter,
            num_train_timesteps=scheduler_steps,
            timestep_sampling=timestep_sampling,
            flow_shift=flow_shift,
            logit_mean=logit_mean,
            logit_std=logit_std,
            sigma_min=0.0,
            sigma_max=1.0,
            use_loss_weighting=use_loss_weighting,
            cfg_dropout_prob=0.0,  # No CFG dropout in Megatron training
            log_interval=100,
            summary_log_interval=10,
        )

        logger.info(
            f"FluxForwardStepWithAutomodel initialized with:\n"
            f"  - Timestep sampling: {timestep_sampling}\n"
            f"  - Flow shift: {flow_shift}\n"
            f"  - Guidance scale: {guidance_scale}\n"
            f"  - Loss weighting: {use_loss_weighting}"
        )

    def __call__(
        self, state: GlobalState, data_iterator: Iterable, model: VisionModule
    ) -> tuple[torch.Tensor, partial]:
        """
        Forward training step using automodel pipeline.

        Args:
            state: Global state for the run.
            data_iterator: Input data iterator.
            model: The FLUX model.

        Returns:
            Tuple containing the output tensor and the loss function.
        """
        timers = state.timers
        straggler_timer = state.straggler_timer

        config = get_model_config(model)

        timers("batch-generator", log_level=2).start()

        with straggler_timer(bdata=True):
            batch = flux_data_step(data_iterator)
            # Store batch for validation callbacks (only during evaluation)
            if not torch.is_grad_enabled():
                state._last_batch = batch
        timers("batch-generator").stop()

        check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
        check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss

        # Prepare batch for automodel pipeline
        # Map Megatron keys to automodel expected keys
        pipeline_batch = self._prepare_batch_for_pipeline(batch)

        # Run the pipeline step
        with straggler_timer:
            if parallel_state.is_pipeline_last_stage():
                output_tensor, loss, loss_mask = self._training_step_with_pipeline(model, pipeline_batch)
                # loss_mask is already created correctly in _training_step_with_pipeline
                batch["loss_mask"] = loss_mask
            else:
                # For non-final pipeline stages, we still need to run the model
                # but loss computation happens only on the last stage
                output_tensor = self._training_step_with_pipeline(model, pipeline_batch)
                loss_mask = None

        # Use the loss_mask from training step (already has correct shape)
        if loss_mask is None:
            # This should only happen for non-final pipeline stages
            loss_mask = torch.ones(1, device="cuda")

        loss_function = self._create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

        return output_tensor, loss_function

    def _prepare_batch_for_pipeline(self, batch: dict) -> dict:
        """
        Prepare Megatron batch for automodel pipeline.

        Maps Megatron batch keys to automodel expected format:
        - latents -> image_latents (for consistency)
        - Keeps prompt_embeds, pooled_prompt_embeds, text_ids as-is
        """
        pipeline_batch = {
            "image_latents": batch["latents"],  # Map to automodel expected key
            "prompt_embeds": batch.get("prompt_embeds"),
            "pooled_prompt_embeds": batch.get("pooled_prompt_embeds"),
            "text_ids": batch.get("text_ids"),
            "data_type": "image",  # FLUX is for image generation
        }

        # Copy any additional keys
        for key in batch:
            if key not in pipeline_batch and key != "latents":
                pipeline_batch[key] = batch[key]

        return pipeline_batch

    def _training_step_with_pipeline(
        self, model: VisionModule, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Perform single training step using automodel pipeline.

        Args:
            model: The FLUX model.
            batch: Data batch prepared for pipeline.

        Returns:
            On last pipeline stage: tuple of (output_tensor, loss, loss_mask).
            On other stages: output tensor.
        """
        device = torch.device("cuda")
        dtype = self.autocast_dtype

        # Pass model in batch so adapter can check for guidance support
        batch["_model"] = model

        with torch.amp.autocast("cuda", enabled=dtype in (torch.half, torch.bfloat16), dtype=dtype):
            # Run the automodel pipeline step (global_step defaults to 0)
            weighted_loss, average_weighted_loss, loss_mask, metrics = self.pipeline.step(
                model=model,
                batch=batch,
                device=device,
                dtype=dtype,
            )

        # Clean up temporary model reference
        batch.pop("_model", None)

        if parallel_state.is_pipeline_last_stage():
            # Match original implementation's reduction pattern
            # Original does: loss = mse(..., reduction="none"), then output_tensor = mean(loss, dim=-1)
            # This keeps most dimensions and only reduces the last one
            # But automodel returns full loss, so we reduce to match expected shape

            # For FLUX with images: weighted_loss is [B, C, H, W]
            # Original pattern: mean over spatial dimensions -> [B, C] or similar
            # But Megatron expects a 1D tensor per sample, so reduce to [B]
            output_tensor = torch.mean(weighted_loss, dim=list(range(1, weighted_loss.ndim)))

            # Always create a fresh loss_mask matching output_tensor shape
            # Ignore any loss_mask from batch as it may have incompatible shape
            loss_mask = torch.ones_like(output_tensor)

            return output_tensor, average_weighted_loss, loss_mask
        else:
            # For intermediate stages, return the tensor for pipeline communication
            return weighted_loss

    def _create_loss_function(
        self, loss_mask: torch.Tensor, check_for_nan_in_loss: bool, check_for_spiky_loss: bool
    ) -> partial:
        """
        Create a partial loss function with the specified configuration.

        Args:
            loss_mask: Used to mask out some portions of the loss.
            check_for_nan_in_loss: Whether to check for NaN values in the loss.
            check_for_spiky_loss: Whether to check for spiky loss values.

        Returns:
            A partial function that can be called with output_tensor to compute the loss.
        """
        return partial(
            masked_next_token_loss,
            loss_mask,
            check_for_nan_in_loss=check_for_nan_in_loss,
            check_for_spiky_loss=check_for_spiky_loss,
        )


# =============================================================================
# Convenience Factory
# =============================================================================


def create_flux_forward_step(
    use_automodel_pipeline: bool = True,
    **kwargs,
):
    """
    Factory function to create either the automodel-based or original forward step.

    Args:
        use_automodel_pipeline: If True, use FluxForwardStepWithAutomodel.
                                If False, use original FluxForwardStep.
        **kwargs: Arguments passed to the forward step constructor.

    Returns:
        Forward step instance.

    Example:
        # Use automodel pipeline
        forward_step = create_flux_forward_step(
            use_automodel_pipeline=True,
            timestep_sampling="logit_normal",
            flow_shift=1.0,
        )

        # Use original implementation
        forward_step = create_flux_forward_step(
            use_automodel_pipeline=False,
            timestep_sampling="logit_normal",
        )
    """
    if use_automodel_pipeline:
        return FluxForwardStepWithAutomodel(**kwargs)
    else:
        from megatron.bridge.diffusion.models.flux.flux_step import FluxForwardStep

        return FluxForwardStep(**kwargs)
