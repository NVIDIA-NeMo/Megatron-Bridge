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

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.peft.base import PEFT  # low-level transform base


def get_peft_model(
    provider: ModelProviderMixin,
    peft: PEFT,
    *,
    training: bool = True,
    wrap_with_ddp: bool = True,
) -> "PEFTModel":
    """Apply a PEFT transform at the correct point in the provider pipeline and return a PEFTModel.

    This function applies Parameter-Efficient Fine-Tuning (PEFT) adaptations to a Megatron model
    by hooking into the provider's pre-wrap stage, ensuring PEFT is applied before DDP wrapping.

    Args:
        provider: A ModelProviderMixin instance that provides the base model. Must be a provider
            to enable PEFT application before DDP wrapping.
        peft: A PEFT instance that defines the specific adaptation technique (LoRA, DoRA, etc.).
        training: Whether to enable gradients for adapter parameters after application.
            Defaults to True.
        wrap_with_ddp: Whether to wrap the model with DistributedDataParallel. Defaults to True.

    Returns:
        PEFTModel: A wrapped model with PEFT adaptations applied.

    Raises:
        TypeError: If provider is not a ModelProviderMixin instance.
        TypeError: If peft is not a PEFT instance.

    Note:
        PEFT must be applied BEFORE DDP wrapping. This function hooks into the provider's
        pre-wrap stage to ensure correct application order.
    """
    if not isinstance(provider, ModelProviderMixin):
        raise TypeError(
            f"provider must be a ModelProviderMixin, got {type(provider)}. "
            "Construct your base model via a provider so PEFT can be applied before DDP."
        )
    if not isinstance(peft, PEFT):
        raise TypeError(f"peft must be a PEFT instance, got {type(peft)}")

    # Hook 1: apply PEFT structure before DDP wrapping.
    def _apply_peft_hook(model_or_stages: Union[List[MegatronModule], MegatronModule]):
        # The low-level PEFT callable performs an in-place graph transform.
        adapted = peft(model_or_stages, training=training)

        # Optionally ensure adapter params require grad for training convenience.
        if training:
            for mod in _iterate_modules(adapted):
                for name, p in mod.named_parameters(recurse=True):
                    if ".adapter." in name:
                        p.requires_grad = True
        return adapted

    provider.register_pre_wrap_hook(_apply_peft_hook)

    # Materialize Megatron model (list of stages or a single module depending on VP config)
    stages = provider.provide_distributed_model(wrap_with_ddp=wrap_with_ddp)
    return PEFTModel(stages, peft)


class PEFTModel(nn.ModuleList):
    """Wrapper for PEFT-adapted Megatron models.

    This class wraps PEFT-adapted Megatron models and maintains compatibility with the
    List[MegatronModule] interface for callers that expect pipeline parallel stages.
    It provides convenience methods for managing adapter parameters and state.

    Attributes:
        peft: The PEFT instance used for adaptation.
        stages: Alias to self, acting as ModuleList of stages.
        adapter_name: Name identifier for the adapter (default: "default").

    Note:
        This class inherits from nn.ModuleList to maintain compatibility with existing
        Megatron pipeline parallel code that expects a list of model stages.
    """

    def __init__(self, stages: Union[List[MegatronModule], MegatronModule], peft: PEFT):
        super().__init__(stages if isinstance(stages, list) else [stages])
        self.peft: PEFT = peft
        self.stages: "PEFTModel" = self  # alias: self acts as ModuleList of stages
        self.adapter_name: str = "default"  # placeholder for future multi-adapter support

    # ------------------------------------------------------------
    # Convenience / UX helpers
    # ------------------------------------------------------------
    @torch.no_grad()
    def print_trainable_parameters(self) -> None:
        """Report global trainable vs total parameter counts across PP/TP.

        This method calculates and prints the number of trainable and frozen parameters
        across all pipeline parallel (PP) and tensor parallel (TP) ranks. The counts
        are aggregated using all-reduce operations in distributed settings.

        The output includes:
            - Number and percentage of trainable parameters
            - Number and percentage of frozen parameters

        Note:
            Output is only printed on the main rank (rank 0) to avoid duplicate messages
            in distributed training.
        """
        local_total = 0
        local_trainable = 0
        for stage in _iterate_modules(self.stages):
            for p in stage.parameters():
                n = p.numel()
                local_total += n
                if p.requires_grad:
                    local_trainable += n

        if dist.is_available() and dist.is_initialized():
            t_total = torch.tensor(
                [local_total], dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu"
            )
            t_train = torch.tensor([local_trainable], dtype=torch.long, device=t_total.device)
            dist.all_reduce(t_total)
            dist.all_reduce(t_train)
            total = int(t_total.item())
            trainable = int(t_train.item())
        else:
            total = local_total
            trainable = local_trainable

        pct = (100.0 * trainable / total) if total > 0 else 0.0
        if _is_main_rank():
            print(f"Trainable: {trainable:,} ({pct:.2f}%), Frozen: {total - trainable:,} ({100.0 - pct:.2f}%)")

    @torch.no_grad()
    def enable_adapters(self) -> None:
        """Enable gradients for adapter parameters only.

        This method sets requires_grad=True for all parameters that contain ".adapter."
        in their name, enabling training of adapter weights while keeping base model
        parameters frozen.

        Note:
            This is useful for switching between full fine-tuning and adapter-only
            training modes during training.
        """
        for stage in _iterate_modules(self.stages):
            for name, p in stage.named_parameters(recurse=True):
                if ".adapter." in name:
                    p.requires_grad = True

    @torch.no_grad()
    def disable_adapters(self) -> None:
        """Disable gradients for adapter parameters only.

        This method sets requires_grad=False for all parameters that contain ".adapter."
        in their name, freezing adapter weights for inference or evaluation.

        Note:
            This is useful for switching to inference mode or when you want to
            freeze adapter parameters during certain training phases.
        """
        for stage in _iterate_modules(self.stages):
            for name, p in stage.named_parameters(recurse=True):
                if ".adapter." in name:
                    p.requires_grad = False

    # ------------------------------------------------------------
    # Adapter-only state I/O (Megatron-native names, no remap)
    # ------------------------------------------------------------
    @torch.no_grad()
    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Extract adapter-only parameters keyed by Megatron-native names.

        This method collects all parameters that contain ".adapter." in their names
        and returns them as a dictionary with cloned tensors. The parameter names
        are kept in their original Megatron format.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping parameter names to cloned
                adapter parameter tensors.

        Note:
            This method does not perform any tensor parallel (TP) or pipeline parallel (PP)
            gathering; it simply snapshots local parameter shards. For distributed
            checkpointing, use adapter_sharded_state_dict instead.
        """
        sd: Dict[str, torch.Tensor] = {}
        for stage in _iterate_modules(self.stages):
            for name, p in stage.named_parameters(recurse=True):
                if ".adapter." in name:
                    sd[name] = p.detach().clone()
        return sd

    def adapter_sharded_state_dict(self) -> Dict[str, Any]:
        """Extract adapter-only parameters with proper sharding metadata for distributed checkpointing.

        This method extracts adapter parameters while preserving their sharding metadata,
        making them compatible with Megatron's distributed checkpointing system. The
        returned dictionary includes both parameter tensors and their associated sharding
        information for proper reconstruction across distributed ranks.

        Returns:
            Dict[str, Any]: A dictionary containing adapter parameters with sharding
                metadata. Keys are prefixed with stage identifiers (e.g., "model0.param_name")
                for multi-stage pipeline parallel models.

        Note:
            This method is preferred over adapter_state_dict() when using Megatron's
            distributed checkpointing functionality, as it preserves the necessary
            sharding metadata for correct parameter reconstruction.
        """
        adapter_state = {}

        for stage_idx, stage in enumerate(self.stages):
            stage_prefix = f"model{stage_idx}" if len(self.stages) > 1 else "model"

            # Get adapter parameters with sharding information
            stage_sharded_state = stage.sharded_state_dict()
            for param_name, param_value in stage_sharded_state.items():
                if ".adapter." in param_name:
                    adapter_state[f"{stage_prefix}.{param_name}"] = param_value

        return adapter_state

    # ------------------------------------------------------------
    # Merge (intentionally conservative in MVP)
    # ------------------------------------------------------------
    @torch.no_grad()
    def merge_and_unload(self) -> List[MegatronModule]:
        """Merge adapters into base weights in-place and return list of unwrapped Megatron stages.

        This method merges the adapter parameters into the base model weights, creating
        a single model without separate adapter components. The merging is performed
        in-place and delegates to the PEFT instance's merge method, which implements
        the specific merging logic for each PEFT technique (LoRA, DoRA, etc.).

        Returns:
            List[MegatronModule]: A list of model stages with adapters merged into base weights.
                The returned stages no longer contain separate adapter parameters.

        Raises:
            NotImplementedError: If the PEFT implementation doesn't support merging.

        Note:
            After calling this method, the model will behave as if it was trained
            with the merged weights from scratch. This is useful for inference
            deployment where you want a single model without adapter overhead.
        """
        # Delegate to the PEFT instance's merge method
        merged_model = self.peft.merge(list(self.stages))

        # Ensure we return a list
        if isinstance(merged_model, list):
            return merged_model
        else:
            return [merged_model]


def _is_main_rank() -> bool:
    """Check if the current process is the main rank (rank 0).

    Returns:
        bool: True if this is the main rank or if distributed training is not initialized,
            False otherwise.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _barrier_if_needed() -> None:
    """Synchronize all processes if distributed training is initialized.

    This function calls a distributed barrier to synchronize all processes,
    but only if distributed training is available and initialized.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _iterate_modules(stages: Union[List[MegatronModule], MegatronModule]) -> Iterable[MegatronModule]:
    """Iterate over model stages, handling both single modules and lists of modules.

    Args:
        stages: Either a single MegatronModule or a list of MegatronModules representing
            pipeline parallel stages.

    Yields:
        MegatronModule: Individual model stages from the input.
    """
    if isinstance(stages, list):
        yield from stages
    else:
        yield stages
