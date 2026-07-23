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

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.utils import unwrap_model

from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora_layers import (
    LinearAdapter,
    LoRALinear,
    LoRATopKRouter,
    TEFusedLoRALinear,
)
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.utils import (
    GroupedExpertLinearAdapter,
    ParallelLinearAdapter,
    SharedOuterGroupedExpertAdapter,
    align_expert_dim_for_tp,
    get_adapter_attributes_from_linear,
    get_effective_lora_dim,
    is_expert_linear,
    is_grouped_expert_linear,
    is_modelopt_linear,
)


logger = logging.getLogger(__name__)


@dataclass
class LoRA(PEFT, ModuleMatcher):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
        exclude_modules (List[str], optional): A list of module names not to apply LoRa to. It will
            match all nn.Linear & nn.Linear-adjacent modules whose name does not match any string in
            exclude_modules. If used, will require target_modules to be empty list or None.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.
        sequence_parallel_input_regather (bool): Reduce retained activation memory for eligible
            column-parallel LoRA-A projections. The full LayerNorm output consumed by LoRA-A is
            gathered temporarily in forward, released after the LoRA-A computation, and gathered
            again during backward for the LoRA-A weight gradient. MCore overlaps the backward
            all-gather with dgrad computation when possible. This has no effect when sequence
            parallelism is disabled and falls back for unsupported adapters or overlapping
            activation recompute. Defaults to False.
        a2a_experimental (bool): Enables the experimental All-to-All (A2A) communication strategy. Defaults to False.
        lora_A_init_method (str): Initialization method for the low-rank matrix A. Defaults to "xavier".
        lora_B_init_method (str): Initialization method for the low-rank matrix B. Defaults to "zero".
        lora_dtype (torch.dtype): Parameter data type for LoRA weights. Default None (will use model's dtype).
        normalize_moe_lora (bool): When True, expert linear layers use dim // moe_router_topk as the LoRA rank
            while non-expert layers keep the full dim. This normalizes the total adapter capacity for MoE models
            so it is comparable to a dense model. Defaults to False.
        share_expert_adapters (bool): When True, grouped MoE expert linears share one adapter across all local
            experts on the EP rank. Set to False to create one adapter per local expert instead. Defaults to True.
        experts_shared_outer_loras (bool): When True, grouped-expert LoRA
            (``TE*ParallelGroupedLinear`` base modules) uses
            :class:`SharedOuterGroupedExpertAdapter` — ``gate_up`` lora_A and
            ``down`` lora_B are shared across experts (expert_dim=1), matching
            SGLang's ``experts_shared_outer_loras=True`` serving contract (PR
            #21466). Default False preserves the adapter layout selected by
            ``share_expert_adapters``.
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    sequence_parallel_input_regather: bool = False
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: torch.dtype = None
    normalize_moe_lora: bool = False
    share_expert_adapters: bool = True
    experts_shared_outer_loras: bool = False

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        # Skip already transformed modules
        adapter_types = (LinearAdapter, LoRALinear, LoRATopKRouter)
        if isinstance(module, adapter_types):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            _, full_name = ans
            if isinstance(module, nn.Linear) and not is_modelopt_linear(module):
                return LinearAdapter(
                    module,
                    dim=self.dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    lora_A_init_method=self.lora_A_init_method,
                    lora_dtype=self.lora_dtype,
                )

            is_expert = is_expert_linear(full_name)
            attrs = get_adapter_attributes_from_linear(
                module,
                is_expert=is_expert,
                sequence_parallel_input_regather=self.sequence_parallel_input_regather and not is_expert,
            )

            dim = get_effective_lora_dim(
                module, dim=self.dim, normalize_moe_lora=self.normalize_moe_lora, is_expert=is_expert
            )
            dim = align_expert_dim_for_tp(
                module,
                dim,
                normalize_moe_lora=self.normalize_moe_lora,
                is_expert=is_expert,
                input_is_parallel=attrs.input_is_parallel,
            )
            is_grouped_expert_name = is_grouped_expert_linear(full_name)
            use_shared_outer_adapter = self.experts_shared_outer_loras and is_grouped_expert_name
            use_per_expert_adapter = (
                is_grouped_expert_name and not self.share_expert_adapters and not use_shared_outer_adapter
            )
            use_grouped_expert_adapter = use_shared_outer_adapter or use_per_expert_adapter

            enable_op_fuser = (
                not use_grouped_expert_adapter
                and not is_expert
                and getattr(module.config, "use_transformer_engine_op_fuser", False)
                # TP not yet supported
                and parallel_state.get_tensor_model_parallel_world_size() == 1
            )

            logger.info(f"Adding lora to: {full_name}")
            if use_shared_outer_adapter:
                adapter_cls = SharedOuterGroupedExpertAdapter
            elif use_per_expert_adapter:
                adapter_cls = GroupedExpertLinearAdapter
            else:
                adapter_cls = ParallelLinearAdapter
            adapter_kwargs = dict(
                base_linear_name=full_name,
                activation="identity",
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                input_is_parallel=attrs.input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=module.config,
                alpha=self.alpha,
                base_linear_is_parallel=attrs.base_linear_is_parallel,
            )
            if use_grouped_expert_adapter:
                first_param = next(module.parameters())
                adapter_kwargs.update(
                    num_local_experts=module.num_gemms,
                    params_device=first_param.device,
                    params_dtype=first_param.dtype,
                )
            else:
                adapter_kwargs.update(
                    is_expert=is_expert,
                    a2a_experimental=self.a2a_experimental,
                    sequence_parallel_input_regather=self.sequence_parallel_input_regather,
                    disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
                    disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                )
            adapter = adapter_cls(attrs.in_features, attrs.out_features, dim, **adapter_kwargs)
            if isinstance(module, TopKRouter):
                return LoRATopKRouter(module, adapter)
            if enable_op_fuser:
                return TEFusedLoRALinear(module, adapter)
            else:
                return LoRALinear(module, adapter)
        return module


@dataclass
class VLMLoRA(LoRA):
    """
    Implements the LoRA for Vision-Language Models.
    VLMLoRA additionally allows the user to specify whether the language or vision
    models should be frozen.
    For example, a common finetuning workload for multimodal models is to apply adapters to language model and fully
    finetune the vision model.

    """

    freeze_vision_model: bool = True
    freeze_vision_projection: bool = True
    freeze_language_model: bool = True

    def freeze_model(self, model: nn.Module, training: bool = True) -> None:
        modules_to_freeze = []

        model = unwrap_model(model)[0]
        if hasattr(model, "llava_model"):
            model = model.llava_model

        if self.freeze_vision_model and model.vision_model is not None:
            modules_to_freeze.append(model.vision_model)
        if self.freeze_vision_projection and model.vision_projection is not None:
            modules_to_freeze.append(model.vision_projection)
        if self.freeze_language_model and model.language_model is not None:
            modules_to_freeze.append(model.language_model)

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        if training:
            if isinstance(model, list):
                for model_chunk in model:
                    model_chunk.train(mode=True)
            elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.train(mode=True)
            else:
                model.train(mode=True)
