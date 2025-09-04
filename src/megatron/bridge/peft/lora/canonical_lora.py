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
from torch import nn

from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora.canonical_lora_layers import (
    LoRALinearSplitFC1UpGate,
    LoRALinearSplitQKV,
    ModuleDict,
)
from megatron.bridge.peft.lora.lora_layers import LinearAdapter, LoRALinear
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.utils import ParallelLinearAdapter, get_adapter_attributes_from_linear, is_expert_linear


logger = logging.getLogger(__name__)


@dataclass
class CanonicalLoRA(PEFT, ModuleMatcher):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.
    Canonical LoRA applies LoRA on Q, K, V projection matrices separately, as well as Up and Gate projection
    matrices separately. This follows more closely with Huggingface's implementation of LoRA.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_q', 'linear_k', 'linear_v', 'linear_proj',
                                           'linear_fc1_up', 'linear_fc1_gate', 'linear_fc2'].
                - 'linear_q', 'linear_k', 'linear_v': Apply LoRA to the linear layer used for query, key, and value
                        projections in self-attention. This is fused into one matrix in LoRA, but left as three
                        separate matrices in Canonical LoRA.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1_up', 'linear_fc1_gate': Apply LoRA to the Up proj and Gate proj layers.
                        These two together constitute the first fully-connected layer in MLP in LoRA.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_q', '*.layers.1.*.linear_q'] to add LoRA to only linear_q
                on the first two layers.
        exclude_modules (List[str], optional): A list of module names not to apply LoRA to. It will
            match all nn.Linear & nn.Linear-adjacent modules whose name does not match any string in
            exclude_modules. If used, will require target_modules to be empty list or None.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.
        lora_A_init_method (str): Initialization method for LoRA A matrix. Defaults to "xavier".
        lora_B_init_method (str): Initialization method for LoRA B matrix. Defaults to "zero".
    """

    target_modules: List[str] = field(
        default_factory=lambda: [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_proj",
            "linear_fc1_up",
            "linear_fc1_gate",
            "linear_fc2",
        ]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"

    def __post_init__(self) -> None:
        """
        Initialize the canonical mapping and call the parent post_init.

        Construct a mapping from the target module as supported in LoRA() to the specific parts of the layer for which
        adapter is applied.

        For example, if user specifies target_module = ['linear_q', 'linear_k', 'linear_proj', 'linear_fc1_up'], then
        canonical_lora_mapping = {
            "linear_qkv": {'linear_q', 'linear_k'},
            "linear_proj": {'linear_proj'},  # the value of this key does not matter
            "linear_fc1": {'linear_fc1_up'},
        }

        If user specifies target_module = ['*.layers.0.*.linear_q', '*.layers.1.*.linear_q'], then
        canonical_lora_mapping = {
            "'*.layers.0.*.linear_qkv'": {'linear_q'},
            "'*.layers.1.*.linear_qkv'": {'linear_q'},
        }

        """
        for target in self.target_modules:
            assert not target.endswith("linear_qkv"), (
                "Canonical LoRA does not support target 'linear_qkv'. Either use 'linear_qkv' with LoRA() or "
                "use ['linear_q', 'linear_k', 'linear_v'] with Canonical LoRA"
            )
            assert not target.endswith("linear_fc1"), (
                "Canonical LoRA does not support target 'linear_fc1'. Either use 'linear_fc1' with LoRA() or "
                "use ['linear_fc1_up', 'linear_fc1_gate'] with Canonical LoRA"
            )

            if "linear_q" in target:
                self.canonical_mapping[target.replace("linear_q", "linear_qkv")].add("linear_q")
            elif "linear_k" in target:
                self.canonical_mapping[target.replace("linear_k", "linear_qkv")].add("linear_k")
            elif "linear_v" in target:
                self.canonical_mapping[target.replace("linear_v", "linear_qkv")].add("linear_v")
            elif "linear_fc1_up" in target:
                self.canonical_mapping[target.replace("linear_fc1_up", "linear_fc1")].add("linear_fc1_up")
            elif "linear_fc1_gate" in target:
                self.canonical_mapping[target.replace("linear_fc1_gate", "linear_fc1")].add("linear_fc1_gate")
            else:
                self.canonical_mapping[target].add(target)

    def transform(self, m: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (Optional[str]): Name of the module (if applicable). Defaults to None.
            prefix (Optional[str]): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """

        # Skip already transformed modules
        if isinstance(m, (LinearAdapter, LoRALinear, LoRALinearSplitQKV, LoRALinearSplitFC1UpGate)):
            return m

        if (ans := self.match(m, name, prefix)) is not None:
            (match, full_name) = ans
            if isinstance(m, nn.Linear):
                return LinearAdapter(
                    m, dim=self.dim, alpha=self.alpha, dropout=self.dropout, lora_A_init_method=self.lora_A_init_method
                )

            input_is_parallel, in_features, out_features, disable_sp_comm, base_linear_is_parallel = (
                get_adapter_attributes_from_linear(m)
            )

            adapter_kwargs = dict(
                dim=self.dim,
                base_linear_name=full_name,
                activation="identity",
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(m, "config", None),
                alpha=self.alpha,
                is_expert=is_expert_linear(full_name),
                disable_sequence_parallel_comm=disable_sp_comm,
                base_linear_is_parallel=base_linear_is_parallel,
            )
            if name in ["linear_proj", "linear_fc2"]:
                adapter = ParallelLinearAdapter(in_features, out_features, **adapter_kwargs)
                logger.info(f"Adding lora to: {full_name}")
                return LoRALinear(m, adapter)

            canonical_submodules = self.canonical_mapping[match]
            logger.info(f"Adding lora to: {full_name} ({canonical_submodules})")
            if name == "linear_qkv":
                adapter_q, adapter_k, adapter_v = None, None, None
                kv_out_features = m.config.kv_channels * m.config.num_query_groups
                if "linear_q" in canonical_submodules:
                    adapter_q = ParallelLinearAdapter(in_features, in_features, **adapter_kwargs)
                if "linear_k" in canonical_submodules:
                    adapter_k = ParallelLinearAdapter(in_features, kv_out_features, **adapter_kwargs)
                if "linear_v" in canonical_submodules:
                    adapter_v = ParallelLinearAdapter(in_features, kv_out_features, **adapter_kwargs)
                adapters = ModuleDict({"adapter_q": adapter_q, "adapter_k": adapter_k, "adapter_v": adapter_v})
                return LoRALinearSplitQKV(m, adapters)

            if name == "linear_fc1":
                adapter_up, adapter_gate = None, None
                if "linear_fc1_up" in canonical_submodules:
                    adapter_up = ParallelLinearAdapter(in_features, out_features // 2, **adapter_kwargs)
                if "linear_fc1_gate" in canonical_submodules:
                    adapter_gate = ParallelLinearAdapter(in_features, out_features // 2, **adapter_kwargs)
                adapters = ModuleDict({"adapter_up": adapter_up, "adapter_gate": adapter_gate})
                return LoRALinearSplitFC1UpGate(m, adapters)

        return m

    def merge(self, model):
        """Merge canonical LoRA adapter weights into base model weights.

        Args:
            model: The model with canonical LoRA adapters applied

        Returns:
            The model with canonical LoRA adapters merged into base weights
        """
        # Use the same pattern as LoRAMerge
        merge_transform = CanonicalLoRAMerge()

        return merge_transform(model, training=False)


class CanonicalLoRAMerge(PEFT):
    """Implements the canonical LoRA weight merge for parameter-efficient fine-tuning."""

    @torch.no_grad()
    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """Merge canonical LoRA adapters with base model weights."""
        if isinstance(module, LoRALinearSplitQKV):
            logging.info(f"merging LoRALinearSplitQKV {(prefix if prefix else '') + '.' + (name if name else '')}")
            base_weight = module.to_wrap.weight
            config = module.to_wrap.config

            # Calculate dimensions for Q, K, V slices
            kv_channels = config.kv_channels
            num_query_groups = config.num_query_groups
            num_attention_heads = config.num_attention_heads
            heads_per_group = num_attention_heads // num_query_groups

            # Merge each adapter if it exists
            for adapter_name in ["adapter_q", "adapter_k", "adapter_v"]:
                if adapter_name not in module.adapter or module.adapter[adapter_name] is None:
                    continue

                adapter = module.adapter[adapter_name]
                lora_weight = (
                    adapter.alpha
                    / adapter.dim
                    * adapter.linear_out.weight.to(base_weight.device)
                    @ adapter.linear_in.weight.to(base_weight.device)
                )

                # Determine which slice to merge into
                if adapter_name == "adapter_q":
                    q_size = heads_per_group * num_query_groups * kv_channels
                    start_idx = 0
                    end_idx = q_size
                elif adapter_name == "adapter_k":
                    q_size = heads_per_group * num_query_groups * kv_channels
                    k_size = num_query_groups * kv_channels
                    start_idx = q_size
                    end_idx = start_idx + k_size
                elif adapter_name == "adapter_v":
                    q_size = heads_per_group * num_query_groups * kv_channels
                    k_size = num_query_groups * kv_channels
                    start_idx = q_size + k_size
                    end_idx = start_idx + num_query_groups * kv_channels
                else:
                    continue

                # Merge into the appropriate slice of base weight
                if base_weight.data[start_idx:end_idx].shape == lora_weight.shape:
                    base_weight.data[start_idx:end_idx] += lora_weight
                else:
                    logger.warning(f"Skipping merge for {adapter_name} - shape mismatch: {base_weight.data[start_idx:end_idx].shape} vs {lora_weight.shape}")

            # Set merged flag to gate future adapter computation
            setattr(module, "_merged", True)
            return module

        elif isinstance(module, LoRALinearSplitFC1UpGate):
            logging.info(f"merging LoRALinearSplitFC1UpGate {(prefix if prefix else '') + '.' + (name if name else '')}")
            base_weight = module.to_wrap.weight

            # Merge each adapter if it exists
            for adapter_name in ["adapter_gate", "adapter_up"]:
                if adapter_name not in module.adapter or module.adapter[adapter_name] is None:
                    continue

                adapter = module.adapter[adapter_name]
                lora_weight = (
                    adapter.alpha
                    / adapter.dim
                    * adapter.linear_out.weight.to(base_weight.device)
                    @ adapter.linear_in.weight.to(base_weight.device)
                )

                # Determine slice for gate vs up
                if adapter_name == "adapter_gate":
                    start_idx = 0
                    end_idx = base_weight.shape[0] // 2
                elif adapter_name == "adapter_up":
                    start_idx = base_weight.shape[0] // 2
                    end_idx = base_weight.shape[0]
                else:
                    continue

                # Merge into the appropriate slice
                if base_weight.data[start_idx:end_idx].shape == lora_weight.shape:
                    base_weight.data[start_idx:end_idx] += lora_weight
                else:
                    logger.warning(f"Skipping merge for {adapter_name} - shape mismatch: {base_weight.data[start_idx:end_idx].shape} vs {lora_weight.shape}")

            # Set merged flag to gate future adapter computation
            setattr(module, "_merged", True)
            return module

        return module
