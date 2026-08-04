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

import math
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from megatron.core.transformer.moe.moe_utils import apply_random_logits

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.lora_merge import LoRAMerge


class LoRALinear(AdapterWrapper):
    """An adapter wrapper that adds the output of the adapter to the output of the wrapped module.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    @property
    def weight(self) -> torch.Tensor:
        """Return the effective base weight, including the LoRA delta when enabled."""
        base_weight = self.to_wrap.weight
        if not self._adapter_enabled:
            return base_weight

        linear_in_weight = self.adapter.linear_in.weight
        linear_out_weight = self.adapter.linear_out.weight

        merged_weight = LoRAMerge().merge(
            base_weight,
            linear_out_weight,
            linear_in_weight,
            self.adapter.alpha,
            self.adapter.dim,
            tp_group=getattr(self.adapter, "tp_group", None),
        )
        if merged_weight.shape != base_weight.shape:
            raise RuntimeError(
                "LoRA effective weight shape mismatch: "
                f"base={tuple(base_weight.shape)}, merged={tuple(merged_weight.shape)}"
            )
        return merged_weight.to(device=base_weight.device, dtype=base_weight.dtype)

    @property
    def bias(self) -> torch.Tensor | None:
        """Return the wrapped linear bias."""
        return getattr(self.to_wrap, "bias", None)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass that combines the wrapped module output with the adapter output.

        Args:
            x: Input tensor.
            *args: Additional positional arguments for the wrapped module.
            **kwargs: Additional keyword arguments for the wrapped module.

        Returns:
            A tuple containing:
                - Combined output (linear_output + adapter_output) if adapter is enabled,
                  otherwise just the linear_output
                - Bias term (if present, otherwise None)
        """
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return linear_output, bias
        adapter_output = self.adapter_forward(self.adapter, layernorm_output.contiguous(), *args, **kwargs)
        adapter_output = adapter_output.reshape(linear_output.shape)
        return linear_output + adapter_output, bias


class LoRATopKRouter(AdapterWrapper):
    """Adapter wrapper that applies LoRA to router gating logits."""

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        """Forward pass that adds LoRA delta to router logits before routing."""
        self.to_wrap._maintain_float32_expert_bias()
        jittered_input = self.to_wrap.apply_input_jitter(x)
        logits = self.to_wrap.gating(jittered_input)
        if self._adapter_enabled:
            adapter_output = self.adapter(jittered_input.contiguous())
            logits = logits + adapter_output.to(dtype=logits.dtype)
        if self.to_wrap.config.moe_router_force_load_balancing:
            logits = apply_random_logits(logits)
        return self.to_wrap.routing(logits, *args, **kwargs)


class TEFusedLoRALinear(LoRALinear):
    """LoRA adapter wrapper using Transformer Engine operation fuser"""

    def __init__(self, to_wrap: nn.Module, adapter: nn.Module):
        super().__init__(to_wrap, adapter)
        self._fused_branches: Optional[tuple[te.ops.Sequential, te.ops.Sequential]] = None

    def _make_fused_branches(self) -> tuple[te.ops.Sequential, te.ops.Sequential]:
        """Construct fused modules for main and LoRA branches"""

        # Extract layer size and tensor parallel config
        kwargs = {
            "in_features": self.to_wrap.weight.size(1),
            "out_features": self.to_wrap.weight.size(0),
            "tensor_parallel_mode": None,
            "tensor_parallel_group": None,
            "sequence_parallel": False,
        }
        # TODO: Restore once TP is supported
        # tensor_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
        # if tensor_parallel_size > 1:
        #     kwargs["tensor_parallel_group"] = parallel_state.get_tensor_model_parallel_group()
        #     if isinstance(self.to_wrap, (te.Linear, te.LayerNormLinear)):
        #         kwargs["tensor_parallel_mode"] = self.to_wrap.parallel_mode
        #         kwargs["sequence_parallel"] = self.to_wrap.sequence_parallel
        #     if kwargs["tensor_parallel_mode"] == "row":
        #         kwargs["in_features"] *= tensor_parallel_size
        #     elif kwargs["tensor_parallel_mode"] == "column":
        #         kwargs["out_features"] *= tensor_parallel_size

        # wgrad accumulation fusion
        accumulate_into_main_grad = False
        if isinstance(self.to_wrap, (te.Linear, te.LayerNormLinear)):
            accumulate_into_main_grad = self.to_wrap.fuse_wgrad_accumulation
        kwargs["accumulate_into_main_grad"] = accumulate_into_main_grad

        # Construct fused branches
        main_branch = self._make_main_branch(**kwargs)
        lora_branch = self._make_lora_branch(**kwargs)

        # Get submodule forward hooks
        forward_pre_hooks = []
        forward_post_hooks = []
        for submodule in self.modules():
            for hook in submodule._forward_pre_hooks.values():
                forward_pre_hooks.append((submodule, hook))
            for hook in submodule._forward_hooks.values():
                forward_post_hooks.append((submodule, hook))

        # Attempt to emulate submodule forward hooks if needed
        # Note: Assume hooks do not interact with submodule inputs
        # or outputs since they are internal to the op fuser.
        if forward_pre_hooks:

            def forward_pre_hook(module, *_) -> None:
                for submodule, hook in forward_pre_hooks:
                    # Assume that hook does not interact with
                    # input
                    hook(submodule, None)

            main_branch.register_forward_pre_hook(forward_pre_hook)
        if forward_post_hooks:

            def forward_post_hook(module, *_) -> None:
                for submodule, hook in forward_post_hooks:
                    # Assume that hook does not interact with
                    # input or output
                    hook(submodule, None, None)

            lora_branch.register_forward_hook(forward_post_hook)

        return main_branch, lora_branch

    def _make_main_branch(
        self,
        *,
        in_features: int,
        out_features: int,
        tensor_parallel_mode: Optional[str],
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup],
        sequence_parallel: bool,
        accumulate_into_main_grad: bool,
    ) -> te.ops.Sequential:
        """Construct fused module for main branch (norm + fork + linear)"""

        # Check wrapped linear class
        if not isinstance(self.to_wrap, (te.Linear, te.LayerNormLinear, torch.nn.Linear)):
            raise ValueError(f"Unsupported class for wrapped linear ({self.to_wrap.__class__.__name__})")

        # Ops in main branch
        main_branch = te.ops.Sequential()

        # Norm op
        if isinstance(self.to_wrap, te.LayerNormLinear):
            norm_type = self.to_wrap.normalization
            kwargs = {
                "eps": self.to_wrap.eps,
                "device": "meta",
                "dtype": self.to_wrap.layer_norm_weight.dtype,
                "zero_centered_gamma": self.to_wrap.zero_centered_gamma,
            }
            op = None
            if norm_type == "LayerNorm":
                op = te.ops.LayerNorm(in_features, **kwargs)
                op.weight = self.to_wrap.layer_norm_weight
                op.bias = self.to_wrap.layer_norm_bias
            elif norm_type == "RMSNorm":
                op = te.ops.RMSNorm(in_features, **kwargs)
                op.weight = self.to_wrap.layer_norm_weight
            else:
                raise ValueError(f"Unsupported normalization ({norm_type})")
            main_branch.append(op)
            main_branch.append(te.ops.Quantize(forward=True, backward=False))

        # Fork to LoRA branch
        # Note: GEMM with beta=1 in backward pass
        main_branch.append(te.ops.MakeExtraOutput(in_place=True))

        # Linear op
        weight = self.to_wrap.weight
        bias = self.to_wrap.bias
        if isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None
        op = te.ops.Linear(
            in_features,
            out_features,
            bias=bias is not None,
            device="meta",
            dtype=weight.dtype,
            tensor_parallel_mode=tensor_parallel_mode,
            tensor_parallel_group=tensor_parallel_group,
            sequence_parallel=sequence_parallel,
            accumulate_into_main_grad=accumulate_into_main_grad,
        )
        op.weight = weight
        op.bias = bias
        main_branch.append(op)

        return main_branch

    def _make_lora_branch(
        self,
        *,
        in_features: int,
        out_features: int,
        tensor_parallel_mode: Optional[str],
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup],
        sequence_parallel: bool,
        accumulate_into_main_grad: bool,
    ) -> te.ops.Sequential:
        """Construct fused module for LoRA branch (linear_in + linear_out + add)"""

        from megatron.bridge.peft.utils import ParallelLinearAdapter

        if not isinstance(self.adapter, ParallelLinearAdapter):
            raise ValueError(f"Unsupported class for LoRA adapter ({self.adapter.__class__.__name__})")

        linear_in_weight = self.adapter.linear_in.weight
        linear_out_weight = self.adapter.linear_out.weight
        lora_dim = linear_out_weight.size(1)
        dropout = getattr(self.adapter.dropout, "p", 0.0)
        dropout_position = self.adapter.dropout_position
        scale = self.adapter.alpha / self.adapter.dim

        # Ops in LoRA branch
        lora_branch = te.ops.Sequential()

        # LoRA pre-processing
        if dropout > 0 and dropout_position == "pre":
            lora_branch.append(te.ops.Dropout(dropout))

        # LoRA A linear op
        op = te.ops.Linear(
            in_features,
            lora_dim,
            bias=False,
            device="meta",
            dtype=linear_in_weight.dtype,
            tensor_parallel_mode=tensor_parallel_mode,
            tensor_parallel_group=tensor_parallel_group,
            sequence_parallel=sequence_parallel,
            accumulate_into_main_grad=accumulate_into_main_grad,
        )
        op.weight = linear_in_weight
        lora_branch.append(op)

        # LoRA B linear op
        if tensor_parallel_mode == "column":
            # All-gather along dim -1
            raise NotImplementedError("Column tensor parallelism is not yet supported")
        op = te.ops.Linear(
            lora_dim,
            out_features,
            bias=False,
            device="meta",
            dtype=linear_out_weight.dtype,
            tensor_parallel_mode=None if tensor_parallel_mode is None else "column",
            tensor_parallel_group=tensor_parallel_group,
            sequence_parallel=False,
            accumulate_into_main_grad=accumulate_into_main_grad,
        )
        op.weight = linear_out_weight
        lora_branch.append(op)

        # LoRA post-processing
        if scale != 1:
            lora_branch.append(te.ops.ConstantScale(scale))
        if dropout > 0 and dropout_position == "post":
            lora_branch.append(te.ops.Dropout(dropout))
        if tensor_parallel_mode == "row":
            # All-gather along dim -1
            raise NotImplementedError("Row tensor parallelism is not yet supported")

        # Add with main branch
        # Note: GEMM with beta=1 in forward pass
        lora_branch.append(te.ops.AddExtraInput(in_place=True))

        return lora_branch

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # pylint: disable=C0115,C0116

        # If adapter is disabled, fall back to base forward
        if not self._adapter_enabled:
            return super().forward(x)

        # Construct fused impl if needed
        # Note: We initialize during the first forward pass in
        # case the params are modified after the constructor.
        # Note: The fused impl is stored in a tuple to avoid
        # registering submodules.
        if self._fused_branches is None:
            self._fused_branches = self._make_fused_branches()

        # Apply fused impl
        main_branch, lora_branch = self._fused_branches
        linear_output, linear_input = main_branch(x)
        with te.fp8_autocast(enabled=False):
            out = lora_branch(linear_input, linear_output)
        return out, None


class LinearAdapter(nn.Linear):
    """Linear with LoRA that preserves the base weight and bias checkpoint keys.

    Args:
        orig_linear: The linear module to augment.
        dim: LoRA's dimension (in_features -> dim -> out_features).
        alpha: LoRA's scaling alpha.
        dropout: Dropout probability (default: 0.0).
        dropout_position: Where to apply dropout relative to LoRA (choices: ['pre', 'post'], default='pre').
        lora_A_init_method: Initialization method for lora_A (choices: ['xavier', 'uniform']).
        lora_dtype: Adapter weight dtype. Defaults to the original linear's weight dtype.
    """

    def __init__(
        self,
        orig_linear: nn.Linear,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "pre",
        lora_A_init_method: Literal["xavier", "uniform"] = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize LinearAdapter by copying from original Linear and adding LoRA components.

        Args:
            orig_linear: The original Linear module to adapt.
            dim: LoRA rank dimension.
            alpha: LoRA scaling factor.
            dropout: Dropout probability.
            dropout_position: When to apply dropout ('pre' or 'post' LoRA computation).
            lora_A_init_method: Initialization method for LoRA matrix A.
            lora_dtype: Data type for LoRA weights.
        """
        assert isinstance(orig_linear, nn.Linear)
        super(LinearAdapter, self).__init__(
            in_features=orig_linear.in_features,
            out_features=orig_linear.out_features,
            bias=orig_linear.bias is not None,
            device=orig_linear.weight.device,
            dtype=orig_linear.weight.dtype,
        )
        # copy weights
        self.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.bias.data.copy_(orig_linear.bias.data)
        # initialize the adapter
        self._init_adapter(
            dim=dim,
            alpha=alpha,
            dropout=dropout,
            dropout_position=dropout_position,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )
        self._adapter_enabled = True

    def enable_adapter_layers(self) -> None:
        """Enable the adapter layers, allowing them to contribute to the forward pass output."""
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        """Disable the adapter layers, making the forward pass return only the base module output."""
        self._adapter_enabled = False

    @torch.no_grad
    def _init_adapter(
        self,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "pre",
        lora_A_init_method: Literal["xavier", "uniform"] = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the LoRA weights and freeze the base parameters.

        Args:
            dim: LoRA's dimension (in_features -> dim -> out_features).
            alpha: LoRA's scaling alpha.
            dropout: Dropout probability (default: 0.0).
            dropout_position: Where to apply dropout relative to LoRA (choices: ['pre', 'post'], default='pre').
            lora_A_init_method: Initialization method for lora_A (choices: ['xavier', 'uniform']).
            lora_dtype: Adapter weight dtype. Defaults to the base weight dtype.
        """
        self.dim = dim
        self.alpha = alpha
        self.scale = alpha / dim

        # Freeze original weights
        device = self.weight.device
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        in_features = self.in_features
        out_features = self.out_features
        dtype = lora_dtype or self.weight.dtype

        self.linear_in = nn.Linear(in_features, dim, bias=False, dtype=dtype, device=device)
        self.linear_out = nn.Linear(dim, out_features, bias=False, dtype=dtype, device=device)
        if lora_A_init_method == "xavier":
            torch.nn.init.xavier_uniform_(self.linear_in.weight.data)
        else:
            nn.init.kaiming_uniform_(self.linear_in.weight.data, a=math.sqrt(5))
        self.linear_out.weight.data.fill_(0)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        assert dropout_position in ["pre", "post"], dropout_position
        self.dropout_position = dropout_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining Linear output with LoRA adaptation.

        Args:
            x: Input tensor.

        Returns:
            Combined output from original linear layer and LoRA adaptation.
        """
        # pylint: disable=C0115,C0116
        res = torch.nn.functional.linear(x, self.weight, self.bias)

        if not self._adapter_enabled:
            return res

        if self.dropout_position == "pre":
            x = self.dropout(x)
        lora_res = self.linear_out(self.linear_in(x))
        lora_res = lora_res * self.scale
        if self.dropout_position == "post":
            lora_res = self.dropout(lora_res)
        return res + lora_res
