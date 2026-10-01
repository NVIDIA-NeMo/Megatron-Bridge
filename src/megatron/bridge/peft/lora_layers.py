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
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from megatron.core.transformer.moe.moe_utils import apply_random_logits

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.lora_merge import LoRAMerge
from megatron.bridge.peft.utils import ParallelLinearAdapter


class _AddExpertLoRA(torch.autograd.Function):
    """Accumulate an expert LoRA projection into a consumed grouped-linear output."""

    @staticmethod
    def forward(
        ctx,
        rank_input: torch.Tensor,
        weight: torch.Tensor,
        base_output: torch.Tensor,
        destination: torch.Tensor,
    ):
        if (
            base_output.ndim != 2
            or rank_input.ndim != 2
            or not base_output.is_contiguous()
            or base_output.storage_offset() != 0
            or base_output.untyped_storage().nbytes() != base_output.numel() * base_output.element_size()
            or destination.shape != base_output.shape
            or destination.data_ptr() != base_output.data_ptr()
            or destination.requires_grad
        ):
            raise RuntimeError("Grouped-linear output does not cover one complete contiguous allocation")
        if rank_input.shape[0] != base_output.shape[0] or weight.shape != (
            base_output.shape[1],
            rank_input.shape[1],
        ):
            raise RuntimeError("Expert LoRA projection does not match the grouped-linear output")
        if rank_input.device != base_output.device or weight.device != base_output.device:
            raise RuntimeError("Expert LoRA projection and grouped-linear output must share a device")
        if rank_input.dtype != base_output.dtype or weight.dtype != base_output.dtype:
            raise RuntimeError("Expert LoRA projection and grouped-linear output must share a dtype")

        ctx.save_for_backward(rank_input, weight)
        ctx.mark_dirty(destination)
        torch.addmm(destination, rank_input, weight.t(), beta=1, out=destination)
        return destination

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        rank_input, weight = ctx.saved_tensors
        grad_rank_input = grad_output @ weight if ctx.needs_input_grad[0] else None
        grad_weight = grad_output.t() @ rank_input if ctx.needs_input_grad[1] else None
        grad_base_output = grad_output if ctx.needs_input_grad[2] else None
        return grad_rank_input, grad_weight, grad_base_output, None


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

    def _can_reuse_grouped_expert_output(self, x: torch.Tensor, linear_output: torch.Tensor) -> bool:
        adapter = self.adapter
        grouped_linear_type = getattr(te, "GroupedLinear", None)
        if grouped_linear_type is None or not isinstance(self.to_wrap, grouped_linear_type):
            return False
        if not isinstance(adapter, ParallelLinearAdapter) or not adapter.is_expert:
            return False
        if not isinstance(adapter.activation, nn.Identity) or not isinstance(adapter.dropout, nn.Identity):
            return False
        if not adapter.disable_sequence_parallel_comm:
            return False
        if adapter.config.cpu_offloading and adapter.config.cpu_offloading_activations:
            return False
        linear_in = adapter.linear_in
        linear_out = adapter.linear_out
        surface = adapter.base_linear_name.rsplit(".", maxsplit=1)[-1]
        if surface == "linear_fc1":
            if adapter.input_is_parallel:
                return False
        elif surface == "linear_fc2":
            if not adapter.input_is_parallel:
                return False
        else:
            return False
        return (
            not adapter.use_a2a
            and getattr(adapter.config, "expert_tensor_parallel_size", 1) == 1
            and x.ndim == 2
            and linear_output.ndim == 2
            and linear_in.input_size == x.shape[-1]
            and linear_out.output_size_per_partition == linear_out.output_size
            and linear_out.weight.shape[0] == linear_output.shape[-1]
            and linear_out.bias is None
            and not linear_out.gradient_accumulation_fusion
            and linear_out.gtp_remat_size == 1
        )

    def _reuse_grouped_expert_output(self, x: torch.Tensor, linear_output: torch.Tensor) -> torch.Tensor:
        adapter = self.adapter
        rank_output, _ = adapter.linear_in(x)
        rank_output = adapter.activation(rank_output)
        rank_output = rank_output * (adapter.alpha / adapter.dim)
        destination = linear_output.detach()
        return _AddExpertLoRA.apply(rank_output, adapter.linear_out.weight, linear_output, destination)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        """Forward pass that combines the wrapped module output with the adapter output.

        Args:
            x: Input tensor.
            *args: Additional positional arguments for the wrapped module.
            **kwargs: Additional keyword arguments for the wrapped module.

        Returns:
            When the wrapped module returns a Megatron-style tuple, a
            ``(combined_output, bias)`` tuple; when it returns a bare tensor
            (e.g. a plain ``nn.Linear``), a bare tensor so the wrapper stays a
            drop-in replacement in simple (non-parallel) models.
        """
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return linear_output if not self._base_returns_tuple else (linear_output, bias)
        adapter_input = layernorm_output.contiguous()
        if self._can_reuse_grouped_expert_output(adapter_input, linear_output):
            combined_output = self._reuse_grouped_expert_output(adapter_input, linear_output)
        else:
            combined_output = linear_output.clone()
            del linear_output
            adapter_output = self.adapter_forward(self.adapter, adapter_input, *args, **kwargs)
            adapter_output = adapter_output.reshape(combined_output.shape)
            combined_output.add_(adapter_output)
        if not self._base_returns_tuple:
            return combined_output
        return combined_output, bias


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


class LinearAdapter(nn.Module):
    """Delta-only LoRA adapter for a plain ``nn.Linear``, mirroring :class:`ParallelLinearAdapter`'s role.

    This adapter holds *only* the low-rank LoRA delta (``linear_in`` -> ``linear_out``) and
    produces just the scaled adaptation term. It is intended to be wrapped together with the
    original linear by :class:`LoRALinear` (the ``to_wrap`` / ``adapter`` pattern), so that
    base weights and adapter weights live in distinct submodules and adapter state is
    checkpointed under the ``adapter.`` prefix.

    Args:
        orig_linear: The linear module to augment (only its shape/dtype/device are used; its
            weights are *not* copied).
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
        """Initialize the LoRA delta weights from the original Linear's shape and dtype.

        Args:
            orig_linear: The original Linear module to adapt (weights are not copied).
            dim: LoRA rank dimension.
            alpha: LoRA scaling factor.
            dropout: Dropout probability.
            dropout_position: When to apply dropout ('pre' or 'post' LoRA computation).
            lora_A_init_method: Initialization method for LoRA matrix A.
            lora_dtype: Data type for LoRA weights.
        """
        super().__init__()
        assert isinstance(orig_linear, nn.Linear)
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self._init_adapter(
            dim=dim,
            alpha=alpha,
            dropout=dropout,
            dropout_position=dropout_position,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
            device=orig_linear.weight.device,
            base_dtype=orig_linear.weight.dtype,
        )

    @torch.no_grad
    def _init_adapter(
        self,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "pre",
        lora_A_init_method: Literal["xavier", "uniform"] = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        base_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the LoRA delta weights.

        Args:
            dim: LoRA's dimension (in_features -> dim -> out_features).
            alpha: LoRA's scaling alpha.
            dropout: Dropout probability (default: 0.0).
            dropout_position: Where to apply dropout relative to LoRA (choices: ['pre', 'post'], default='pre').
            lora_A_init_method: Initialization method for lora_A (choices: ['xavier', 'uniform']).
            lora_dtype: Adapter weight dtype. Defaults to the base weight dtype.
            device: Device for the LoRA weights.
            base_dtype: Base weight dtype, used when ``lora_dtype`` is not provided.
        """
        self.dim = dim
        self.alpha = alpha
        self.scale = alpha / dim

        in_features = self.in_features
        out_features = self.out_features
        dtype = lora_dtype or base_dtype

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
        """Compute the scaled LoRA delta only (no base-weight term).

        Args:
            x: Input tensor.

        Returns:
            The scaled low-rank adaptation ``scale * linear_out(linear_in(x))`` with dropout
            applied per ``dropout_position``.
        """
        # pylint: disable=C0115,C0116
        if self.dropout_position == "pre":
            x = self.dropout(x)
        lora_res = self.linear_out(self.linear_in(x))
        lora_res = lora_res * self.scale
        if self.dropout_position == "post":
            lora_res = self.dropout(lora_res)
        return lora_res
