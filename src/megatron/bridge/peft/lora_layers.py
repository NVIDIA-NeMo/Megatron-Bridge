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
from typing import Any, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.utils.import_utils import safe_import


if torch.cuda.is_available():
    bitsandbytes, HAVE_BNB = safe_import("bitsandbytes")
else:
    bitsandbytes = None
    HAVE_BNB = False


class LoRALinear(AdapterWrapper):
    """An adapter wrapper that adds the output of the adapter to the output of the wrapped module.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass that combines the wrapped module output with the adapter output.

        Args:
            x: Input tensor.
            *args: Additional positional arguments for the wrapped module.
            **kwargs: Additional keyword arguments for the wrapped module.

        Returns:
            A tuple containing:
                - Combined output (linear_output + adapter_output)
                - Bias term (if present, otherwise None)
        """
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        adapter_output = self.adapter(layernorm_output.contiguous())
        adapter_output = adapter_output.reshape(linear_output.shape)
        return linear_output + adapter_output, bias


class TELinearAdapter(te.Linear):
    """
    TELinear + LoRA, maintains ckpts structure (i.e. Linear's weight/bias remain at the same FQN)

    The _init_adapter and forward methods provide the LoRA functionality. We want to be able to
    use those inside LinearAdapter but also for monkey-patching modules, without repeating the
    same code -> therefore those are decorated with @staticmethod.

    Args:
        orig_linear: The linear module to augment.
        dim: LoRA's dimension (in_features -> dim -> out_features).
        alpha: LoRA's scaling alpha.
        dropout: Dropout probability (default: 0.0).
        dropout_position: Where to apply dropout relative to LoRA (choices: ['pre', 'post'], default='post').
        lora_A_init_method: Initialization method for lora_A (choices: ['xavier', 'uniform']).
        lora_dtype: Weight's dtype, by default will use orig_linear's but if they
                    are quantized weights (e.g. 4bit) needs to be specified explicitly.
    """

    def __init__(
        self,
        orig_linear: "te.Linear",
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "post",
        lora_A_init_method: Literal["xavier", "uniform"] = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize TELinearAdapter by copying from original TELinear and adding LoRA components.

        Args:
            orig_linear: The original TELinear module to adapt.
            dim: LoRA rank dimension.
            alpha: LoRA scaling factor.
            dropout: Dropout probability.
            dropout_position: When to apply dropout ('pre' or 'post' LoRA computation).
            lora_A_init_method: Initialization method for LoRA matrix A.
            lora_dtype: Data type for LoRA weights.
        """
        assert orig_linear.__class__ == te.Linear
        # TELinear has bias set to empty tensor
        has_bias = orig_linear.bias is not None and orig_linear.bias.shape[0] != 0
        super(TELinearAdapter, self).__init__(
            in_features=orig_linear.in_features,
            out_features=orig_linear.out_features,
            bias=has_bias,
            device=orig_linear.weight.device,
            params_dtype=orig_linear.weight.dtype,
        )
        # copy weights
        self.weight.data.copy_(orig_linear.weight.data)
        if has_bias:
            self.bias.data.copy_(orig_linear.bias.data)
        # initialize the adapter
        TELinearAdapter._init_adapter(
            self,
            dim=dim,
            alpha=alpha,
            dropout=dropout,
            dropout_position=dropout_position,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    @torch.no_grad
    @staticmethod
    def _init_adapter(
        obj: Union["TELinearAdapter", nn.Module],
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "post",
        lora_A_init_method: Literal["xavier", "uniform"] = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Add LoRA weights to obj. The obj is either a LinearAdapter or an nn.Module (when monkey-patching).

        Args:
            obj: Input module to adapt (LinearAdapter or nn.Module).
            dim: LoRA's dimension (in_features -> dim -> out_features).
            alpha: LoRA's scaling alpha.
            dropout: Dropout probability (default: 0.0).
            dropout_position: Where to apply dropout relative to LoRA (choices: ['pre', 'post'], default='post').
            lora_A_init_method: Initialization method for lora_A (choices: ['xavier', 'uniform']).
            lora_dtype: Weight's dtype, by default will use orig_linear's but if they
                        are quantized weights (e.g. 4bit) needs to be specified explicitly.
        """
        obj.dim = dim
        obj.scale = alpha / dim

        # Freeze original weights
        device = obj.weight.device
        obj.weight.requires_grad = False
        if obj.bias is not None:
            obj.bias.requires_grad = False

        in_features = obj.in_features
        out_features = obj.out_features
        dtype = lora_dtype or obj.weight.dtype

        obj.lora_a = nn.Linear(in_features, dim, bias=False, dtype=dtype, device=device)
        obj.lora_b = nn.Linear(dim, out_features, bias=False, dtype=dtype, device=device)
        if lora_A_init_method == "xavier":
            torch.nn.init.uniform_(obj.lora_a.weight.data)
        else:
            nn.init.kaiming_uniform_(obj.lora_a.weight.data, a=math.sqrt(5))
        obj.lora_b.weight.data.fill_(0)
        obj.dropout = nn.Dropout(p=dropout)
        assert dropout_position in ["pre", "post"], dropout_position
        obj.dropout_position = dropout_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining TELinear output with LoRA adaptation.

        Args:
            x: Input tensor.

        Returns:
            Combined output from original linear layer and LoRA adaptation.
        """
        # pylint: disable=C0115,C0116
        res = super(TELinearAdapter, self).forward(x)
        if self.dropout_position == "pre":
            x = self.dropout(x)
        # LoRA fwd is performed in original precision regardless of FP8 enabled
        lora_res = self.lora_b(self.lora_a(x))
        lora_res = lora_res * self.scale
        if self.dropout_position == "post":
            lora_res = self.dropout(lora_res)
        return res + lora_res


class LinearAdapter(nn.Linear):
    """
    Linear + LoRA, maintains ckpts structure (i.e. Linear's weight/bias remain at the same FQN)

    The _init_adapter and forward methods provide the LoRA functionality. We want to be able to
    use those inside LinearAdapter but also for monkey-patching modules, without repeating the
    same code -> therefore those are decorated with @staticmethod.

    Args:
        orig_linear: The linear module to augment.
        dim: LoRA's dimension (in_features -> dim -> out_features).
        alpha: LoRA's scaling alpha.
        dropout: Dropout probability (default: 0.0).
        dropout_position: Where to apply dropout relative to LoRA (choices: ['pre', 'post'], default='post').
        lora_A_init_method: Initialization method for lora_A (choices: ['xavier', 'uniform']).
        lora_dtype: Weight's dtype, by default will use orig_linear's but if they
                   are quantized weights (e.g. 4bit) needs to be specified explicitly.
    """

    def __init__(
        self,
        orig_linear: nn.Linear,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "post",
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
        LinearAdapter._init_adapter(
            self,
            dim=dim,
            alpha=alpha,
            dropout=dropout,
            dropout_position=dropout_position,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    @torch.no_grad
    @staticmethod
    def _init_adapter(
        obj: Union["LinearAdapter", nn.Module],
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "post",
        lora_A_init_method: Literal["xavier", "uniform"] = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Add LoRA weights to obj. The obj is either a LinearAdapter or an nn.Module (when monkey-patching).

        Args:
            obj: Input module to adapt (LinearAdapter or nn.Module).
            dim: LoRA's dimension (in_features -> dim -> out_features).
            alpha: LoRA's scaling alpha.
            dropout: Dropout probability (default: 0.0).
            dropout_position: Where to apply dropout relative to LoRA (choices: ['pre', 'post'], default='post').
            lora_A_init_method: Initialization method for lora_A (choices: ['xavier', 'uniform']).
            lora_dtype: Weight's dtype, by default will use orig_linear's but if they
                       are quantized weights (e.g. 4bit) needs to be specified explicitly.
        """
        obj.dim = dim
        obj.scale = alpha / dim

        # Freeze original weights
        device = obj.weight.device
        obj.weight.requires_grad = False
        if obj.bias is not None:
            obj.bias.requires_grad = False

        in_features = obj.in_features
        out_features = obj.out_features
        dtype = lora_dtype or obj.weight.dtype

        obj.lora_a = nn.Linear(in_features, dim, bias=False, dtype=dtype, device=device)
        obj.lora_b = nn.Linear(dim, out_features, bias=False, dtype=dtype, device=device)
        if lora_A_init_method == "xavier":
            torch.nn.init.uniform_(obj.lora_a.weight.data)
        else:
            nn.init.kaiming_uniform_(obj.lora_a.weight.data, a=math.sqrt(5))
        obj.lora_b.weight.data.fill_(0)
        obj.dropout = nn.Dropout(p=dropout)
        assert dropout_position in ["pre", "post"], dropout_position
        obj.dropout_position = dropout_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining Linear output with LoRA adaptation.

        Args:
            x: Input tensor.

        Returns:
            Combined output from original linear layer and LoRA adaptation.
        """
        # pylint: disable=C0115,C0116
        # If LinearAdapter is used to monkey-patch a nn.Linear module, we want to use nn.Linear's
        # forward in the case where it uses quantized weights. We store a reference to nn.Linear's
        # forward in `super_fwd` attribute. If the attribute does not exist we do the usual linear.
        if (fwd := getattr(self, "super_fwd", None)) is not None:
            assert fwd != self.forward
            res = fwd(x)
        else:
            res = torch.nn.functional.linear(x, self.weight, self.bias)

        if self.dropout_position == "pre":
            x = self.dropout(x)
        lora_res = self.lora_b(self.lora_a(x))
        lora_res = lora_res * self.scale
        if self.dropout_position == "post":
            lora_res = self.dropout(lora_res)
        return res + lora_res


def patch_linear_module(
    orig_linear: Union[nn.Linear, "te.Linear"],
    dim: int = 8,
    alpha: int = 32,
    dropout: float = 0.0,
    dropout_position: Literal["pre", "post"] = "post",
    lora_A_init_method: Literal["xavier", "uniform"] = "xavier",
    lora_dtype: Optional[torch.dtype] = None,
) -> Union[nn.Linear, "te.Linear"]:
    """Monkey-patch a nn.Linear or te.Linear to be a LinearAdapter.

    This function replaces a nn.Linear with a LinearAdapter without copying weights,
    making it suitable for cases where the original module was initialized with meta device.

    The orig_linear might not contain valid weights, for example, the given orig_linear was
    initialized within a context-manager that uses a "meta" device. Therefore, we cannot copy
    the weight/bias from the orig_linear to the LinearAdapter, since those have not been allocated.

    To circumvent this scenario, LinearAdapter's additional functionality (_init_adapter, _forward)
    is based on static functions, so that we can use them for patching or when allocating a
    new LinearAdapter object.

    Args:
        orig_linear: The module we add adapter to.
        dim: LoRA dimension. Defaults to 8.
        alpha: LoRA alpha scale. Defaults to 32.
        dropout: Dropout probability. Defaults to 0.0.
        dropout_position: Location to apply dropout wrt LoRA.
            Defaults to 'post' (choices: 'pre', 'post').
        lora_A_init_method: LoRA_A initialization method. Defaults to 'xavier'.
        lora_dtype: LoRA weights' dtype. By default will use orig_linear's dtype
            but orig_linear might use non-trainable dtype (e.g., 4bit), in which case the user must
            specify the dtype manually. Defaults to None.

    Returns:
        The monkey-patched (nn.Linear + LoRA) nn.Module.

    Raises:
        NotImplementedError: If orig_linear is not nn.Linear or te.Linear.
        AssertionError: If orig_linear already has super_fwd attribute.
    """
    assert isinstance(orig_linear, nn.Linear) or (orig_linear.__class__ == te.Linear)
    assert not hasattr(orig_linear, "super_fwd"), orig_linear.super_fwd

    if isinstance(orig_linear, nn.Linear):
        LinearAdapter._init_adapter(orig_linear, dim, alpha, dropout, dropout_position, lora_A_init_method, lora_dtype)
        cls = orig_linear.__class__
        new_cls = type("PatchedLinearAdapter", (LinearAdapter, cls), {})
    elif orig_linear.__class__ == te.Linear:
        TELinearAdapter._init_adapter(
            orig_linear, dim, alpha, dropout, dropout_position, lora_A_init_method, lora_dtype
        )
        cls = orig_linear.__class__
        new_cls = type("PatchedTELinearAdapter", (TELinearAdapter, cls), {})
    else:
        raise NotImplementedError("Expected isinstance(orig_linear, (nn.Linear, te.Linear))")

    # If the model uses quantized weights, we want to use orig_linear's forward
    if (
        HAVE_BNB
        and getattr(orig_linear, "quant_state", None) is not None
        and orig_linear.quant_state.__class__ == bitsandbytes.functional.QuantState
    ):
        orig_linear.super_fwd = orig_linear.forward

    orig_linear.__class__ = new_cls
    return orig_linear
