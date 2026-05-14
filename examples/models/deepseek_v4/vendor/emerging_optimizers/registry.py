# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from inspect import signature
from typing import Any, Callable, Concatenate

from absl import logging
from torch import optim
from torch.optim.optimizer import ParamsT


_OPTIMIZERS: dict[str, type[optim.Optimizer]] = {}


def register_optimizer(name: str) -> Callable[[type[optim.Optimizer]], type[optim.Optimizer]]:
    """Decorator to register an optimizer class in the registry."""

    def decorator(cls: type[optim.Optimizer]) -> type[optim.Optimizer]:
        if name.lower() in _OPTIMIZERS:
            raise ValueError(f"Optimizer {name} already registered.")
        _OPTIMIZERS[name.lower()] = cls
        return cls

    return decorator


def get_optimizer_name_list() -> list[str]:
    """Returns the list of optimizer names in the registry."""
    return list(_OPTIMIZERS.keys())


def get_optimizer_cls(name: str) -> type[optim.Optimizer]:
    """Returns the optimizer class from the registry.

    Args:
        name: The name of the optimizer to get.

    Returns:
        The optimizer class.

    Raises:
        ValueError: If the optimizer is not found in the registry.

    Warning:
        To get the optimizer class, you need to import the optimizer module first so that the
        register_optimizer decorator is called and the optimizer is registered. See example below.

    Example:
        >>> from emerging_optimizers.orthogonalized_optimizers import muon
        >>> from emerging_optimizers import registry
        >>> opt_cls = registry.get_optimizer_cls("muon")
        >>> opt_cls
        <class 'emerging_optimizers.orthogonalized_optimizers.muon.Muon'>
    """
    logging.debug(f"Available optimizers: {list(_OPTIMIZERS.keys())}")
    optimizer = _OPTIMIZERS.get(name.lower())
    if optimizer is None:
        raise ValueError(f"Optimizer {name} not found in the registry.")
    return optimizer


def validate_optimizer_args(opt_cls: type, kwargs: dict[str, Any]) -> None:
    """Checks if kwargs are valid for the optimizer class signature."""
    sig = signature(opt_cls)

    supported_params = set[str](sig.parameters.keys())
    unknown_args = set[str](kwargs.keys()) - supported_params
    unknown_args.discard("params")
    if unknown_args:
        raise TypeError(
            f"Optimizer '{opt_cls.__name__}' does not accept arguments: {unknown_args}.\n"
            f"Valid options are: {list(supported_params)}"
        )


def get_configured_optimizer_cls(name: str, **kwargs: Any) -> Callable[Concatenate[ParamsT, ...], optim.Optimizer]:
    """Returns a callable that creates an optimizer with the given arguments."""
    opt_cls = get_optimizer_cls(name)
    validate_optimizer_args(opt_cls, kwargs)

    return partial(opt_cls, **kwargs)
