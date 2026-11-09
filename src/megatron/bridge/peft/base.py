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
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, TypeVar, Union

import torch
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.recompute import maybe_enable_recompute_inputs_grad
from megatron.bridge.peft.walk_utils import walk


logger: logging.Logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType", bound=Union[nn.Module, list[MegatronModule]])


def excluded_module_paths(model: nn.Module) -> frozenset[str]:
    """Module paths a model keeps out of PEFT.

    A model, or one pipeline chunk of it, may define ``peft_excluded_module_paths()`` returning the dotted paths
    (as ``named_modules()`` reports them) of modules that no PEFT method may replace or wrap module by module. The
    case this exists for is parameter sharing by aliasing: ERNIE 5.5's shared-decoder MTP body holds the very
    ``nn.Parameter`` objects of the decoder's last layer in its own modules, so wrapping either side would give the
    shared weights two independent adapters and move them behind wrapper attributes. Modules on the returned paths
    keep their frozen base parameters; everything else is transformed as usual.

    Args:
        model: A model chunk, possibly behind a wrapper that exposes it as ``.module`` (Float16Module, DDP).

    Returns:
        The excluded paths relative to ``model`` (prefixed with ``module.`` when the method lives on the wrapped
        module), or an empty set when the model does not define the method.

    Raises:
        TypeError: If the method returns anything but an iterable of nonempty strings.
    """
    target, prefix = model, ""
    getter = getattr(target, "peft_excluded_module_paths", None)
    if getter is None and isinstance(getattr(model, "module", None), nn.Module):
        target, prefix = model.module, "module."
        getter = getattr(target, "peft_excluded_module_paths", None)
    if getter is None:
        return frozenset()
    declared = getter()
    message = (
        f"{type(target).__name__}.peft_excluded_module_paths() must return an iterable of nonempty module paths, "
        f"got {declared!r}"
    )
    if isinstance(declared, (str, bytes)):
        raise TypeError(message)
    try:
        paths = list(declared)  # materialize once: a generator must not be consumed by the check below
    except TypeError as error:
        raise TypeError(message) from error
    if not all(isinstance(path, str) and path for path in paths):
        raise TypeError(message)
    return frozenset(prefix + path for path in paths)


def _skipping_excluded(func, excluded: frozenset[str]):
    """Wrap a walk callback so that it returns modules on ``excluded`` paths unchanged."""

    def guarded(module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None, **kwargs):
        full_name = f"{prefix}.{name}" if prefix else name
        if full_name in excluded:
            return module
        return func(module, name=name, prefix=prefix, **kwargs)

    return guarded


@dataclass
class PEFT(ABC):
    """Abstract base class for Parameter-Efficient Fine-Tuning (PEFT) methods.

    This class defines the interface for PEFT methods, which are used to fine-tune
    large language models efficiently by modifying only a small subset of the model's
    parameters.

    Example:
        class MyPEFT(PEFT):
            def transform(self, module, name=None, prefix=None):
                # Implement the transform logic
                pass

        from megatron.bridge.models import get_base_model

        peft = MyPEFT()
        base_model = get_base_model(model_config)  # Returns list[MegatronModule]
        adapted_model = peft(base_model, training=True)
    """

    # Runtime state that should not be serialized in checkpoints
    params_to_save: set[str] = field(default_factory=set, init=False, repr=False)

    @abstractmethod
    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """Transform a single module according to the PEFT method.

        This method is called for each module in the model during the PEFT application process.
        It should be implemented by subclasses to define how individual modules are transformed
        for the specific PEFT technique.

        Args:
            module (nn.Module): The individual module to be transformed.
            name (Optional[str]): The name of the module within the model structure. Defaults to None.
            prefix (Optional[str]): A prefix to be added to the module name, typically used for
                                    nested modules. Defaults to None.

        Returns:
            nn.Module: The transformed module. This can be the original module with modifications,
                       a new module replacing the original, or the original module if no
                       transformation is needed for this specific module.

        Note:
            This method is automatically called for each module in the model when the PEFT
            instance is applied to the model using the __call__ method.
        """
        raise NotImplementedError("The transform method should be implemented by subclasses.")

    def __call__(self, model: ModelType, training: bool = True) -> ModelType:
        """Apply the PEFT method to the entire model.

        This method freezes the model parameters and walks through the model
        structure, applying the transform method to each module. Modules a model chunk
        names in ``peft_excluded_module_paths()`` are left as they are (see
        :func:`excluded_module_paths`).

        Args:
            model: The model to be fine-tuned. Can be a single model or a list of model chunks
                   (for pipeline parallelism).
            training (bool): Whether the model will be used for training. If False,
                           additional freezing may be applied. Defaults to True.

        Returns:
            The same type as the input model, transformed with PEFT applied.
        """
        if isinstance(self, ModuleMatcher):
            # Rebuild alias/match bookkeeping from the *current* target_modules so that
            # any post-construction mutation (common in recipes) is respected. Then walk
            # the model once to record which aliases matched and warn on typos.
            self._init_target_match_state()

            def _validate_only(
                module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None
            ) -> nn.Module:
                self.match(module, name, prefix)
                return module

            # Skip the excluded modules here too, so a target that only names excluded modules is reported
            # as unmatched instead of being counted as a hit that then adapts nothing.
            self._walk_model(model, _validate_only, skip_excluded=True)
            self._validate_target_matches()
            self._reset_target_match_state()

        self.freeze_model(model, training=training)

        self._walk_model(model, self.transform, skip_excluded=True)

        if training:
            maybe_enable_recompute_inputs_grad(model)

        if not training:
            self.freeze_model(model, training=training)

        # Set model training mode appropriately
        if isinstance(model, list):
            for model_chunk in model:
                model_chunk.train(mode=training)
        else:
            model.train(mode=training)

        return model

    def _walk_model(self, model: ModelType, func, *, skip_excluded: bool = False) -> None:
        """Apply ``func`` to every module of ``model`` (a module, a DDP-wrapped module or a list of chunks).

        With ``skip_excluded``, the modules a chunk names in ``peft_excluded_module_paths()`` are returned
        unchanged (see :func:`excluded_module_paths`); each chunk's exclusions apply to that chunk only.
        """

        def prepared(root: nn.Module):
            if not skip_excluded:
                return func
            excluded = excluded_module_paths(root)
            return _skipping_excluded(func, excluded) if excluded else func

        if isinstance(model, list):
            for model_chunk in model:
                walk(model_chunk, prepared(model_chunk))
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            walk(model.module, prepared(model.module))
        else:
            walk(model, prepared(model))

    def enable_adapter_layers(self, model: ModelType) -> None:
        """Enable adapter layers for all PEFT-wrapped modules in the model."""

        def enable(module: nn.Module) -> nn.Module:
            method = getattr(module, "enable_adapter_layers", None)
            if callable(method):
                method()
            return module

        self._walk_model(model, enable)

    def disable_adapter_layers(self, model: ModelType) -> None:
        """Disable adapter layers for all PEFT-wrapped modules in the model."""

        def disable(module: nn.Module) -> nn.Module:
            method = getattr(module, "disable_adapter_layers", None)
            if callable(method):
                method()
            return module

        self._walk_model(model, disable)

    @contextmanager
    def disable_adapter(self, model: ModelType):
        """
        Disables the adapter module.
        """
        adapter_states: list[tuple[nn.Module, bool]] = []

        def disable(module: nn.Module) -> nn.Module:
            method = getattr(module, "disable_adapter_layers", None)
            if callable(method):
                adapter_states.append((module, getattr(module, "_adapter_enabled", True)))
                method()
            return module

        try:
            self._walk_model(model, disable)
            yield
        finally:
            for module, was_enabled in adapter_states:
                method_name = "enable_adapter_layers" if was_enabled else "disable_adapter_layers"
                getattr(module, method_name)()

    def freeze_model(self, model: ModelType, training: bool = True) -> None:
        """Apply a default freeze method to the model.

        This method freezes all the model parameters. This method can be overridden by subclasses to
        implement custom freeze strategies (e.g. freeze only parts of the model)

        Args:
            model: The model to be fine-tuned.
            training (bool): Whether the model is being used for training. Affects training mode handling.
        """

        def freeze_parameters(module):
            """Freeze all parameters in a module."""
            for param in module.parameters(recurse=False):
                param.requires_grad = False
            return module

        self._walk_model(model, freeze_parameters)

        if training:
            if isinstance(model, list):
                for model_chunk in model:
                    model_chunk.train(mode=True)
            elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.train(mode=True)
            else:
                model.train(mode=True)

    def set_params_to_save(self, model: ModelType) -> None:
        """Set parameters to be saved for PEFT checkpointing.

        This method identifies which parameters should be saved during checkpointing
        to reduce storage requirements (only adapter parameters, not the full model).

        Args:
            model: The model after PEFT has been applied.
        """
        # Handle both single models and lists of models
        models_to_process = model if isinstance(model, list) else [model]

        self.params_to_save = set()
        for model_chunk in models_to_process:
            # Add all trainable parameters (adapters)
            for name, param in model_chunk.named_parameters():
                if param.requires_grad:
                    self.params_to_save.add(name)

            # Add any relevant buffers (e.g., running stats from batch norm)
            for module_name, module in model_chunk.named_modules():
                if hasattr(module, "track_running_stats"):
                    for buffer_name, buffer in module.named_buffers():
                        if buffer is not None:
                            self.params_to_save.add(module_name + "." + buffer_name)

    def adapter_key_filter(self, key) -> bool:
        """Filter function for adapter parameters during checkpointing.

        This method determines if a parameter should be included in checkpoints.
        Used to save only adapter weights, not the full model.

        Args:
            key (str or tuple): Parameter name/key to check. Can be a string for regular
                               checkpointing or a tuple for distributed checkpointing.

        Returns:
            bool: True if the parameter should be saved.
        """
        # Handle distributed checkpointing where keys can be tuples
        if isinstance(key, tuple):
            return key[1].requires_grad

        # Handle regular string keys
        return key in self.params_to_save or ".adapter." in key or key.endswith(".adapters")
