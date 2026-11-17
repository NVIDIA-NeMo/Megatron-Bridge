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

"""ERNIE 5.5 model: Megatron-Core's ``HybridModel`` plus the ERNIE post-construction contract."""

from __future__ import annotations

from typing import Any

import torch
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor, ShardedTensorFactory
from megatron.core.models.hybrid import hybrid_model as mcore_hybrid_model
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch import install_ernie55_fused_ops
from megatron.bridge.models.ernie.modeling_ernie55.hca_attention import check_ernie55_attention_layers
from megatron.bridge.models.ernie.modeling_ernie55.hyper_connection import use_ernie55_hyper_connections
from megatron.bridge.models.ernie.modeling_ernie55.mtp import check_ernie55_mtp_aliases, share_mtp_body_with_last_layer
from megatron.bridge.models.ernie.modeling_ernie55.mtp_loss import PER_ROW_MTP_LOSS_SUPPORTED, per_row_mtp_loss_in
from megatron.bridge.models.ernie.modeling_ernie55.muon import tag_ernie55_muon_params
from megatron.bridge.models.ernie.modeling_ernie55.output_layer import (
    install_ernie55_multimax,
    register_multimax_grad_norm_group,
)


__all__ = ["Ernie55HybridModel"]


class Ernie55HybridModel(HybridModel):
    """``HybridModel`` that applies the ERNIE 5.5 contract right after Megatron-Core builds it.

    Build it with :func:`~megatron.bridge.models.ernie.modeling_ernie55.ernie55_hybrid_stack_spec`
    and Megatron-Core's ``HybridModel`` arguments (passed through unchanged). After construction:

    * every hyper-connection module, the MTP stack's included, uses the ERNIE 5.5 eager fp32
      arithmetic (:mod:`~megatron.bridge.models.ernie.modeling_ernie55.hyper_connection`);
    * the ``multimax`` gradient-norm group is registered with Megatron-Core, on every pipeline
      stage (the optimizer's per-group collectives need the same group list on every rank);
    * the output layer, when this stage has one, applies multimax to the main and the MTP logits
      and owns the ``multimax`` parameters
      (:mod:`~megatron.bridge.models.ernie.modeling_ernie55.output_layer`);
    * with ``ernie_mtp_shared_last_layer``, every MTP depth's inner layers share the Parameters of
      the decoder's last layers (:mod:`~megatron.bridge.models.ernie.modeling_ernie55.mtp`);
    * every attention module is checked to be an ERNIE H or D layer;
    * ``ernie_fused_ops`` takes effect: the Triton kernels' dispatchers are installed in Megatron-Core,
      and the SiTU-GLU kernel is turned on or off for this model's clamp scales
      (:mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch`).

    With ``ernie_per_row_loss`` the forward normalizes every MTP depth per row, with the row's supervised
    tokens counted over the context-parallel group, instead of Megatron-Core's per-rank token mean
    (:mod:`~megatron.bridge.models.ernie.modeling_ernie55.mtp_loss`).

    Shared parameters are written to checkpoints once, under their decoder names: ``state_dict()``
    and ``sharded_state_dict()`` leave out the aliased MTP entries, and loading a state dict fills
    them from the decoder entries, so strict loading works with checkpoints saved either way.
    ``load_state_dict(assign=True)``, which would give the MTP body its own copies, is rejected, and
    every load ends with :func:`~megatron.bridge.models.ernie.modeling_ernie55.mtp.check_ernie55_mtp_aliases`.

    All of this happens before DDP, the optimizer and any checkpoint load see the model.
    """

    def __init__(self, config: TransformerConfig, *args: Any, **kwargs: Any) -> None:
        # Every stage, with or without an output layer: see register_multimax_grad_norm_group.
        register_multimax_grad_norm_group()
        super().__init__(config, *args, **kwargs)
        if self.config.enable_mhc_connections:
            use_ernie55_hyper_connections(self)
        output_layer = getattr(self, "output_layer", None)
        if output_layer is not None:
            install_ernie55_multimax(output_layer, self.config)
        self.ernie55_mtp_aliases: dict[str, str] = {}
        if getattr(self.config, "ernie_mtp_shared_last_layer", False):
            if not self.config.mtp_num_layers:
                raise ValueError("ERNIE 5.5 SD-MTP: ernie_mtp_shared_last_layer is set, but the model has no MTP.")
            if getattr(self, "mtp_process", False):
                self.ernie55_mtp_aliases = share_mtp_body_with_last_layer(self)
                self.register_state_dict_post_hook(_drop_mtp_aliases_from_state_dict)
                self.register_load_state_dict_pre_hook(_fill_mtp_aliases_from_decoder)
                self.register_load_state_dict_post_hook(_check_mtp_aliases_after_load)
        check_ernie55_attention_layers(self)
        tag_ernie55_muon_params(self)
        install_ernie55_fused_ops(self.config)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Megatron-Core's ``HybridModel.forward``, with the per-row MTP loss under ``ernie_per_row_loss``.

        Arguments and return value are ``HybridModel.forward``'s. With ``ernie_per_row_loss`` on a stage that runs
        MTP, Megatron-Core's ``process_mtp_loss`` is replaced by
        :func:`~megatron.bridge.models.ernie.modeling_ernie55.mtp_loss.ernie55_process_mtp_loss` for this call
        (Megatron-Core builds without the grouped MTP roll keep their own loss, which is the same at CP = 1; the
        config validation rejects per-row MTP at CP > 1 on them).
        """
        if (
            PER_ROW_MTP_LOSS_SUPPORTED
            and getattr(self.config, "ernie_per_row_loss", False)
            and getattr(self, "mtp_process", False)
        ):
            with per_row_mtp_loss_in(mcore_hybrid_model):
                return super().forward(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def peft_excluded_module_paths(self) -> frozenset[str]:
        """Modules PEFT leaves untouched: the SD-MTP body and the decoder layers it aliases.

        With ``ernie_mtp_shared_last_layer`` the MTP body holds the decoder's last-layer ``nn.Parameter`` objects in
        its own modules (:func:`~megatron.bridge.models.ernie.modeling_ernie55.mtp.share_mtp_body_with_last_layer`).
        A PEFT method that wraps modules one by one (``LoRALinear`` and friends) would give every shared weight two
        adapters, one per path, and move the parameters behind wrapper attributes, which
        :func:`~megatron.bridge.models.ernie.modeling_ernie55.mtp.check_ernie55_mtp_aliases` then rejects.
        :class:`~megatron.bridge.peft.base.PEFT` skips the paths returned here, so the shared body stays frozen and
        adapters go on every other targeted module. Stages without the MTP return an empty set.

        Returns:
            Dotted module paths, relative to this model, of both sides of every SD-MTP alias.
        """
        return frozenset(
            name.rpartition(".")[0] for alias, source in self.ernie55_mtp_aliases.items() for name in (alias, source)
        )

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        """Megatron-Core's sharded state dict without the MTP entries that alias decoder parameters.

        Args:
            prefix: Module name prefix.
            sharded_offsets: Pipeline offsets (empty at the model level).
            metadata: Sharded-state-dict metadata.

        Returns:
            The sharded state dict.
        """
        state = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.ernie55_mtp_aliases:
            shared = [self.get_parameter(source) for source in sorted(set(self.ernie55_mtp_aliases.values()))]
            aliased = _AliasedTensors(shared)
            alias_keys = {prefix + alias for alias in self.ernie55_mtp_aliases}
            mtp_prefix = f"{prefix}mtp."
            for key in [key for key in state if key.startswith(mtp_prefix)]:
                if key in alias_keys or _drop_aliased_entries(state, key, aliased):
                    del state[key]
        return state


def _drop_mtp_aliases_from_state_dict(
    module: Ernie55HybridModel, state_dict: dict[str, Any], prefix: str, local_metadata: dict[str, Any]
) -> None:
    """``state_dict`` post-hook: drop the aliased MTP entries, which repeat their decoder entries."""
    for alias in module.ernie55_mtp_aliases:
        state_dict.pop(prefix + alias, None)


def _fill_mtp_aliases_from_decoder(
    module: Ernie55HybridModel,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: dict[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """``load_state_dict`` pre-hook: an aliased MTP entry takes its decoder entry; a different value is an error.

    ``assign=True`` is rejected before anything is loaded: it would wrap the decoder entry and the
    MTP entry in two separate Parameters.
    """
    if local_metadata.get("assign_to_params_buffers", False):
        raise RuntimeError(
            "ERNIE 5.5 SD-MTP: load_state_dict(assign=True) would give the MTP body its own copies of the "
            "decoder parameters it shares; load without assign."
        )
    for alias, source in module.ernie55_mtp_aliases.items():
        alias_key, source_key = prefix + alias, prefix + source
        if source_key not in state_dict:
            continue
        if alias_key not in state_dict:
            state_dict[alias_key] = state_dict[source_key]
        elif not torch.equal(state_dict[alias_key], state_dict[source_key]):
            error_msgs.append(
                f"ERNIE 5.5 SD-MTP: {alias_key} shares {source_key}, but the state dict holds different values "
                "for the two."
            )


def _check_mtp_aliases_after_load(module: Ernie55HybridModel, incompatible_keys: object) -> None:
    """``load_state_dict`` post-hook: the load must have kept every alias shared."""
    check_ernie55_mtp_aliases(module)


class _AliasedTensors:
    """Identity and memory of the shared parameters, to find sharded entries that hold them.

    Entries normally hold the parameter itself under its own name. Some modules split one
    parameter into several entries (views), so entries are also matched by memory, which is only
    meaningful for allocated (non-meta) tensors.
    """

    def __init__(self, parameters: list) -> None:
        self.ids = {id(param) for param in parameters}
        self.ranges = [
            (param.data_ptr(), param.data_ptr() + param.numel() * param.element_size())
            for param in parameters
            if param.numel() > 0 and param.device.type != "meta"
        ]

    def holds(self, entry) -> bool:
        if not isinstance(entry, (ShardedTensor, ShardedTensorFactory)) or not isinstance(entry.data, torch.Tensor):
            return False
        data = entry.data
        if id(data) in self.ids:
            return True
        if data.numel() == 0 or data.device.type == "meta":
            return False
        start = data.data_ptr()
        return any(low <= start < high for low, high in self.ranges)


def _drop_aliased_entries(container: dict, key: str, aliased: _AliasedTensors) -> bool:
    """Drop aliased entries below ``container[key]``; return whether ``container[key]`` itself should go."""
    entry = container[key]
    if isinstance(entry, dict):
        children = [child for child in entry if _drop_aliased_entries(entry, child, aliased)]
        for child in children:
            del entry[child]
        return bool(children) and not entry
    return aliased.holds(entry)
