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

"""Hybrid stack spec for ERNIE 5.5: Megatron-Core's ``hybrid_stack_spec`` with the ERNIE layers."""

from __future__ import annotations

import copy
import dataclasses
import functools

from megatron.core.transformer.experimental_attention_variant.csa import CompressedSparseAttentionSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.ernie.modeling_ernie55.hca_attention import (
    Ernie55HCACoreAttention,
    Ernie55HCASelfAttention,
    Ernie55HCASelfAttentionSubmodules,
)
from megatron.bridge.models.ernie.modeling_ernie55.hybrid_stack import Ernie55HybridStack
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_attention import Ernie55LatentMQASelfAttention
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_core import (
    Ernie55LatentMQACoreAttention,
    Ernie55LatentMQACoreSubmodules,
)
from megatron.bridge.models.ernie.modeling_ernie55.moe import Ernie55MoELayer, Ernie55SplitFeatureRouter
from megatron.bridge.models.ernie.modeling_ernie55.norm import (
    downcasting_norm,
    ernie55_rmsnorm,
    ernie55_rmsnorm_spec,
)


__all__ = ["ernie55_hybrid_stack_spec"]

# Hybrid slots Megatron-Core may build an ``H`` (compress ratio 128) layer from: ``hca_layer``
# takes precedence when set; otherwise ``qk_layernorm`` picks ``csa_qk_layernorm_layer`` over
# ``csa_layer``.
_HCA_LAYER_SLOTS = ("csa_layer", "csa_qk_layernorm_layer", "hca_layer")


def _copy_spec_tree(node: object, memo: dict[int, object]) -> object:
    """Copy module specs, submodule dataclasses, ``functools.partial`` objects and containers.

    Every other leaf is shared: module classes, functions and similar immutable objects.
    ``copy.deepcopy`` would return classes unchanged too, but it fails on the placeholders
    Megatron-Core substitutes for Transformer Engine modules when TE is not installed; sharing
    them keeps the factory usable on machines without TE. Partials are rebuilt around the same
    function with copied arguments, because Megatron-Core binds submodule dataclasses into them.
    Objects referenced more than once stay shared within the copy.
    """
    if id(node) in memo:
        return memo[id(node)]
    if isinstance(node, (list, tuple)):
        result = type(node)(_copy_spec_tree(item, memo) for item in node)
    elif isinstance(node, dict):
        result = {key: _copy_spec_tree(value, memo) for key, value in node.items()}
    elif isinstance(node, functools.partial):
        args = _copy_spec_tree(node.args, memo)
        keywords = _copy_spec_tree(node.keywords, memo)
        result = type(node)(node.func, *args, **keywords)
        result.__dict__.update(_copy_spec_tree(vars(node), memo))
    elif dataclasses.is_dataclass(node) and not isinstance(node, type):
        result = copy.copy(node)
        memo[id(node)] = result
        for spec_field in dataclasses.fields(node):
            object.__setattr__(result, spec_field.name, _copy_spec_tree(getattr(node, spec_field.name), memo))
        return result
    else:
        return node
    memo[id(node)] = result
    return result


def _use_downcasting_norm(submodules: object, slot: str, expected: type, *, reference_switch: bool = True) -> None:
    """Replace the ``expected`` norm in ``submodules.<slot>`` by its downcasting version (in place)."""
    norm = getattr(submodules, slot)
    if norm is not expected:
        raise TypeError(f"ERNIE 5.5 expects {expected.__name__} in the {slot} slot, got {norm!r}.")
    setattr(submodules, slot, downcasting_norm(norm, reference_switch=reference_switch))


def _use_ernie55_rmsnorm(submodules: object, slot: str, expected: type) -> None:
    """Replace the ``expected`` norm in ``submodules.<slot>`` by :func:`ernie55_rmsnorm` around it (in place)."""
    norm = getattr(submodules, slot)
    if norm is not expected:
        raise TypeError(f"ERNIE 5.5 expects {expected.__name__} in the {slot} slot, got {norm!r}.")
    setattr(submodules, slot, ernie55_rmsnorm(norm))


def _use_ernie55_compressor_norm(compressor: object, expected: type) -> None:
    """Give a private copy of Megatron-Core's ``Compressor`` spec the :func:`ernie55_rmsnorm` norm (in place)."""
    if isinstance(compressor, functools.partial):
        submodules = compressor.keywords.get("submodules")
    elif isinstance(compressor, ModuleSpec):
        submodules = compressor.submodules
    else:
        submodules = None
    if submodules is None or not hasattr(submodules, "norm"):
        raise TypeError(f"ERNIE 5.5 expects a Megatron-Core Compressor spec with a norm slot, got {compressor!r}.")
    _use_ernie55_rmsnorm(submodules, "norm", expected)


def _ernie55_hca_layer_spec(mcore_layer: ModuleSpec) -> ModuleSpec:
    """Return a copy of a Megatron-Core DSv4 CSA layer spec turned into the ERNIE 5.5 H layer.

    The layer keeps Megatron-Core's ``TransformerLayer``, input norm, projections, ``Compressor``
    spec and bias-dropout-add. The attention becomes :class:`Ernie55HCASelfAttention` with a
    ``TELinear`` output gate, the core becomes :class:`Ernie55HCACoreAttention` without an indexer,
    and the q/kv latent norms are explicit ``TENorm`` modules (ERNIE 5.5 always normalizes the
    latents, whatever ``qk_layernorm`` says). Every RMSNorm (input, q/kv latents, compressor) is built
    through :func:`~megatron.bridge.models.ernie.modeling_ernie55.norm.ernie55_rmsnorm`, which picks the
    reference arithmetic under ``ernie_reference_rmsnorm``.
    """
    from megatron.core.extensions.transformer_engine import TELinear, TENorm

    layer = _copy_spec_tree(mcore_layer, {})
    attention = layer.submodules.self_attention
    mcore_submodules = attention.submodules
    mcore_core = mcore_submodules.core_attention
    if isinstance(mcore_core, functools.partial):
        core_submodules = mcore_core.keywords["submodules"]
    elif isinstance(mcore_core, ModuleSpec):
        core_submodules = mcore_core.submodules
    else:
        raise TypeError(f"Unexpected Megatron-Core CSA core_attention spec: {mcore_core!r}.")
    attention.module = Ernie55HCASelfAttention
    attention.submodules = Ernie55HCASelfAttentionSubmodules(
        **{
            spec_field.name: getattr(mcore_submodules, spec_field.name)
            for spec_field in dataclasses.fields(mcore_submodules)
        },
        linear_output_gate=TELinear,
    )
    # Megatron-Core's core builder accepts a partial (as well as a ModuleSpec).
    _use_ernie55_compressor_norm(core_submodules.compressor, TENorm)
    attention.submodules.core_attention = functools.partial(
        Ernie55HCACoreAttention,
        submodules=CompressedSparseAttentionSubmodules(compressor=core_submodules.compressor, indexer=None),
    )
    attention.submodules.q_layernorm = ernie55_rmsnorm(TENorm)
    attention.submodules.kv_layernorm = ernie55_rmsnorm(TENorm)
    _use_downcasting_norm(layer.submodules, "input_layernorm", TENorm)
    return layer


def _use_unfused_dense_mlp(mlp_layer: ModuleSpec, *, norm: type, linear_fc1: type) -> None:
    """Give the dense ``-`` layer a standalone downcasting ``pre_mlp_layernorm`` and a plain fc1 (in place)."""
    mlp = mlp_layer.submodules.mlp
    if not (
        isinstance(mlp, functools.partial)
        and mlp.func == MLP.as_mlp_submodule
        and isinstance(mlp.keywords.get("submodules"), MLPSubmodules)
    ):
        raise TypeError(f"ERNIE 5.5 expects MLP.as_mlp_submodule in the hybrid mlp_layer spec, got {mlp!r}.")
    mlp_layer.submodules.pre_mlp_layernorm = downcasting_norm(norm)
    mlp.keywords["submodules"].linear_fc1 = linear_fc1


def _use_ernie_moe(moe_layer: ModuleSpec) -> None:
    """Swap :class:`Ernie55MoELayer` and :class:`Ernie55SplitFeatureRouter` into the ``E`` layer (in place)."""
    moe = moe_layer.submodules.mlp
    if not (isinstance(moe, functools.partial) and moe.func is MoELayer):
        raise TypeError(f"ERNIE 5.5 expects an MoELayer in the hybrid moe_layer spec, got {moe!r}.")
    moe_submodules = moe.keywords.get("submodules")
    if not (isinstance(moe_submodules, MoESubmodules) and moe_submodules.router is TopKRouter):
        raise TypeError(f"ERNIE 5.5 expects TopKRouter in the hybrid MoE spec, got {moe_submodules!r}.")
    moe_layer.submodules.mlp = functools.partial(
        Ernie55MoELayer,
        *moe.args,
        **{**moe.keywords, "submodules": dataclasses.replace(moe_submodules, router=Ernie55SplitFeatureRouter)},
    )


def ernie55_hybrid_stack_spec(config: object | None = None) -> ModuleSpec:
    """Return a private copy of Megatron-Core's ``hybrid_stack_spec`` with the ERNIE 5.5 layers.

    Megatron-Core's shared spec is never mutated. Changes:

    * ``module``: :class:`~megatron.bridge.models.ernie.modeling_ernie55.hybrid_stack.Ernie55HybridStack`,
      the decoder stack that carries fp32 residual streams with hyper-connections.
    * Downcasting norms (:func:`~megatron.bridge.models.ernie.modeling_ernie55.norm.downcasting_norm`
      around ``TENorm``) in every slot that can receive an fp32 stream: the ``input_layernorm`` of
      ``D`` and ``H`` layers, the ``pre_mlp_layernorm`` of ``-`` and ``E`` layers, and the MTP
      ``hnorm`` (``mtp_block_spec``). They round the fp32 layer input to bf16 once, as the reference
      model's norms do; with bf16 inputs they are plain ``TENorm`` modules.
    * The D and H q/kv latent norms and the H compressor's norm hold
      :func:`~megatron.bridge.models.ernie.modeling_ernie55.norm.ernie55_rmsnorm` around ``TENorm``.
      With ``ernie_reference_rmsnorm`` these builders and the downcasting ones of the layers build
      :class:`~megatron.bridge.models.ernie.modeling_ernie55.norm.Ernie55RMSNorm`, the reference
      implementation's eager arithmetic, instead of ``TENorm``; the decoder's final norm follows the
      same switch in :class:`~megatron.bridge.models.ernie.modeling_ernie55.hybrid_stack.Ernie55HybridStack`.
      The MTP block's ``enorm``, ``hnorm`` and final ``layer_norm`` stay ``TENorm``, as in the reference,
      which builds them from Megatron-Core's Transformer Engine MTP spec.
    * ``dsa_layer`` (``D``):

      * ``self_attention.module`` becomes :class:`Ernie55LatentMQASelfAttention`;
      * ``core_attention`` becomes :class:`Ernie55LatentMQACoreAttention`, reusing the original
        ``DSAIndexer`` spec and adding a ``TELinear`` output gate;
      * ``q_layernorm`` and ``kv_layernorm`` become ``TENorm`` (through ``ernie55_rmsnorm_spec``).
        Explicit latent norms keep the q/kv RMSNorms standalone (Megatron-Core's QK-norm resolver
        would otherwise fuse them into the up-projections, which AbsorbedMLA bypasses), whatever
        the stack's attention variant.

    * ``csa_layer``, ``csa_qk_layernorm_layer`` and, on Megatron-Core dev, ``hca_layer`` (``H``):
      every slot an ``H`` layer can be built from gets its own copy of Megatron-Core's
      ``csa_qk_layernorm_layer`` with :class:`Ernie55HCASelfAttention`, an
      :class:`Ernie55HCACoreAttention` core without an indexer, a ``TELinear`` output gate and
      ``TENorm`` latent and compressor norms (through ``ernie55_rmsnorm``). ``C`` and ``W`` layers
      built from these slots raise, because ERNIE 5.5 has none. Builders should still call
      :func:`~megatron.bridge.models.ernie.modeling_ernie55.check_ernie55_attention_layers` on the
      built model (:class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel`
      does), in case a future Megatron-Core adds a slot this function does not set.
    * ``mlp_layer`` (dense ``-``): a standalone ``pre_mlp_layernorm`` and a plain
      ``TEColumnParallelLinear`` fc1 instead of the fused ``TELayerNormColumnParallelLinear``. This
      matches the reference's unfused dense layers and its parameter names
      (``pre_mlp_layernorm.weight``, ``mlp.linear_fc1.weight``, ``mlp.linear_fc2.weight``), and gives
      the norm its own slot like the ``E`` layer.
    * ``moe_layer`` (``E``): ``MoELayer`` becomes :class:`Ernie55MoELayer` and
      ``MoESubmodules.router`` becomes :class:`Ernie55SplitFeatureRouter`; the grouped experts and the
      shared expert stay Megatron-Core's.

    The MLP activation (SwiGLU or SiTU-GLU) comes from the config, not the spec. The nested MTP
    stack reuses these slots, so MTP D, H, dense and MoE layers get the same modules; the factory
    raises ``TypeError`` if Megatron-Core's spec ever provides separate MTP layer slots
    (``mtp_stack_submodules``). Megatron-Core builds the MTP stack itself as a plain
    ``HybridStack`` (its streams stay bf16);
    :class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel` gives it the
    ERNIE hyper-connection arithmetic.

    Args:
        config: Model config (flat or nested); accepted so builders can call the factory with
            their config. The spec does not depend on it.

    Returns:
        A new ``Ernie55HybridStack`` module spec.

    Raises:
        TypeError: If Megatron-Core's ``hybrid_stack_spec`` no longer has the layout this factory edits.
    """
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear, TENorm
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

    del config
    spec = _copy_spec_tree(hybrid_stack_spec, {})
    mtp_slots = getattr(spec.submodules, "mtp_stack_submodules", None)
    if mtp_slots is not None:
        raise TypeError(
            "ERNIE 5.5 expects the MTP stack to reuse the decoder's layer slots, but Megatron-Core's "
            f"hybrid_stack_spec sets mtp_stack_submodules ({type(mtp_slots).__name__}); its MTP layers "
            "would not get the ERNIE modules."
        )
    attention = spec.submodules.dsa_layer.submodules.self_attention
    attention.module = Ernie55LatentMQASelfAttention
    indexer = attention.submodules.core_attention.submodules.indexer
    attention.submodules.core_attention = ModuleSpec(
        module=Ernie55LatentMQACoreAttention,
        submodules=Ernie55LatentMQACoreSubmodules(indexer=indexer, linear_output_gate=TELinear),
    )
    # AbsorbedMLA builds its latent norms with build_module, which takes a ModuleSpec but not a partial.
    attention.submodules.q_layernorm = ernie55_rmsnorm_spec(TENorm)
    attention.submodules.kv_layernorm = ernie55_rmsnorm_spec(TENorm)
    _use_downcasting_norm(spec.submodules.dsa_layer.submodules, "input_layernorm", TENorm)

    slot_names = {spec_field.name for spec_field in dataclasses.fields(spec.submodules)}
    mcore_hca_layer = hybrid_stack_spec.submodules.csa_qk_layernorm_layer
    for slot in _HCA_LAYER_SLOTS:
        if slot in slot_names:
            setattr(spec.submodules, slot, _ernie55_hca_layer_spec(mcore_hca_layer))

    _use_unfused_dense_mlp(spec.submodules.mlp_layer, norm=TENorm, linear_fc1=TEColumnParallelLinear)
    _use_ernie_moe(spec.submodules.moe_layer)
    _use_downcasting_norm(spec.submodules.moe_layer.submodules, "pre_mlp_layernorm", TENorm)

    mtp_block_spec = spec.submodules.mtp_block_spec
    if mtp_block_spec is None:
        raise TypeError("ERNIE 5.5 expects Megatron-Core's hybrid_stack_spec to provide mtp_block_spec.")
    for mtp_layer in mtp_block_spec.submodules.layer_specs:
        # The reference builds its MTP block from Megatron-Core's TE MTP spec: enorm, hnorm and the final norm
        # stay TENorm whatever ernie_reference_rmsnorm says.
        _use_downcasting_norm(mtp_layer.submodules, "hnorm", TENorm, reference_switch=False)

    spec.module = Ernie55HybridStack
    return spec
