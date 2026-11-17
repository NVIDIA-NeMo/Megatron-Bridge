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

"""ERNIE 5.5 shared-decoder MTP (SD-MTP): the MTP body reuses the last decoder layers' parameters.

With Hugging Face ``mtp_shared_last_layer`` (``ernie_mtp_shared_last_layer``), every MTP depth runs
its own D + MoE layer pair, but that pair's Parameters are the decoder's last D + MoE pair: the
attention (indexer, sink, VHA and gate included), both hyper-connection modules, the router
weights and scales, the routed experts and the shared expert. In a hybrid model each ERNIE layer is
two symbols, so the MTP stack ``"DE"`` of every depth is aliased onto the last two decoder layers.

Owned by the MTP depth and not shared: ``enorm``, ``hnorm``, ``e_proj``, ``h_proj``, its
``hc_head_*`` contraction and final norm, and all buffers (each MTP router keeps its own
``expert_bias`` and token counts, and updates them from its own routing).

Each shared tensor is one ``nn.Parameter`` used twice in the forward, so the gradients of the LM
loss and the MTP loss add up in it. ``named_parameters()`` and ``parameters()`` return it once,
under the decoder name, so DDP buckets, the optimizer and the gradient norm see it once, and
:class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel` writes it once to
checkpoints.

The aliasing must happen before the model is wrapped for training and before any checkpoint
load, and the decoder's last layers must be on the MTP's pipeline stage (Megatron-Core runs mHC MTP
on the last stage).

Sharing is plain ``nn.Parameter`` sharing, so anything that later re-creates parameters module by
module splits each shared tensor into two silently diverging copies: materializing a meta-device
model (``init_model_with_meta_device``), ``load_state_dict(assign=True)``, and the per-module
sharding of Megatron-FSDP or torch FSDP2. :func:`share_mtp_body_with_last_layer` rejects meta-device
models, :class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel` rejects
``assign=True`` loads, :func:`validate_ernie55_mtp_sharing` rejects the FSDP and meta-device
settings for the code that knows them (the training setup), and :func:`check_ernie55_mtp_aliases`
verifies the sharing after anything that may have replaced parameters.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from megatron.core.models.hybrid.hybrid_block import HybridStack


__all__ = ["check_ernie55_mtp_aliases", "share_mtp_body_with_last_layer", "validate_ernie55_mtp_sharing"]

# Parameter attributes Megatron-Core's DDP, optimizers and checkpointing read; aliased pairs must agree.
_PARAMETER_ATTRIBUTES = (
    "allreduce",
    "tensor_model_parallel",
    "partition_dim",
    "partition_stride",
    "sequence_parallel",
    "keep_in_fp32",
    "is_embedding_or_output_parameter",
    "grad_norm_group",
)


def _symbols(stack: HybridStack) -> str:
    return "".join(stack.layer_type_list)


def _set_parameter(root: torch.nn.Module, name: str, parameter: torch.nn.Parameter) -> None:
    owner_name, _, leaf = name.rpartition(".")
    owner = root.get_submodule(owner_name) if owner_name else root
    if leaf not in owner._parameters:
        raise RuntimeError(f"ERNIE 5.5 SD-MTP: {name!r} is not a registered parameter.")
    owner._parameters[leaf] = parameter


def share_mtp_body_with_last_layer(model: torch.nn.Module) -> dict[str, str]:
    """Alias every MTP depth's inner layers' Parameters to the decoder's last layers (in place).

    Args:
        model: A Megatron-Core ``HybridModel`` stage that builds MTP.

    Returns:
        ``{mtp parameter name: decoder parameter name}``, relative to ``model``, one entry per
        aliased MTP parameter.

    Raises:
        RuntimeError: If the model has no MTP, if the decoder's last layers are not on this
            stage, if a depth's layer symbols, parameter names, shapes, dtypes or attributes differ
            from the decoder layers they alias, if a configuration feature contradicts sharing
            (``mtp_detach_heads``, ``freeze_base_model_for_mtp``), or if the model is on the meta
            device or built for meta-device initialization (materializing it re-creates the
            parameters module by module, which splits every shared tensor).
    """
    from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern

    def fail(message: str) -> None:
        raise RuntimeError(f"ERNIE 5.5 SD-MTP: {message}")

    config = model.config
    if getattr(config, "mtp_detach_heads", False):
        fail("mtp_detach_heads detaches the MTP from the decoder, whose layers the MTP body shares.")
    if getattr(config, "freeze_base_model_for_mtp", False):
        fail("freeze_base_model_for_mtp freezes the decoder layers the MTP body shares.")
    if getattr(config, "init_model_with_meta_device", False) or any(
        param.device.type == "meta" for param in model.parameters()
    ):
        fail(
            "the model is on the meta device (init_model_with_meta_device); materializing it re-creates the "
            "parameters module by module and would split every shared tensor in two."
        )
    mtp = getattr(model, "mtp", None)
    if not getattr(model, "mtp_process", False) or mtp is None or len(mtp.layers) == 0:
        fail("the model builds no MTP layer on this stage.")
    decoder = model.decoder
    main_pattern = parse_hybrid_pattern(model.hybrid_layer_pattern).main_pattern or ""
    num_decoder_layers = len(main_pattern.replace("|", ""))
    decoder_layers = list(decoder.layers)
    if len(decoder_layers) != len(decoder.layer_type_list):
        fail("the decoder groups its layers (moe_shortcut_connection); sharing needs one module per layer.")
    if not decoder_layers or decoder_layers[-1].layer_number != num_decoder_layers:
        fail(
            "the decoder's last layer must be on the MTP's pipeline stage "
            f"(last layer here: {decoder_layers[-1].layer_number if decoder_layers else None}, "
            f"model: {num_decoder_layers})."
        )

    decoder_index = {id(layer): index for index, layer in enumerate(decoder_layers)}
    aliases: dict[str, str] = {}
    for depth, mtp_layer in enumerate(mtp.layers):
        body = getattr(mtp_layer, "mtp_model_layer", None)
        if not isinstance(body, HybridStack):
            fail(f"MTP depth {depth} has no hybrid inner stack.")
        body_layers = list(body.layers)
        count = len(body_layers)
        if count == 0 or count > len(decoder_layers):
            fail(f"MTP depth {depth} has {count} layers, the decoder stage {len(decoder_layers)}.")
        body_symbols = _symbols(body)
        decoder_symbols = _symbols(decoder)[-count:]
        if len(body_layers) != len(body_symbols) or body_symbols != decoder_symbols:
            fail(
                f"MTP depth {depth} layers {body_symbols!r} differ from the decoder's last layers {decoder_symbols!r}."
            )
        for offset, (body_layer, source) in enumerate(zip(body_layers, decoder_layers[-count:])):
            source_params = dict(source.named_parameters())
            target_params = dict(body_layer.named_parameters())
            if source_params.keys() != target_params.keys():
                missing = sorted(source_params.keys() - target_params.keys())
                extra = sorted(target_params.keys() - source_params.keys())
                fail(f"MTP depth {depth} layer {offset} parameters differ: missing={missing[:8]}, extra={extra[:8]}.")
            mtp_prefix = f"mtp.layers.{depth}.mtp_model_layer.layers.{offset}."
            decoder_prefix = f"decoder.layers.{decoder_index[id(source)]}."
            for name, parameter in source_params.items():
                target = target_params[name]
                if (target.shape, target.dtype, target.requires_grad) != (
                    parameter.shape,
                    parameter.dtype,
                    parameter.requires_grad,
                ):
                    fail(
                        f"{mtp_prefix}{name} is {tuple(target.shape)} {target.dtype} "
                        f"(requires_grad={target.requires_grad}), the decoder's {tuple(parameter.shape)} "
                        f"{parameter.dtype} (requires_grad={parameter.requires_grad})."
                    )
                for attribute in _PARAMETER_ATTRIBUTES:
                    if getattr(target, attribute, None) != getattr(parameter, attribute, None):
                        fail(f"{mtp_prefix}{name} and its decoder parameter differ in {attribute!r}.")
                _set_parameter(body_layer, name, parameter)
                aliases[mtp_prefix + name] = decoder_prefix + name
            for name, parameter in source_params.items():
                if body_layer.get_parameter(name) is not parameter:
                    fail(f"{mtp_prefix}{name} did not take the decoder's parameter.")
    return aliases


def check_ernie55_mtp_aliases(models: torch.nn.Module | Iterable[torch.nn.Module]) -> int:
    """Verify that every SD-MTP alias is still the decoder's own ``nn.Parameter``.

    Cheap (one lookup per alias). Call it after anything that may replace parameters module by
    module: pre-wrap hooks, wrapping, and checkpoint loads. It finds every module under ``models``
    that records SD-MTP aliases (``ernie55_mtp_aliases``, set by
    :class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel`), so wrapped
    models and lists of model chunks work too.

    Args:
        models: A model, a wrapped model or an iterable of model chunks.

    Returns:
        The number of aliases checked (0 for models without SD-MTP).

    Raises:
        RuntimeError: If an MTP parameter is no longer the decoder parameter it aliases, or if a wrapper or
            adapter replaced a module behind an alias so the parameter cannot be resolved by name.
    """
    roots = [models] if isinstance(models, torch.nn.Module) else list(models)
    checked, broken = 0, []
    for root in roots:
        for module in root.modules():
            aliases = getattr(module, "ernie55_mtp_aliases", None)
            if not isinstance(aliases, dict):
                continue
            for alias, source in aliases.items():
                checked += 1
                try:
                    shared = module.get_parameter(alias) is module.get_parameter(source)
                except AttributeError as error:
                    raise RuntimeError(
                        f"ERNIE 5.5 SD-MTP: the recorded alias {alias} -> {source} cannot be resolved ({error}): a "
                        "wrapper or adapter replaced a module on one of the two paths, so the alias no longer names "
                        "the shared parameter. PEFT methods must leave the paths in "
                        "Ernie55HybridModel.peft_excluded_module_paths() unchanged."
                    ) from error
                if not shared:
                    broken.append(f"{alias} -> {source}")
    if broken:
        raise RuntimeError(
            f"ERNIE 5.5 SD-MTP: {len(broken)} MTP parameters no longer share the decoder's parameters "
            f"(e.g. {broken[:3]}); something re-created them module by module (meta-device materialization, "
            "load_state_dict(assign=True), FSDP), so the two copies would train separately."
        )
    return checked


def validate_ernie55_mtp_sharing(config: object, *, use_megatron_fsdp: bool, use_torch_fsdp2: bool) -> None:
    """Raise ``ValueError`` if the training setup cannot keep the SD-MTP parameters shared.

    Megatron-FSDP and torch FSDP2 shard and re-create parameters module by module, and meta-device
    initialization re-creates them when it materializes the model; each splits the shared tensors.
    The model config does not carry the FSDP switches, so the code that builds the training setup
    calls this with them. Configs without ``ernie_mtp_shared_last_layer`` pass.

    Args:
        config: Transformer config.
        use_megatron_fsdp: Whether the model will be wrapped with Megatron-FSDP.
        use_torch_fsdp2: Whether the model will be wrapped with torch FSDP2.

    Raises:
        ValueError: If ``ernie_mtp_shared_last_layer`` is set together with one of them or with
            ``init_model_with_meta_device``.
    """
    if not getattr(config, "ernie_mtp_shared_last_layer", False):
        return
    conflicts = [
        name
        for name, enabled in (
            ("use_megatron_fsdp", use_megatron_fsdp),
            ("use_torch_fsdp2", use_torch_fsdp2),
            ("init_model_with_meta_device", getattr(config, "init_model_with_meta_device", False)),
        )
        if enabled
    ]
    if conflicts:
        raise ValueError(
            f"ERNIE 5.5 SD-MTP: ernie_mtp_shared_last_layer cannot be combined with {', '.join(conflicts)}; "
            "they re-create parameters module by module and would split the shared MTP body from the decoder."
        )
