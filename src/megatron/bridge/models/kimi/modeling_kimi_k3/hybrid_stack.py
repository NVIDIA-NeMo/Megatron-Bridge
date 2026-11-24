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

"""Split Kimi blocks into Hybrid entries without changing Core implementations."""

from copy import copy, deepcopy
from functools import partial
from typing import Any

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.tensor_parallel import checkpoint
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import make_viewless_tensor
from torch import nn

from megatron.bridge.models.kimi.kimi_k3_layers import KimiK3Attention, _mark_tp_replicated
from megatron.bridge.models.kimi.kimi_k3_ops import KimiRMSNorm, attn_res_aggregate
from megatron.bridge.models.kimi.kimi_k3_pipeline import bank_num_rows, pack_stage_boundary, unpack_stage_boundary
from megatron.bridge.models.kimi.kimi_k3_spec import _with_situ_activation


def _residual_parameters(module: nn.Module, config: Any, branch: str) -> None:
    device = torch.cuda.current_device()
    norm = KimiRMSNorm(config.hidden_size, config.layernorm_epsilon, device=device, dtype=config.params_dtype)
    projection = nn.Linear(config.hidden_size, 1, bias=False, device=device, dtype=config.params_dtype)
    _mark_tp_replicated(norm, reduction="sum")
    _mark_tp_replicated(projection, reduction="sum")
    module.add_module(f"{branch}_res_norm", norm)
    module.add_module(f"{branch}_res_proj", projection)


class KimiK3HybridEntry(MegatronModule):
    """One attention or FFN entry, sharing the existing Kimi component implementations."""

    def __init__(self, config: Any, submodules: Any, *, layer_number: int, attention: bool, pg_collection: Any):
        super().__init__(config)
        self.layer_number = layer_number
        self.attention = attention
        self.source_index = (layer_number - 1) // 2
        branch = "self_attention" if attention else "mlp"
        _residual_parameters(self, config, branch)
        norm_name = "input_layernorm" if attention else "pre_mlp_layernorm"
        self.add_module(
            norm_name,
            build_module(
                getattr(submodules, norm_name),
                config=config,
                hidden_size=config.hidden_size,
                eps=config.layernorm_epsilon,
            ),
        )
        if attention:
            self.self_attention = build_module(
                submodules.self_attention,
                config=config,
                layer_number=self.source_index + 1,
                pg_collection=pg_collection,
            )
        else:
            self.mlp = submodules.mlp(config=config, pg_collection=pg_collection, is_mtp_layer=False)
            if hasattr(self.mlp, "set_layer_number"):
                self.mlp.set_layer_number(self.source_index + 1)

    def forward(
        self,
        prefix: torch.Tensor,
        bank: torch.Tensor,
        *,
        attention_mask: Any = None,
        packed_seq_params: Any = None,
        padding_mask: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the updated partial and snapshot tensors; never mutate depth state."""
        if self.attention:
            inputs = (
                attn_res_aggregate(
                    prefix, bank, self.self_attention_res_proj, self.self_attention_res_norm, self.input_layernorm
                )
                if bank.shape[-2]
                else self.input_layernorm(prefix)
            )
            block_start = self.source_index % self.config.kimi_attn_res_block_size == 0
            if block_start:
                bank = torch.cat((bank, prefix.unsqueeze(-2)), dim=-2)
            output, bias = self.self_attention(
                inputs, attention_mask=attention_mask, packed_seq_params=packed_seq_params
            )
            if bias is not None:
                output = output + bias
            prefix = output if block_start else prefix + output
        else:
            inputs = attn_res_aggregate(prefix, bank, self.mlp_res_proj, self.mlp_res_norm, self.pre_mlp_layernorm)
            output, bias = self.mlp(inputs, padding_mask=padding_mask)
            if bias is not None:
                output = output + bias
            prefix = prefix + output
        return prefix, bank


class KimiK3HybridStack(HybridStack):
    """Bridge-owned depth-state composition using Core's Hybrid stack/checkpoint interface.

    Core owns embeddings, output/loss, pipeline allocation and sharded-state
    traversal. This stack reuses Bridge's Kimi attention, MoE, activation and
    residual operations. Its two tensor outputs make checkpoint replay independent
    of mutable lists or state captured from a previous microbatch.
    """

    def __init__(
        self,
        config: Any,
        *,
        pattern: str,
        layer_config_list: Any,
        pp_layer_offset: int = 0,
        pre_process: bool = True,
        post_process: bool = True,
        post_layer_norm: bool = True,
        pg_collection: Any,
        **kwargs: Any,
    ):
        # The generic constructor creates stock Core KDA/AttnRes modules. Build
        # the Kimi entries directly, retaining the inherited sharded-state method.
        MegatronModule.__init__(self, config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.post_layer_norm = post_layer_norm
        self.pp_layer_offset = pp_layer_offset
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.input_tensor = None
        self._has_custom_layers = True
        self.num_layers_per_pipeline_rank = len(layer_config_list)
        if pp_layer_offset % 2 or self.num_layers_per_pipeline_rank % 2:
            raise ValueError("Kimi K3 pipeline stages must contain complete source blocks.")
        entries = pattern.replace("|", "")
        source = copy(config)
        source.num_layers = len(entries) // 2
        source.kimi_kda_layers = tuple(i + 1 for i, entry in enumerate(entries[::2]) if entry == "K")
        source.moe_layer_freq = [int(entry == "E") for entry in entries[1::2]]
        source.pipeline_model_parallel_size = 1
        source.num_layers_in_first_pipeline_stage = None
        source.num_layers_in_last_pipeline_stage = None
        source.pipeline_model_parallel_layout = None
        source.is_hybrid_model = False
        # Specification construction is weight-free; stage ownership remains
        # the allocation supplied by HybridModel, including uneven stages.
        specs = [deepcopy(spec) for spec in get_gpt_decoder_layer_specs(source, use_transformer_engine=True)]
        for index, spec in enumerate(specs):
            spec.submodules.self_attention = ModuleSpec(module=KimiK3Attention)
            spec.submodules.input_layernorm = TENorm
            spec.submodules.pre_mlp_layernorm = TENorm
            spec.submodules.mlp = _with_situ_activation(spec.submodules.mlp)
            if not source.moe_layer_freq[index]:
                spec.submodules.mlp.keywords["submodules"].linear_fc1 = TEColumnParallelLinear
        self.layers = nn.ModuleList(
            [
                KimiK3HybridEntry(
                    source,
                    specs[index // 2].submodules,
                    layer_number=index + 1,
                    attention=index % 2 == 0,
                    pg_collection=pg_collection,
                )
                for index in range(pp_layer_offset, pp_layer_offset + self.num_layers_per_pipeline_rank)
            ]
        )
        if post_process:
            _residual_parameters(self, source, "output_attn")
            self.final_norm = TENorm(config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon)

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Receive the packed partial/snapshot payload from the preceding stage."""
        self.input_tensor = input_tensor

    def _run_entries(
        self, start: int, end: int, prefix: torch.Tensor, bank: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers[start:end]:
            prefix, bank = layer(prefix, bank, **kwargs)
        return prefix, bank

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        attention_mask: Any = None,
        inference_context: Any = None,
        rotary_pos_emb: Any = None,
        *,
        packed_seq_params: Any = None,
        padding_mask: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run entries and pass all differentiable depth state through recomputation."""
        if inference_context is not None:
            raise NotImplementedError(
                "Kimi K3 Hybrid KV-cache inference is not implemented; use the retained GPT path."
            )
        if not self.pre_process:
            hidden_states = self.input_tensor
        if hidden_states is None:
            raise ValueError("Kimi K3 Hybrid stack requires an input activation.")
        if self.pp_layer_offset:
            prefix, bank = unpack_stage_boundary(
                hidden_states,
                self.config.hidden_size,
                bank_num_rows(self.pp_layer_offset // 2, self.config.kimi_attn_res_block_size),
            )
        else:
            prefix = hidden_states
            bank = prefix.new_empty(*prefix.shape[:-1], 0, prefix.shape[-1])
        prefix = make_viewless_tensor(inp=prefix, requires_grad=prefix.requires_grad, keep_graph=True)
        layer_kwargs = dict(
            attention_mask=attention_mask, packed_seq_params=packed_seq_params, padding_mask=padding_mask
        )
        full = self.training and self.config.recompute_granularity == "full"
        if full and self.config.recompute_method not in ("uniform", "block"):
            raise ValueError("Kimi K3 full recompute requires uniform or block method.")
        chunk = self.config.recompute_num_layers if full else len(self.layers)
        if not chunk or chunk < 1:
            raise ValueError("Kimi K3 recompute_num_layers must be positive.")
        start = 0
        while start < len(self.layers):
            end = (
                min(start + (chunk if self.config.recompute_method != "block" else 1), len(self.layers))
                if full
                else len(self.layers)
            )
            run = partial(self._run_entries, start, end, **layer_kwargs)
            recompute = full and (self.config.recompute_method == "uniform" or start < chunk)
            if recompute and (prefix.requires_grad or bank.requires_grad):
                # Distributed saved activations resize their first tensor. Keep
                # that storage separate from snapshots retained across chunks.
                saved_prefix = prefix.clone() if self.config.distribute_saved_activations else prefix
                prefix, bank = checkpoint(run, self.config.distribute_saved_activations, saved_prefix, bank)
            else:
                prefix, bank = run(prefix, bank)
            start = end
        if not self.post_process:
            return pack_stage_boundary(prefix, bank)
        prefix = attn_res_aggregate(prefix, bank, self.output_attn_res_proj, self.output_attn_res_norm, nn.Identity())
        return self.final_norm(prefix)
