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

"""Bridge for NVIDIA Nemotron-H Puzzle heterogeneous MoE models.

The Puzzle architecture (e.g. `NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16`) is a
NemotronH hybrid backbone (mamba / attention / MoE blocks) where the MoE layers
have a *different* `moe_intermediate_size` and `num_experts_per_tok` on almost
every layer, expressed in HF config as `block_configs: list[dict]` (main block)
and `mtp_block_configs: list[dict]` (positions inside one MTP depth).

Both sides map 1:1 to Megatron-Core's sparse per-layer override interface:

- `block_configs`      -> `per_layer_config_overrides`      (length = num_layers)
- `mtp_block_configs`  -> `mtp_per_layer_config_overrides`  (length = mtp_pattern_length)

Non-MoE positions get a `None` entry (no override — the global provider values
apply). MoE positions carry the varying keys only (`moe_ffn_hidden_size`,
`moe_router_topk`); the globally-constant `num_moe_experts` and
`moe_shared_expert_intermediate_size` stay on the provider itself.
"""

from __future__ import annotations

from typing import Any

from megatron.core.models.hybrid.hybrid_model import HybridModel

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.nemotronh.nemotron_h_bridge import NemotronHBridge


# HF block_type -> hybrid layer pattern symbol used by Megatron HybridStack.
_BLOCK_TYPE_TO_SYMBOL = {
    "mamba": "M",
    "attention": "*",
    "moe": "E",
    "mlp": "-",
}


def _blocks_to_pattern(blocks) -> str:
    return "".join(_BLOCK_TYPE_TO_SYMBOL[_block_type(b)] for b in blocks)


def _block_type(b) -> str:
    """block_configs entries may arrive as dataclass instances or plain dicts."""
    return getattr(b, "block_type", None) or b["block_type"]


def _block_attr(b, name: str):
    if hasattr(b, name):
        return getattr(b, name)
    return b.get(name) if isinstance(b, dict) else None


def _block_to_override(b) -> dict[str, Any] | None:
    """Return the sparse override dict for one HF block config entry, or None.

    Only MoE entries carry per-position overrides in Puzzle configs. Non-MoE
    positions inherit the global provider config, which the resolver returns
    verbatim when the override is `None`.
    """
    if _block_type(b) != "moe":
        return None
    moe_int = _block_attr(b, "moe_intermediate_size")
    topk = _block_attr(b, "num_experts_per_tok")
    if moe_int is None or topk is None:
        raise ValueError(f"MoE block config missing required keys: {b!r}")
    return {
        "moe_ffn_hidden_size": int(moe_int),
        "moe_router_topk": int(topk),
    }


@MegatronModelBridge.register_bridge(
    source="NemotronHPuzzleForCausalLM",
    target=HybridModel,
    provider=HybridModelProvider,
    model_type="nemotron_h_puzzle",
)
class NemotronHPuzzleBridge(NemotronHBridge):
    """Bridge for NemotronH-Puzzle heterogeneous MoE hybrid models."""

    # Puzzle checkpoints use `model.*` as the HF top-level prefix (vs `backbone.*`
    # for vanilla NemotronH). See NemotronHBridge._apply_hf_prefix for how this
    # rewrites the shared mapping registry.
    HF_PREFIX = "model"

    @staticmethod
    def _hf_mtp_config(hf_config) -> tuple[int, "str | None"]:
        """Puzzle's HF config expresses MTP via `mtp_block_configs`, not `mtp_hybrid_override_pattern`."""
        mtp_num_layers = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)
        if mtp_num_layers < 0:
            raise ValueError("num_nextn_predict_layers must be non-negative.")
        if mtp_num_layers == 0:
            return 0, None

        blocks = list(getattr(hf_config, "mtp_block_configs", []) or [])
        if not blocks:
            # Fall back to the parent contract in case the config used the older
            # `mtp_hybrid_override_pattern` key.
            mtp_pattern = getattr(hf_config, "mtp_hybrid_override_pattern", None)
            if not mtp_pattern:
                raise ValueError(
                    "NemotronHPuzzle HF config with num_nextn_predict_layers > 0 must set either "
                    "`mtp_block_configs` (preferred) or `mtp_hybrid_override_pattern`."
                )
            return mtp_num_layers, mtp_pattern
        return mtp_num_layers, _blocks_to_pattern(blocks)

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> HybridModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        block_configs = list(getattr(hf_config, "block_configs", []) or [])
        if not block_configs:
            raise ValueError(
                "NemotronHPuzzle HF config is missing `block_configs`; cannot derive "
                "per-layer heterogeneous MoE settings."
            )

        num_layers = len(block_configs)
        main_pattern = _blocks_to_pattern(block_configs)

        mtp_block_configs = list(getattr(hf_config, "mtp_block_configs", []) or [])
        mtp_pattern = _blocks_to_pattern(mtp_block_configs) if mtp_block_configs else None

        total_experts = int(hf_config.n_routed_experts)
        shared_size = int(getattr(hf_config, "moe_shared_expert_intermediate_size", 0) or 0)

        # Sparse main-block overrides: one entry per layer, either a dict of
        # override keys or None. Length must equal num_layers — the mcore
        # validator enforces this in TransformerConfig.__post_init__.
        per_layer_config_overrides: list[dict[str, Any] | None] = [
            _block_to_override(b) for b in block_configs
        ]

        # Sparse MTP per-position overrides: one entry per position inside one
        # MTP depth. All MTP depths share these — see `mtp_pattern_length` on
        # TransformerConfig for the axis definition. Length must equal
        # mtp_pattern_length.
        mtp_per_layer_config_overrides: list[dict[str, Any] | None] | None = None
        mtp_pattern_length: int | None = None
        if mtp_block_configs:
            mtp_pattern_length = len(mtp_block_configs)
            mtp_per_layer_config_overrides = [
                _block_to_override(b) for b in mtp_block_configs
            ]

        # Overwrite scalar hybrid pattern derived by the parent from
        # hybrid_override_pattern (which Puzzle configs don't set) with the one
        # we just constructed from block_configs.
        provider.hybrid_layer_pattern = main_pattern
        provider.hybrid_override_pattern = None
        provider.num_layers = num_layers
        provider.mtp_hybrid_override_pattern = mtp_pattern

        provider.num_moe_experts = total_experts
        if shared_size:
            provider.moe_shared_expert_intermediate_size = shared_size

        provider.per_layer_config_overrides = per_layer_config_overrides
        provider.mtp_per_layer_config_overrides = mtp_per_layer_config_overrides
        provider.mtp_pattern_length = mtp_pattern_length

        # Validator flips this on when either override list is set, but
        # __post_init__ has already run on the parent's provider — set it here
        # explicitly so downstream builders route through get_config_for_layer /
        # get_config_for_mtp_layer.
        provider.heterogeneous_block_specs = True
        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Emit HF-side `block_configs` and `mtp_block_configs` from sparse per-layer overrides."""
        hf_cfg = super().megatron_to_hf_config(provider)

        # Rebuild block_configs from the main hybrid pattern + sparse overrides.
        pattern = hf_cfg.pop("hybrid_override_pattern", None)
        if pattern is None:
            # NemotronHBridge.megatron_to_hf_config normally returns a cleaned
            # pattern; if it's missing we can't rebuild block_configs.
            return hf_cfg

        symbol_to_type = {v: k for k, v in _BLOCK_TYPE_TO_SYMBOL.items()}
        main_types = [symbol_to_type[s] for s in pattern if s in symbol_to_type]

        per_layer_overrides = provider.per_layer_config_overrides or []
        block_configs = [
            _entry_from_override(t, per_layer_overrides[i] if i < len(per_layer_overrides) else None)
            for i, t in enumerate(main_types)
        ]
        hf_cfg["block_configs"] = block_configs

        # MTP block: reconstruct from sparse mtp_per_layer_config_overrides.
        mtp_pattern = hf_cfg.pop("mtp_hybrid_override_pattern", None)
        if mtp_pattern:
            mtp_overrides = provider.mtp_per_layer_config_overrides or []
            mtp_block_configs = [
                _entry_from_override(
                    symbol_to_type[s],
                    mtp_overrides[i] if i < len(mtp_overrides) else None,
                )
                for i, s in enumerate(mtp_pattern)
                if s in symbol_to_type
            ]
            hf_cfg["mtp_block_configs"] = mtp_block_configs

        # Puzzle-specific auto_map overrides the vanilla NemotronH values set
        # by the parent.
        hf_cfg["auto_map"] = {
            "AutoConfig": "configuration_nemotron_h_puzzle.NemotronHPuzzleConfig",
            "AutoModelForCausalLM": "modeling_nemotron_h_puzzle.NemotronHPuzzleForCausalLM",
        }
        hf_cfg["model_type"] = "nemotron_h_puzzle"
        # In Puzzle, `moe_intermediate_size` and `num_experts_per_tok` are strictly
        # blockwise members; NemotronHPuzzleConfig._delete_blockwise_members_from_global_config
        # strips them from the top level. Drop them here so the emitted config
        # round-trips through HF loading without stale globals.
        for key in ("moe_intermediate_size", "num_experts_per_tok"):
            hf_cfg.pop(key, None)
        # layers_block_type is derived from block_configs by NemotronHPuzzleConfig,
        # but emit it explicitly for readability / offline tooling.
        hf_cfg["layers_block_type"] = [entry["block_type"] for entry in block_configs]
        if mtp_pattern:
            hf_cfg["mtp_layers_block_type"] = [entry["block_type"] for entry in hf_cfg["mtp_block_configs"]]
        return hf_cfg


def _entry_from_override(block_type: str, override: dict[str, Any] | None) -> dict:
    """Rebuild one HF block_configs entry from a Megatron per-layer override dict."""
    entry: dict = {"block_type": block_type}
    if block_type != "moe" or not override:
        return entry
    moe_int = override.get("moe_ffn_hidden_size")
    topk = override.get("moe_router_topk")
    if moe_int is not None:
        entry["moe_intermediate_size"] = int(moe_int)
    if topk is not None:
        entry["num_experts_per_tok"] = int(topk)
    return entry
