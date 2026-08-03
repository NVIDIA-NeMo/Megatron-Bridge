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
every layer, expressed in HF config as a `block_configs: list[dict]` sequence.
This bridge is thin: it inherits mapping infra + config translation from
`NemotronHBridge`, only swaps the HF top-level prefix (`model.*` instead of
`backbone.*`) and populates `moe_ffn_hidden_size_per_layer` /
`moe_router_topk_per_layer` on the Megatron provider so
`heterogeneous_block_specs` routing kicks in inside `hybrid_block.py`.
"""

from __future__ import annotations

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

        # MTP: mtp_block_configs is the per-sub-layer breakdown of a single MTP
        # depth. For Puzzle this is [attention, moe]. We forward the sub-pattern
        # via mtp_hybrid_override_pattern so HybridModelProvider.finalize()
        # concatenates it into the unified hybrid_layer_pattern.
        mtp_block_configs = list(getattr(hf_config, "mtp_block_configs", []) or [])
        mtp_pattern = _blocks_to_pattern(mtp_block_configs) if mtp_block_configs else None

        # Puzzle keeps expert count and shared expert size globally constant,
        # but we still populate the per-layer lists with -1 sentinels for the
        # non-MoE positions so the hetero validation path is exercised.
        total_experts = int(hf_config.n_routed_experts)
        shared_size = int(getattr(hf_config, "moe_shared_expert_intermediate_size", 0) or 0)

        moe_ffn_hidden_size_per_layer: list[int] = []
        moe_router_topk_per_layer: list[int] = []
        num_moe_experts_per_layer: list[int] = []
        moe_shared_expert_intermediate_size_per_layer: list[int] = []
        for b in block_configs:
            if _block_type(b) == "moe":
                moe_int = _block_attr(b, "moe_intermediate_size")
                topk = _block_attr(b, "num_experts_per_tok")
                if moe_int is None or topk is None:
                    raise ValueError(
                        f"MoE block config missing required keys: {b!r}"
                    )
                moe_ffn_hidden_size_per_layer.append(int(moe_int))
                moe_router_topk_per_layer.append(int(topk))
                num_moe_experts_per_layer.append(total_experts)
                moe_shared_expert_intermediate_size_per_layer.append(shared_size or -1)
            else:
                moe_ffn_hidden_size_per_layer.append(-1)
                moe_router_topk_per_layer.append(-1)
                num_moe_experts_per_layer.append(-1)
                moe_shared_expert_intermediate_size_per_layer.append(-1)

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

        provider.moe_ffn_hidden_size_per_layer = moe_ffn_hidden_size_per_layer
        provider.moe_router_topk_per_layer = moe_router_topk_per_layer
        provider.num_moe_experts_per_layer = num_moe_experts_per_layer
        provider.moe_shared_expert_intermediate_size_per_layer = (
            moe_shared_expert_intermediate_size_per_layer
        )

        # MTP HybridStack cannot use the main-block per-layer lists (its layer
        # numbers restart at 1 with pp_layer_offset=0). Instead, populate the
        # global scalar moe_ffn_hidden_size / moe_router_topk with the MTP MoE
        # values so MTP falls through to those scalars via the guard in
        # hybrid_block.py.
        if mtp_block_configs:
            mtp_moe = next(
                (b for b in mtp_block_configs if _block_type(b) == "moe"), None
            )
            if mtp_moe is not None:
                mtp_moe_int = _block_attr(mtp_moe, "moe_intermediate_size")
                mtp_topk = _block_attr(mtp_moe, "num_experts_per_tok")
                if mtp_moe_int is not None:
                    provider.moe_ffn_hidden_size = int(mtp_moe_int)
                if mtp_topk is not None:
                    provider.moe_router_topk = int(mtp_topk)

        provider.heterogeneous_block_specs = True
        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Emit HF-side `block_configs` and `mtp_block_configs` from per-layer lists."""
        hf_cfg = super().megatron_to_hf_config(provider)

        # Rebuild block_configs from per-layer lists + hybrid pattern.
        pattern = hf_cfg.pop("hybrid_override_pattern", None)
        if pattern is None:
            # NemotronHBridge.megatron_to_hf_config normally returns a cleaned
            # pattern; if it's missing we can't rebuild block_configs.
            return hf_cfg

        # `pattern` is already the main decoder pattern (no MTP suffix). Convert
        # symbols back to block types.
        symbol_to_type = {v: k for k, v in _BLOCK_TYPE_TO_SYMBOL.items()}
        main_types = [symbol_to_type[s] for s in pattern if s in symbol_to_type]

        moe_ffn_per_layer = provider.moe_ffn_hidden_size_per_layer or []
        moe_topk_per_layer = provider.moe_router_topk_per_layer or []

        block_configs: list[dict] = []
        for i, t in enumerate(main_types):
            entry: dict = {"block_type": t}
            if t == "moe":
                moe_int = moe_ffn_per_layer[i] if i < len(moe_ffn_per_layer) else -1
                topk = moe_topk_per_layer[i] if i < len(moe_topk_per_layer) else -1
                if moe_int != -1:
                    entry["moe_intermediate_size"] = int(moe_int)
                if topk != -1:
                    entry["num_experts_per_tok"] = int(topk)
            block_configs.append(entry)

        hf_cfg["block_configs"] = block_configs

        # MTP block: reconstruct from scalar moe_ffn_hidden_size / moe_router_topk.
        mtp_pattern = hf_cfg.pop("mtp_hybrid_override_pattern", None)
        if mtp_pattern:
            mtp_block_configs: list[dict] = []
            for s in mtp_pattern:
                if s not in symbol_to_type:
                    continue
                t = symbol_to_type[s]
                entry = {"block_type": t}
                if t == "moe":
                    if provider.moe_ffn_hidden_size is not None:
                        entry["moe_intermediate_size"] = int(provider.moe_ffn_hidden_size)
                    if provider.moe_router_topk is not None:
                        entry["num_experts_per_tok"] = int(provider.moe_router_topk)
                mtp_block_configs.append(entry)
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
