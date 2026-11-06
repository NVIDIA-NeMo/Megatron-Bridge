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
"""Rollout routing replay: make the model's MoE routers pick the experts the generation backend picked.

With ``moe_enable_routing_replay`` every Megatron ``TopKRouter`` owns a ``RouterReplay``. Replay fixes
only the expert indices; the routing probabilities are recomputed from the current router logits at
those experts, so the router still receives gradients. The instances register themselves in a
process-wide list in construction order, so each engine claims the ones created during its own build
(layer order, MTP routers excluded from replay) and releases them on close.

Per microbatch the engine arms the routers from ``PreparedMicrobatch.routed_experts`` before the
forward (``REPLAY_FORWARD``) and switches them to ``REPLAY_BACKWARD`` afterwards, so an activation
recompute of the MoE block in the backward pops the same indices again; the call clears everything
at the end.
"""

from __future__ import annotations

import torch
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

from megatron.bridge.engine.datum import PreparedMicrobatch


class RouterReplayState:
    """The ``RouterReplay`` instances one engine built, and the per-call arming of them."""

    def __init__(self, owned: list[RouterReplay], *, sequence_parallel: bool = False) -> None:
        self.owned = list(owned)
        self.routers = [router for router in owned if not router.is_mtp_layer]
        self.sequence_parallel = sequence_parallel

    @staticmethod
    def mark() -> int:
        """Size of the process-wide registry; take it before building a model."""
        return len(RouterReplay.global_router_replay_instances)

    @classmethod
    def claim(cls, mark: int, *, sequence_parallel: bool = False) -> RouterReplayState:
        """Own every instance registered since ``mark`` (the model built in between)."""
        return cls(RouterReplay.global_router_replay_instances[mark:], sequence_parallel=sequence_parallel)

    @property
    def num_layers(self) -> int:
        """Routers that replay (the model's MoE layers in order, MTP excluded)."""
        return len(self.routers)

    def arm_forward(self, microbatch: PreparedMicrobatch) -> None:
        """Set this microbatch's routing as the replay target and switch the routers to replay mode."""
        routed = microbatch.routed_experts
        if routed is None:
            return
        if not self.routers:
            raise ValueError("routed_experts is set but this engine has no replayable routers.")
        if routed.shape[-2] != len(self.routers):
            raise ValueError(f"routed_experts has {routed.shape[-2]} layers, the model has {len(self.routers)}.")
        if self.sequence_parallel:
            # TODO(replay-sp): with sequence parallelism each TP rank's router sees its 1/tp slice of the
            # tokens; slice the routing the same way before setting the targets.
            raise NotImplementedError("router replay with sequence parallelism (TP > 1) is not implemented yet.")
        # The router flattens [s, b] sequence-major; batch 1 for the packed layout.
        ids = routed.transpose(0, 1).reshape(-1, *routed.shape[2:])
        topk = ids.shape[-1]
        # Packing uses -1 for padding positions, which have no recorded expert choices.
        # MCore replays choices via scores.gather(), so replace these sentinels with valid
        # placeholder IDs [0, ..., topk-1]. This creates a new tensor; the packed routing
        # and padding_mask remain unchanged.
        # These placeholders do not skip expert computation by themselves: the
        # EP dispatcher uses padding_mask to remove padded assignments before dispatch.
        fallback = torch.arange(topk, device=ids.device).view(1, 1, topk).expand_as(ids)
        ids = torch.where(ids < 0, fallback, ids)  # padding rows get a valid, distinct routing
        for layer, router in enumerate(self.routers):
            router.set_target_indices(ids[:, layer].contiguous())
            router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    def arm_backward(self, microbatch: PreparedMicrobatch) -> None:
        """After the forward: recomputed MoE blocks in the backward replay the queued indices."""
        if microbatch.routed_experts is None:
            return
        for router in self.routers:
            router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

    def clear(self) -> None:
        """Back to native routing; drops the targets and queued indices (a caller's RECORD result stays)."""
        for router in self.routers:
            router.target_topk_idx = None
            router.replay_backward_list = []
            router.clear_router_replay_action()

    def release(self) -> None:
        """Remove this engine's instances from the process-wide registry (they hold the routers alive)."""
        registry = RouterReplay.global_router_replay_instances
        for router in self.owned:
            if router in registry:
                registry.remove(router)
        self.owned = []
        self.routers = []
