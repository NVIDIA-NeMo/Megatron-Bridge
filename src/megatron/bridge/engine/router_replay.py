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
        # With pipeline parallelism this rank owns only its stage's routers; see span_pipeline_stages.
        self.model_layer_indices: list[int] | None = None

    def span_pipeline_stages(self, pp_group: torch.distributed.ProcessGroup) -> None:
        """Learn every stage's MoE layers, so ``routed_experts`` keeps one column per MoE layer of the
        whole model and each stage replays the columns of its own routers. Collective over ``pp_group``."""
        stages: list[list[int] | None] = [None] * pp_group.size()
        torch.distributed.all_gather_object(stages, self._local_layer_indices(), group=pp_group)
        self.model_layer_indices = sorted(layer for stage in stages for layer in stage)

    def _local_layer_indices(self) -> list[int]:
        if any(router.layer_number is None for router in self.routers):
            raise RuntimeError("router layer numbers are not set; the model was not built through Megatron's spec")
        return [router.layer_number - 1 for router in self.routers]

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
        """MoE layers that replay (the whole model's, in order, MTP excluded): the columns of ``routed_experts``."""
        if self.model_layer_indices is not None:
            return len(self.model_layer_indices)
        return len(self.routers)

    @property
    def layer_indices(self) -> list[int]:
        """Zero-based decoder layer index of each replaying router, in replay order.

        Megatron numbers layers from 1 and copies the number onto the router's ``RouterReplay`` at
        build time. A backend that records routing for every decoder layer (vllm does) must select
        these columns; models that mix dense and MoE layers have fewer routers than layers. With
        pipeline parallelism the list covers every stage, not only this rank's routers.
        """
        if self.model_layer_indices is not None:
            return self.model_layer_indices
        return self._local_layer_indices()

    def arm_forward(self, microbatch: PreparedMicrobatch) -> None:
        """Set this microbatch's routing as the replay target and switch the routers to replay mode."""
        routed = microbatch.routed_experts
        if routed is None:
            return
        if not self.num_layers:
            raise ValueError("routed_experts is set but this engine has no replayable routers.")
        if routed.shape[-2] != self.num_layers:
            raise ValueError(f"routed_experts has {routed.shape[-2]} layers, the model has {self.num_layers}.")
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
        # Only padded positions may be -1 (check_microbatches enforces it); they get a valid, distinct routing.
        ids = torch.where(ids < 0, fallback, ids)
        columns = range(len(self.routers))
        if self.model_layer_indices is not None:
            # This stage's routers replay their own layers' columns among the whole model's MoE layers.
            columns = [self.model_layer_indices.index(layer) for layer in self._local_layer_indices()]
        for column, router in zip(columns, self.routers):
            router.set_target_indices(ids[:, column].contiguous())
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
