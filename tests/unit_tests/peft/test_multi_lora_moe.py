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

"""Unit tests for multi-adapter LoRA on grouped MoE expert linears.

Two layers of coverage:

CPU (fake token dispatcher, EP=1, no collectives):
  * the co-permuted slot ids match the permutation the dispatcher applied
  * ``sort_idx``/``inverse_idx``/``group_offsets``/``slot_token_counts`` agree with a
    reference stable sort by ``(slot, local_expert)``
  * the sequence-parallel narrow picks this rank's shard of the micro-batch
  * unsupported dispatch configurations raise instead of silently mis-routing
  * ``install_moe_slot_routing`` is idempotent and skips adapter-free MoE layers

Single-GPU integration (needs CUDA + model-parallel init):
  * the grouped forward equals a per-(slot, expert) reference LoRA computation
  * every slot's parameters receive a gradient, including slots that got no tokens
    and the all-empty case (Megatron's grad buffers expect one hook per parameter)
"""

import os
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from megatron.bridge.peft.multi_lora_layers import (
    MultiLoRAGroupedExpertLinear,
    _build_expert_slot_routing,
    _co_permute_slot_ids,
    install_moe_slot_routing,
)


# ======================================================================
# Test doubles
# ======================================================================


def _fake_dispatcher_cls():
    """Subclass the real dispatcher so the routing's isinstance gate is satisfied.

    Only the attributes the slot routing reads are populated; the real ``__init__``
    needs a full transformer config and process groups.
    """
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    class _FakeAlltoAllDispatcher(MoEAlltoAllTokenDispatcher):
        def __init__(
            self,
            sorted_indices: torch.Tensor,
            num_local_tokens: int,
            *,
            num_local_experts: int = 1,
            tokens_per_expert: torch.Tensor | None = None,
            ep_size: int = 1,
            tp_size: int = 1,
        ) -> None:
            self.reversed_local_input_permutation_mapping = sorted_indices
            self.hidden_shape_before_permute = (num_local_tokens, 8)
            self.ep_size = ep_size
            self.tp_size = tp_size
            self.num_local_experts = num_local_experts
            # With EP=1 there is a single source chunk per local expert, so the
            # local-expert sort is the identity — as it is in Megatron for ep*tp == 1.
            if tokens_per_expert is not None:
                self.num_global_tokens_per_local_expert = tokens_per_expert.view(1, -1)
                self.sort_input_by_local_experts = torch.arange(num_local_experts)

    return _FakeAlltoAllDispatcher


def _FakeDispatcher(*args, **kwargs):  # noqa: N802 - reads as a class at call sites
    return _fake_dispatcher_cls()(*args, **kwargs)


def _expert_major_permutation(token_experts: list[int], num_experts: int) -> torch.Tensor:
    """Indices that group tokens by expert, mirroring ``moe_utils.permute`` for topk=1.

    ``permute`` builds these with ``token_indices.masked_select(routing_map.T)``,
    i.e. expert-major with token order preserved inside each expert.
    """
    order: list[int] = []
    for expert in range(num_experts):
        order.extend(i for i, e in enumerate(token_experts) if e == expert)
    return torch.tensor(order, dtype=torch.long)


def _reference_routing(slot_ids: list[int], expert_ids: list[int], n_adapters: int, num_local_experts: int):
    """Reference (slot, expert) segmentation computed with plain Python."""
    keys = [s * num_local_experts + e for s, e in zip(slot_ids, expert_ids)]
    order = sorted(range(len(keys)), key=lambda i: (keys[i], i))  # stable
    counts = [0] * (n_adapters * num_local_experts)
    for k in keys:
        counts[k] += 1
    return order, counts


# ======================================================================
# CPU: slot-id co-permutation and segmentation
# ======================================================================


def test_co_permute_follows_dispatcher_permutation():
    # 6 tokens, slots [0,0,0,1,1,1], routed to experts [1,0,1,0,1,0].
    token_experts = [1, 0, 1, 0, 1, 0]
    sorted_indices = _expert_major_permutation(token_experts, num_experts=2)
    slot_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    dispatcher = _FakeDispatcher(
        sorted_indices,
        num_local_tokens=6,
        num_local_experts=2,
        tokens_per_expert=torch.tensor([3, 3]),
    )

    permuted = _co_permute_slot_ids(dispatcher, slot_ids, num_local_experts=2)

    # Expert 0 receives tokens 1,3,5 (slots 0,1,1); expert 1 receives 0,2,4 (slots 0,0,1).
    assert permuted.tolist() == [0, 1, 1, 0, 0, 1]


def test_routing_matches_reference_segmentation():
    token_experts = [1, 0, 1, 0, 1, 0, 0, 1]
    sorted_indices = _expert_major_permutation(token_experts, num_experts=2)
    tokens_per_expert = torch.tensor([4, 4])
    dispatcher = _FakeDispatcher(
        sorted_indices,
        num_local_tokens=8,
        num_local_experts=2,
        tokens_per_expert=tokens_per_expert,
    )
    tokens_per_adapter = torch.tensor([3, 5], dtype=torch.int32)

    routing = _build_expert_slot_routing(
        dispatcher,
        tokens_per_expert,
        tokens_per_adapter,
        n_adapters=2,
        num_local_experts=2,
        device=torch.device("cpu"),
    )

    dispatched_slots = [0, 0, 0, 1, 1, 1, 1, 1]
    permuted_slots = [dispatched_slots[i] for i in sorted_indices.tolist()]
    permuted_experts = [token_experts[i] for i in sorted_indices.tolist()]
    expected_order, expected_counts = _reference_routing(permuted_slots, permuted_experts, 2, 2)

    assert routing.num_tokens == 8
    assert routing.sort_idx.tolist() == expected_order
    assert routing.group_offsets.tolist() == torch.tensor(expected_counts).cumsum(0).tolist()
    assert routing.slot_token_counts.tolist() == [3, 5]
    assert routing.group_offsets.dtype == torch.int32


def test_inverse_index_round_trips():
    token_experts = [0, 1, 0, 1]
    sorted_indices = _expert_major_permutation(token_experts, num_experts=2)
    tokens_per_expert = torch.tensor([2, 2])
    routing = _build_expert_slot_routing(
        _FakeDispatcher(sorted_indices, num_local_tokens=4, num_local_experts=2, tokens_per_expert=tokens_per_expert),
        tokens_per_expert,
        torch.tensor([2, 2], dtype=torch.int32),
        n_adapters=2,
        num_local_experts=2,
        device=torch.device("cpu"),
    )

    rows = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    assert torch.equal(rows.index_select(0, routing.sort_idx).index_select(0, routing.inverse_idx), rows)


def test_sequence_parallel_narrow_selects_this_rank_shard():
    """With SP the MoE input is a contiguous shard of the micro-batch's tokens."""
    # 4 local tokens out of an 8-token micro-batch: this is TP rank 1's half,
    # whose slot ids are the second half of [0,0,0,0,1,1,1,1].
    token_experts = [0, 0, 1, 1]
    sorted_indices = _expert_major_permutation(token_experts, num_experts=2)
    tokens_per_expert = torch.tensor([2, 2])
    dispatcher = _FakeDispatcher(
        sorted_indices, num_local_tokens=4, num_local_experts=2, tokens_per_expert=tokens_per_expert
    )

    with (
        patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size", return_value=2),
        patch("megatron.core.parallel_state.get_tensor_model_parallel_rank", return_value=1),
    ):
        routing = _build_expert_slot_routing(
            dispatcher,
            tokens_per_expert,
            torch.tensor([4, 4], dtype=torch.int32),
            n_adapters=2,
            num_local_experts=2,
            device=torch.device("cpu"),
        )

    # All four local tokens belong to slot 1 (the second half of the micro-batch).
    assert routing.slot_token_counts.tolist() == [0, 4]


def test_token_count_mismatch_raises():
    token_experts = [0, 1]
    dispatcher = _FakeDispatcher(
        _expert_major_permutation(token_experts, num_experts=2), num_local_tokens=2, num_local_experts=1
    )
    with (
        patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size", return_value=1),
        pytest.raises(RuntimeError, match="adapter-slot token ids"),
    ):
        _build_expert_slot_routing(
            dispatcher,
            torch.tensor([2]),
            torch.tensor([2, 3], dtype=torch.int32),  # sums to 5, not 2
            n_adapters=2,
            num_local_experts=1,
            device=torch.device("cpu"),
        )


def test_expert_tensor_parallel_dispatch_raises():
    slot_ids = torch.zeros(4, dtype=torch.int32)
    expert_tp = _FakeDispatcher(torch.arange(4), num_local_tokens=4, tp_size=2)
    with pytest.raises(NotImplementedError, match="expert_tensor_parallel_size=1"):
        _co_permute_slot_ids(expert_tp, slot_ids, num_local_experts=1)


def test_non_alltoall_dispatcher_raises():
    """The all-gather dispatcher records a permutation mapping too, so the gate is by type.

    Its dispatch stages differ completely (TP*EP all-gather, no EP all-to-all, no
    local-expert chunk sort), so replaying the all-to-all stages against it would
    silently pair tokens with the wrong slots.
    """

    class _AllGatherLike:
        # Same attribute MoEAllGatherTokenDispatcher sets.
        reversed_local_input_permutation_mapping = torch.arange(4)
        ep_size = 1
        tp_size = 1
        num_local_experts = 1

    with pytest.raises(NotImplementedError, match="MoEAlltoAllTokenDispatcher"):
        _co_permute_slot_ids(_AllGatherLike(), torch.zeros(4, dtype=torch.int32), num_local_experts=1)


# ======================================================================
# CPU: hook installation
# ======================================================================


class _FakeExperts(nn.Module):
    def __init__(self, child: nn.Module | None) -> None:
        super().__init__()
        if child is not None:
            self.linear_fc1 = child

    def forward(self, hidden, tokens_per_expert, probs):  # pragma: no cover - never called
        return hidden, None


class _FakeMoELayer(nn.Module):
    """Stands in for the isinstance check only; BaseMoELayer needs a full config."""

    def __init__(self, child: nn.Module | None) -> None:
        super().__init__()
        self.experts = _FakeExperts(child)
        self.token_dispatcher = None


def test_install_moe_slot_routing_is_idempotent_and_selective():
    wrapped = MultiLoRAGroupedExpertLinear.__new__(MultiLoRAGroupedExpertLinear)
    nn.Module.__init__(wrapped)

    with_adapter = _FakeMoELayer(wrapped)
    without_adapter = _FakeMoELayer(nn.Linear(4, 4))
    model = nn.ModuleList([with_adapter, without_adapter])

    # install_moe_slot_routing imports BaseMoELayer at call time, so patching the
    # source module is enough to make the fake layers match.
    with patch("megatron.core.transformer.moe.moe_layer.BaseMoELayer", _FakeMoELayer):
        assert install_moe_slot_routing([model]) == 1
        # A second pass must not stack a duplicate hook on the same experts module.
        assert install_moe_slot_routing([model]) == 0

    assert len(with_adapter.experts._forward_pre_hooks) == 1
    assert len(without_adapter.experts._forward_pre_hooks) == 0


# ======================================================================
# Single-GPU integration
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU + model-parallel init")
class TestMultiLoRAGroupedExpertLinearGPU:
    @pytest.fixture(autouse=True)
    def _mp(self):
        import megatron.core.parallel_state as parallel_state
        import torch.distributed as dist

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29557")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            torch.cuda.set_device(0)
            dist.init_process_group(backend="nccl", world_size=1, rank=0)
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        from megatron.core.process_groups_config import ProcessGroupCollection

        from megatron.bridge.training.initialize import _set_random_seed

        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        yield
        try:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    def _build_base(self, *, hidden=16, ffn=32, num_local_experts=2, **config_overrides):
        """A grouped expert linear as TEGroupedMLP builds its ``linear_fc1``."""
        from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear
        from megatron.core.transformer.transformer_config import TransformerConfig

        from megatron.bridge.peft.utils import init_method_normal

        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden,
            ffn_hidden_size=ffn,
            num_attention_heads=1,
            num_moe_experts=num_local_experts,
            moe_grouped_gemm=True,
            # Not the TransformerConfig default ('allgather'), and required: the
            # slot routing replays this dispatcher's permutation stages.
            moe_token_dispatcher_type="alltoall",
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            bf16=True,
            params_dtype=torch.bfloat16,
            **config_overrides,
        )
        base = TEColumnParallelGroupedLinear(
            num_gemms=num_local_experts,
            input_size=hidden,
            output_size=ffn,
            config=config,
            init_method=init_method_normal(0.02),
            bias=False,
            skip_bias_add=True,
            is_expert=True,
            tp_comm_buffer_name="fc1",
        ).cuda()
        return base, config

    def _build(self, *, hidden=16, ffn=32, num_local_experts=2, n_adapters=2, dim=8, alpha=16):
        base, config = self._build_base(hidden=hidden, ffn=ffn, num_local_experts=num_local_experts)
        layer = MultiLoRAGroupedExpertLinear(
            to_wrap=base,
            n_adapters=n_adapters,
            dim=dim,
            alpha=alpha,
            full_name="decoder.layers.0.mlp.experts.linear_fc1",
            num_local_experts=num_local_experts,
        )
        layer.adapters.to(device="cuda", dtype=torch.bfloat16)
        return layer, config

    @staticmethod
    def _routing_for(slots: list[int], experts: list[int], n_adapters: int, num_local_experts: int):
        """Build ExpertSlotRouting directly for a known (slot, expert) assignment."""
        from megatron.bridge.peft.multi_lora_layers import ExpertSlotRouting

        keys = torch.tensor(
            [s * num_local_experts + e for s, e in zip(slots, experts)], device="cuda", dtype=torch.long
        )
        sort_idx = torch.argsort(keys, stable=True)
        inverse_idx = torch.empty_like(sort_idx)
        inverse_idx.scatter_(0, sort_idx, torch.arange(keys.numel(), device="cuda"))
        counts = torch.bincount(keys, minlength=n_adapters * num_local_experts)
        return ExpertSlotRouting(
            sort_idx=sort_idx,
            inverse_idx=inverse_idx,
            group_offsets=counts.cumsum(dim=0, dtype=torch.int32),
            slot_token_counts=counts.view(n_adapters, num_local_experts).sum(dim=1),
            num_tokens=keys.numel(),
        )

    def test_forward_matches_per_token_reference(self):
        layer, _ = self._build()
        # Tokens are expert-major (as they arrive from the dispatcher) with slots
        # interleaved inside each expert — the case a contiguous per-slot span
        # cannot express.
        experts = [0, 0, 0, 1, 1, 1]
        slots = [0, 1, 0, 1, 1, 0]
        for idx, rank, alpha in ((0, 8, 16.0), (1, 4, 8.0)):
            layer.init_adapter_slot(idx, rank=rank, alpha=alpha)
        # B is zero-initialised, so give it content or every delta is zero.
        for adapter in layer.adapters:
            with torch.no_grad():
                adapter.linear_out.weight.normal_(std=0.02)
        layer._apply_rank_mask(1)

        x = torch.randn(len(slots), 16, dtype=torch.bfloat16, device="cuda")
        layer.expert_slot_routing = self._routing_for(slots, experts, 2, 2)
        # m_splits is how TEGroupedMLP passes tokens_per_expert to the grouped linear.
        out, _ = layer(x, [3, 3])

        base_out, _ = layer.to_wrap(x, [3, 3])
        expected = base_out.clone()
        for i, (slot, expert) in enumerate(zip(slots, experts)):
            adapter = layer.adapters[slot]
            scale = layer.alpha_values[slot] / layer.rank_values[slot]
            mid = x[i] @ adapter.linear_in.weight[expert].T
            expected[i] += (mid @ adapter.linear_out.weight[expert].T) * scale

        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

    def test_every_slot_gets_a_gradient(self):
        """Slot 1 receives no tokens, but Megatron's grad buffers still need its hook."""
        layer, _ = self._build()
        for idx in (0, 1):
            layer.init_adapter_slot(idx, rank=8, alpha=16.0)
        for adapter in layer.adapters:
            with torch.no_grad():
                adapter.linear_out.weight.normal_(std=0.02)

        x = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")
        layer.expert_slot_routing = self._routing_for([0, 0, 0, 0], [0, 0, 1, 1], 2, 2)
        out, _ = layer(x, [2, 2])
        out.sum().backward()

        for idx, adapter in enumerate(layer.adapters):
            assert adapter.linear_in.weight.grad is not None, f"slot {idx} linear_in got no grad"
            assert adapter.linear_out.weight.grad is not None, f"slot {idx} linear_out got no grad"

    def test_no_tokens_still_produces_gradients(self):
        layer, config = self._build()
        for idx in (0, 1):
            layer.init_adapter_slot(idx, rank=8, alpha=16.0)

        # The base grouped linear has its own all-empty handling; stub it out so
        # this exercises only the adapter's empty-batch path.
        class _EmptyBase(nn.Module):
            def forward(self, x, *args, **kwargs):
                return x.new_zeros((x.shape[0], config.ffn_hidden_size), requires_grad=True), None

        layer.to_wrap = _EmptyBase()

        x = torch.zeros(0, 16, dtype=torch.bfloat16, device="cuda")
        layer.expert_slot_routing = self._routing_for([], [], 2, 2)
        out, _ = layer(x, [0, 0])
        out.sum().backward()

        for idx, adapter in enumerate(layer.adapters):
            assert adapter.linear_in.weight.grad is not None, f"slot {idx} linear_in got no grad"
            assert adapter.linear_out.weight.grad is not None, f"slot {idx} linear_out got no grad"

    def test_rank_mask_zeroes_padded_rank_on_every_expert(self):
        layer, _ = self._build(dim=8)
        with torch.no_grad():
            for adapter in layer.adapters:
                adapter.linear_in.weight.fill_(1.0)
                adapter.linear_out.weight.fill_(1.0)

        layer.init_adapter_slot(0, rank=3, alpha=6.0)

        adapter = layer.adapters[0]
        assert adapter.linear_in.weight[:, 3:, :].abs().max().item() == 0
        assert adapter.linear_out.weight[..., 3:].abs().max().item() == 0
        assert adapter.linear_in.weight[:, :3, :].abs().min().item() == 1
        assert adapter.linear_out.weight[..., :3].abs().min().item() == 1

    def test_reset_adapter_draws_from_the_expert_rng_stream(self):
        """Expert adapters must re-init from the expert tracker, not the dense one.

        The dense model-parallel seed varies with the *dense* TP rank, and
        expert-data-parallel peers differ in exactly that rank whenever
        tensor_model_parallel_size != expert_tensor_parallel_size (miles' MoE
        recipes run TP>1 with ETP=1). Drawing from the dense stream there would
        give each EDP replica of a reused slot different weights.
        """
        import megatron.core.tensor_parallel.random as mcore_random

        layer, _ = self._build()
        real_tracker = mcore_random.get_cuda_rng_tracker()
        forked: list[object] = []

        class _RecordingTracker:
            def fork(self, name=None):
                forked.append(name)
                return real_tracker.fork(name) if name is not None else real_tracker.fork()

        with patch.object(mcore_random, "get_cuda_rng_tracker", lambda: _RecordingTracker()):
            layer.clear_adapter_slot(0)

        assert forked == [mcore_random.get_expert_parallel_rng_tracker_name()], (
            f"expert adapter re-init forked {forked}, not the expert rng stream"
        )
        # The stream must also actually exist, or the fork would raise at runtime.
        assert mcore_random.get_expert_parallel_rng_tracker_name() in real_tracker.get_states()

    @pytest.mark.parametrize(
        "field, value, message",
        [
            # Most bridge MoE providers default fusion on (e.g. Qwen3-MoE), and the
            # fused permute records TE's row_id_map rather than a token gather index.
            ("moe_permute_fusion", True, "moe_permute_fusion=False"),
            ("moe_token_dispatcher_type", "allgather", "alltoall"),
            ("moe_pad_expert_input_to_capacity", True, "moe_pad_expert_input_to_capacity"),
        ],
    )
    def test_unsupported_dispatch_configs_are_rejected_at_build_time(self, field, value, message):
        """Config-level requirements must fail uniformly at wrap time, not inside the hook.

        A hook that raised on only some ranks would leave its expert-parallel peers
        waiting in the companion all-to-all.

        The field is set on the resolved config rather than passed to
        ``TransformerConfig``, whose own validation would otherwise reject some of
        these combinations before the adapter is ever constructed.
        """
        base, config = self._build_base()
        setattr(config, field, value)
        with pytest.raises(NotImplementedError, match=message):
            MultiLoRAGroupedExpertLinear(
                to_wrap=base,
                n_adapters=2,
                dim=8,
                alpha=16,
                full_name="decoder.layers.0.mlp.experts.linear_fc1",
                num_local_experts=2,
            )

    def test_expert_tensor_parallel_is_rejected(self):
        # Build the base with real (ETP=1) state, then construct the adapter layer
        # under a patched ETP size so only the layer's own guard is exercised.
        base, _ = self._build_base()
        with (
            patch("megatron.core.parallel_state.get_expert_tensor_parallel_world_size", return_value=2),
            pytest.raises(NotImplementedError, match="expert_tensor_parallel_size=1"),
        ):
            MultiLoRAGroupedExpertLinear(
                to_wrap=base,
                n_adapters=2,
                dim=8,
                alpha=16,
                full_name="decoder.layers.0.mlp.experts.linear_fc1",
                num_local_experts=2,
            )
