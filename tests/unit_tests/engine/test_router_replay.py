from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

from megatron.bridge.engine.datum import PreparedMicrobatch
from megatron.bridge.engine.router_replay import RouterReplayState


@pytest.fixture(autouse=True)
def _clean_registry():
    RouterReplay.clear_global_router_replay_instances()
    yield
    RouterReplay.clear_global_router_replay_instances()


def _microbatch(routed: torch.Tensor | None) -> PreparedMicrobatch:
    ids = torch.zeros(1, 4, dtype=torch.int64)
    return PreparedMicrobatch(
        input_ids=ids,
        target_ids=ids,
        position_ids=ids,
        loss_mask=torch.ones(1, 4),
        padding_mask=torch.zeros(1, 4, dtype=torch.bool),
        packed_seq_params=None,
        routed_experts=routed,
    )


def _build(num_layers: int, mtp: bool = False) -> RouterReplayState:
    mark = RouterReplayState.mark()
    for _ in range(num_layers):
        RouterReplay()
    if mtp:
        RouterReplay(is_mtp_layer=True)
    return RouterReplayState.claim(mark)


def test_claim_owns_only_this_builds_routers_and_skips_mtp():
    first = _build(2)
    second = _build(3, mtp=True)
    assert first.num_layers == 2 and second.num_layers == 3 and len(second.owned) == 4
    assert len(RouterReplay.global_router_replay_instances) == 6
    first.release()
    assert len(RouterReplay.global_router_replay_instances) == 4 and first.num_layers == 0
    assert RouterReplay.global_router_replay_instances == second.owned


def test_arm_forward_sets_targets_per_layer_and_fills_padding():
    state = _build(2, mtp=True)
    routed = torch.full((1, 4, 2, 3), -1, dtype=torch.int64)
    routed[0, :2, 0] = torch.tensor([[5, 6, 7], [8, 9, 10]])
    routed[0, :2, 1] = torch.tensor([[1, 2, 3], [4, 5, 6]])
    state.arm_forward(_microbatch(routed))
    layer0, layer1 = state.routers
    assert layer0.router_replay_action is RouterReplayAction.REPLAY_FORWARD
    assert layer0.target_topk_idx.tolist() == [[5, 6, 7], [8, 9, 10], [0, 1, 2], [0, 1, 2]]
    assert layer1.target_topk_idx.tolist() == [[1, 2, 3], [4, 5, 6], [0, 1, 2], [0, 1, 2]]
    assert len(layer0.replay_backward_list) == 1
    mtp = state.owned[-1]
    assert mtp.router_replay_action is None and mtp.target_topk_idx is None
    # The replay hook gathers scores at the targets: the same call path MCore uses.
    scores = torch.arange(4 * 12, dtype=torch.float32).view(4, 12)
    probs, idx = layer0.get_replay_topk(scores, 3)
    assert torch.equal(idx, layer0.target_topk_idx) and torch.equal(probs, scores.gather(1, idx))

    state.arm_backward(_microbatch(routed))
    assert layer1.router_replay_action is RouterReplayAction.REPLAY_BACKWARD
    probs, idx = layer1.get_replay_topk(scores, 3)
    assert torch.equal(idx, layer1.target_topk_idx) and layer1.replay_backward_list == []

    state.clear()
    assert layer0.router_replay_action is None and layer0.target_topk_idx is None
    assert layer0.replay_backward_list == []


def test_layer_indices_follow_megatron_layer_numbers():
    state = _build(3, mtp=True)
    # Megatron numbers layers from 1; a dense layer in between leaves a gap, MTP routers are excluded.
    for router, number in zip(state.routers, (1, 2, 4)):
        router.layer_number = number
    state.owned[-1].layer_number = 5
    assert state.layer_indices == [0, 1, 3]
    with pytest.raises(RuntimeError, match="layer numbers are not set"):
        _build(1).layer_indices


def test_a_pipeline_stage_replays_the_columns_of_its_own_layers(monkeypatch):
    # The second of two stages: the model's MoE layers are numbered 2, 4, 6, 8 and this stage owns 6 and 8.
    state = _build(2)
    for router, number in zip(state.routers, (6, 8)):
        router.layer_number = number

    def all_gather_object(gathered, local, group):
        gathered[:] = [[1, 3], local]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    state.span_pipeline_stages(SimpleNamespace(size=lambda: 2))
    assert state.layer_indices == [1, 3, 5, 7] and state.num_layers == 4

    routed = torch.arange(4 * 4 * 3).view(1, 4, 4, 3)
    state.arm_forward(_microbatch(routed))
    assert torch.equal(state.routers[0].target_topk_idx, routed[0, :, 2])
    assert torch.equal(state.routers[1].target_topk_idx, routed[0, :, 3])
    with pytest.raises(ValueError, match="2 layers, the model has 4"):
        state.arm_forward(_microbatch(routed[:, :, :2]))


def test_arm_is_a_no_op_without_routing():
    state = _build(2)
    state.arm_forward(_microbatch(None))
    state.arm_backward(_microbatch(None))
    assert all(r.router_replay_action is None and r.target_topk_idx is None for r in state.routers)


def test_arm_forward_rejects_mismatched_layers_and_missing_routers():
    with pytest.raises(ValueError, match="2 layers, the model has 3"):
        _build(3).arm_forward(_microbatch(torch.zeros(1, 4, 2, 3, dtype=torch.int64)))
    with pytest.raises(ValueError, match="no replayable routers"):
        RouterReplayState([]).arm_forward(_microbatch(torch.zeros(1, 4, 2, 3, dtype=torch.int64)))
    state = _build(3)
    state.sequence_parallel = True
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        state.arm_forward(_microbatch(torch.zeros(1, 4, 3, 3, dtype=torch.int64)))


def test_arm_forward_orders_bshd_tokens_sequence_major():
    state = _build(1)
    routed = torch.arange(2 * 3, dtype=torch.int64).view(2, 3, 1, 1)  # [B=2, S=3]
    state.arm_forward(_microbatch(routed))
    # Megatron routers flatten [s, b]: token order s0b0, s0b1, s1b0, ...
    assert state.routers[0].target_topk_idx[:, 0].tolist() == [0, 3, 1, 4, 2, 5]
