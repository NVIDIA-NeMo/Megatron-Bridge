"""Two-GPU numerical contracts for replicated decision heads.

Run with ``uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest
 tests/functional_tests/test_groups/decision/test_decision_distributed_gradients.py``.
"""

import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from torch import nn

from megatron.bridge.decision.losses import DecisionLossConfig, DecisionTargets
from megatron.bridge.decision.megatron_glue import all_reduce_mean_grads, final_hidden_states_processor


pytestmark = pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2" or not torch.cuda.is_available(), reason="requires a two-rank CUDA launch"
)


@pytest.fixture
def two_gpu_group():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group("nccl")
    yield dist.group.WORLD
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if owned:
        dist.destroy_process_group()


def test_sequence_parallel_replicated_head_preserves_backbone_gradient(two_gpu_group):
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    group = parallel_state.get_tensor_model_parallel_group()
    rank = dist.get_rank(group)
    torch.manual_seed(71)
    torch.backends.cuda.matmul.allow_tf32 = False
    backbone = nn.Linear(4, 4, bias=False, device="cuda")
    head = nn.Linear(4, 3, device="cuda")
    reference_backbone, reference_head = copy.deepcopy(backbone), copy.deepcopy(head)
    values = torch.randn(6, 2, 4, device="cuda")
    reference_values = values.detach().clone().requires_grad_()
    local_values = values.chunk(2)[rank].detach().clone().requires_grad_()
    processor = final_hidden_states_processor(tp_group=group)
    actual = head(processor(hidden_states=backbone(local_values), config=SimpleNamespace(sequence_parallel=True)))
    expected = reference_head(reference_backbone(reference_values).transpose(0, 1))
    probe = torch.randn_like(expected)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(local_values.grad, reference_values.grad.chunk(2)[rank], atol=1e-6, rtol=1e-6)
    # SP partitions the backbone's work; its replicated parameter gradient is
    # the sum of local token contributions. The replicated head sees one loss.
    dist.all_reduce(backbone.weight.grad, group=group)
    torch.testing.assert_close(backbone.weight.grad, reference_backbone.weight.grad, atol=1e-6, rtol=1e-6)
    for actual_parameter, reference_parameter in zip(head.parameters(), reference_head.parameters()):
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad, atol=1e-6, rtol=1e-6)


def _targets(labels, *, rewards=None):
    return DecisionTargets(
        labels=torch.tensor(labels, device="cuda"),
        option_mask=torch.ones(len(labels), 2, device="cuda", dtype=torch.bool),
        question_types=torch.ones(len(labels), device="cuda", dtype=torch.long),
        rewards=None if rewards is None else torch.tensor(rewards, device="cuda"),
    )


def test_variable_feedback_window_matches_single_process_gradient_and_step(two_gpu_group):
    rank = dist.get_rank(two_gpu_group)
    objective = DecisionLossConfig(mode="rl", kl_coef=0, anchor_weight=0).build()
    windows = [
        [_targets([0]), _targets([-100])],
        [_targets([1, 1, 1]), _targets([-100], rewards=[[0.2, 0.8]])],
    ]
    targets = windows[rank]
    total = sum(objective.normalization_weight(t) for t in targets)
    dist.all_reduce(total, group=two_gpu_group)
    head = nn.Linear(1, 2, bias=False, device="cuda")
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0.3], [-0.2]], device="cuda"))
    reference = copy.deepcopy(head)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    for microbatch in targets:
        logits = head(torch.ones(len(microbatch.labels), 1, device="cuda"))
        loss, _ = objective(logits, microbatch)
        (loss * 2 * objective.normalization_weight(microbatch) / total).backward()
    all_reduce_mean_grads(head, group=two_gpu_group)
    combined = _targets(
        [0, -100, 1, 1, 1, -100],
        rewards=[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.2, 0.8]],
    )
    combined.reward_mask = torch.tensor([False, False, False, False, False, True], device="cuda")
    reference_loss, _ = objective(reference(torch.ones(6, 1, device="cuda")), combined)
    reference_loss.backward()
    torch.testing.assert_close(head.weight.grad, reference.weight.grad, atol=1e-6, rtol=1e-6)
    optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(head.weight, reference.weight, atol=1e-6, rtol=1e-6)
