"""Release-pin regression coverage for mixed-dtype optimizer checkpoint lookup."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.optimizer import DistributedOptimizer, OptimizerConfig

from megatron.bridge.training.config import SchedulerConfig
from megatron.bridge.training.optim import _align_distributed_optimizer_checkpoint_indices, setup_optimizer


class _StateStorage:
    """CPU storage implementing the precision-aware state accessor contract."""

    def __init__(self, groups, state):
        self.param_groups = groups
        self.state = state

    def get_unscaled_state(self, param, key):
        return self.state[param][key]

    def set_scaled_state(self, param, key, value):
        self.state[param][key].copy_(value)


def _build_optimizer(dtypes, sizes, precision_aware):
    # Two populated groups plus an empty group exercise group-local indices.
    dtypes = dtypes * 2
    sizes = sizes * 2
    params = [
        torch.nn.Parameter(torch.full((size,), i + 1.0, dtype=dtype))
        for i, (dtype, size) in enumerate(zip(dtypes, sizes, strict=True))
    ]
    groups = [{"params": params[:2]}, {"params": params[2:]}, {"params": []}]
    buffers = [{dtype: [{"param_map": {param: None}}]} for param, dtype in zip(params, dtypes, strict=True)]
    index_map, _ = DistributedOptimizer._build_optimizer_group_ranges(groups, buffers)
    fp32 = [[p for p in group["params"] if p.dtype == torch.float32] for group in groups]
    low_precision = [[p for p in group["params"] if p.dtype != torch.float32] for group in groups]
    shards = {p: p.detach().float().clone() for p in params}
    inner_groups = [
        {"params": [shards[p] for p in [*high, *low]]} for high, low in zip(fp32, low_precision, strict=True)
    ]
    state = {
        shards[p]: {
            "exp_avg": torch.full_like(shards[p], i + 11.0),
            "exp_avg_sq": torch.full_like(shards[p], i + 21.0),
        }
        for i, p in enumerate(params)
    }
    if precision_aware:
        for p in params:
            state[shards[p]]["master_param"] = shards[p].clone()
    opt = object.__new__(DistributedOptimizer)
    opt.optimizer = _StateStorage(inner_groups, state)
    opt.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=precision_aware)
    opt.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    opt.is_stub_optimizer = False
    opt.model_param_group_index_map = index_map
    opt.model_fp32_groups = fp32
    opt.model_float16_groups = low_precision
    return opt, params, shards


@pytest.mark.unit
@pytest.mark.parametrize("precision_aware", [False, True])
@pytest.mark.parametrize("sizes", [(3, 7), (3, 3)])
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
@torch.no_grad()
def test_setup_preserves_checkpoint_parameter_and_state_identity(precision_aware, sizes, dtypes):
    opt, params, shards = _build_optimizer(dtypes, sizes, precision_aware)
    # Include a nested chain and a non-distributed member, as in composite optimizers.
    chain = SimpleNamespace(chained_optimizers=[SimpleNamespace(chained_optimizers=[opt]), object()])
    overrides = Mock()
    overrides.build_config_overrides.return_value = {}
    with (
        patch("megatron.bridge.training.optim.get_model_config", return_value=SimpleNamespace(use_mup=False)),
        patch("megatron.bridge.training.optim.get_megatron_optimizer", return_value=chain),
        patch("megatron.bridge.training.optim._get_scheduler"),
    ):
        actual, _ = setup_optimizer(
            OptimizerConfig(optimizer="adam"),
            SchedulerConfig(),
            [Mock()],
            optimizer_config_override_provider=overrides,
        )
    assert actual is chain
    for i, param in enumerate(params):
        tensors = opt._get_main_param_and_optimizer_states(param)
        torch.testing.assert_close(tensors["param"], param.float())
        torch.testing.assert_close(tensors["exp_avg"], torch.full_like(shards[param], i + 11.0))
        torch.testing.assert_close(tensors["exp_avg_sq"], torch.full_like(shards[param], i + 21.0))
        restored = {key: value + 100.0 for key, value in tensors.items()}
        opt._set_main_param_and_optimizer_states(param, restored)
        # Check the intended storage directly: a symmetric save/load swap could otherwise pass.
        state = opt.optimizer.state[shards[param]]
        master = state["master_param"] if precision_aware else shards[param]
        torch.testing.assert_close(master, param.float() + 100.0)
        torch.testing.assert_close(state["exp_avg"], torch.full_like(shards[param], i + 111.0))
        torch.testing.assert_close(state["exp_avg_sq"], torch.full_like(shards[param], i + 121.0))
    before = dict(opt.model_param_group_index_map)
    _align_distributed_optimizer_checkpoint_indices(opt)
    assert opt.model_param_group_index_map == before


@pytest.mark.unit
@pytest.mark.parametrize("stub,fsdp", [(True, False), (False, True)])
def test_early_return_optimizers_do_not_require_shard_groups(stub, fsdp):
    opt = object.__new__(DistributedOptimizer)
    opt.is_stub_optimizer = stub
    opt.ddp_config = SimpleNamespace(use_megatron_fsdp=fsdp)
    _align_distributed_optimizer_checkpoint_indices(opt)
    assert not hasattr(opt, "model_param_group_index_map")
