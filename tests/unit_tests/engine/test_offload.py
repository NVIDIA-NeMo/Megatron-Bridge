from types import SimpleNamespace

import pytest
import torch
from megatron.core.distributed import DistributedDataParallel

from megatron.bridge.engine.offload import HostState, check_movable, offload_to_cpu, reload_to_cuda


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host memory needs CUDA")

# Tensor sizes are multiples of the caching allocator's 512-byte block, so torch.cuda.memory_allocated
# moves by exactly the bytes offload reports.
BUFFER_NUMEL = 1 << 20  # 2 MiB bf16 params, 4 MiB fp32 grads
MASTER_NUMEL = 1 << 18  # 1 MiB fp32
FROZEN_NUMEL = 1 << 16  # 256 KiB fp32


def _allocated() -> int:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


class _Buffer:
    """The four attributes and two methods of MCore's _ParamAndGradBuffer that offload uses."""

    def __init__(self, numel: int):
        self.param_data = torch.randn(numel, device="cuda", dtype=torch.bfloat16)
        self.grad_data = torch.randn(numel, device="cuda", dtype=torch.float32)
        self.param_data_cpu = None
        self.param_data_size = 0
        self.grad_data_size = 0

    def offload_to_cpu(self, move_params: bool = True, move_grads: bool = True) -> None:
        if move_grads and self.grad_data.storage().size() > 0:
            self.grad_data_size = self.grad_data.storage().size()
            self.grad_data.storage().resize_(0)
        if move_params and self.param_data.storage().size() > 0:
            self.param_data_size = self.param_data.storage().size()
            if self.param_data_cpu is not None:
                self.param_data_cpu.copy_(self.param_data, non_blocking=True)
            else:
                self.param_data_cpu = self.param_data.cpu().pin_memory()
            self.param_data.storage().resize_(0)

    def reload_from_cpu(self, move_params: bool = True, move_grads: bool = True) -> None:
        if move_params and self.param_data_cpu is not None and self.param_data.storage().size() == 0:
            self.param_data.storage().resize_(self.param_data_size)
            self.param_data.copy_(self.param_data_cpu, non_blocking=True)
        if move_grads and self.grad_data_size > 0:
            self.grad_data.storage().resize_(self.grad_data_size)
            self.grad_data.zero_()
            self.grad_data_size = 0


def _ddp_chunk(buffer: _Buffer) -> DistributedDataParallel:
    chunk = DistributedDataParallel.__new__(DistributedDataParallel)
    torch.nn.Module.__init__(chunk)
    chunk.buffers = [buffer]
    chunk.expert_parallel_buffers = []
    chunk.register_parameter("weight", torch.nn.Parameter(buffer.param_data[:4].view(2, 2)))
    chunk.weight.main_grad = buffer.grad_data[:4].view(2, 2)
    frozen = torch.nn.Parameter(torch.randn(FROZEN_NUMEL, device="cuda"), requires_grad=False)
    chunk.register_parameter("frozen", frozen)
    return chunk


def _optimizer(master: torch.Tensor) -> SimpleNamespace:
    state = {master: {"exp_avg": torch.randn_like(master), "exp_avg_sq": torch.rand_like(master), "step": 3}}
    inner = SimpleNamespace(shard_fp32_from_float16_groups=[[master, None]], optimizer=SimpleNamespace(state=state))
    return SimpleNamespace(chained_optimizers=[inner])


@cuda_only
def test_round_trip_preserves_params_grads_and_optimizer_tensors_and_identity():
    buffer = _Buffer(BUFFER_NUMEL)
    chunk = _ddp_chunk(buffer)
    master = torch.randn(MASTER_NUMEL, device="cuda")
    optimizer = _optimizer(master)
    adam = optimizer.chained_optimizers[0].optimizer.state[master]
    before = {
        "params": buffer.param_data.clone(),
        "grads": buffer.grad_data.clone(),
        "frozen": chunk.frozen.data.clone(),
        "master": master.clone(),
        "exp_avg": adam["exp_avg"].clone(),
        "exp_avg_sq": adam["exp_avg_sq"].clone(),
    }
    host = HostState()
    expected_bytes = sum(
        t.numel() * t.element_size()
        for t in (buffer.param_data, buffer.grad_data, chunk.frozen.data, master, adam["exp_avg"], adam["exp_avg_sq"])
    )
    resident = _allocated()

    released = offload_to_cpu([chunk], optimizer, host, keep_gradients=True)
    assert released == expected_bytes
    assert resident - _allocated() == expected_bytes  # the GPU really gave the bytes back
    assert buffer.param_data.storage().size() == 0 and buffer.grad_data.storage().size() == 0
    assert chunk.frozen.data.device.type == "cpu" and chunk.frozen.data.is_pinned()
    assert master.device.type == "cpu" and adam["exp_avg"].device.type == "cpu"
    assert optimizer.chained_optimizers[0].shard_fp32_from_float16_groups[0][0] is master  # identity kept
    assert list(optimizer.chained_optimizers[0].optimizer.state) == [master]
    host_ptrs = {id(master): master.data.data_ptr(), id(chunk.frozen): chunk.frozen.data.data_ptr()}

    restored = reload_to_cuda([chunk], host)
    assert restored == expected_bytes and _allocated() == resident  # back to the resident footprint
    assert torch.equal(buffer.param_data, before["params"]) and torch.equal(buffer.grad_data, before["grads"])
    assert torch.equal(chunk.weight.main_grad, before["grads"][:4].view(2, 2))  # the view is still live
    assert torch.equal(chunk.frozen.data, before["frozen"]) and chunk.frozen.device.type == "cuda"
    assert torch.equal(master, before["master"]) and master.device.type == "cuda"
    assert torch.equal(adam["exp_avg"], before["exp_avg"]) and torch.equal(adam["exp_avg_sq"], before["exp_avg_sq"])
    assert adam["step"] == 3 and host.swapped == [] and host.grads == {}

    # A second cycle reuses the pinned host copies instead of allocating new ones, and frees the same bytes.
    assert offload_to_cpu([chunk], optimizer, host, keep_gradients=True) == expected_bytes
    assert resident - _allocated() == expected_bytes
    assert master.data.data_ptr() == host_ptrs[id(master)]
    assert chunk.frozen.data.data_ptr() == host_ptrs[id(chunk.frozen)]
    reload_to_cuda([chunk], host)
    assert _allocated() == resident
    assert torch.equal(master, before["master"])


@cuda_only
def test_closed_window_frees_gradients_and_reloads_them_zeroed():
    buffer = _Buffer(BUFFER_NUMEL)
    chunk = _ddp_chunk(buffer)
    params = buffer.param_data.clone()
    host = HostState()
    resident = _allocated()
    released = offload_to_cpu([chunk], None, host, keep_gradients=False)
    # Gradients are released from the GPU even though no host copy is kept.
    assert released == sum(t.numel() * t.element_size() for t in (params, buffer.grad_data, chunk.frozen.data))
    assert resident - _allocated() == released
    assert host.grads == {}
    assert reload_to_cuda([chunk], host) == released and _allocated() == resident
    assert torch.equal(buffer.param_data, params)
    assert buffer.grad_data.storage().size() == BUFFER_NUMEL and torch.count_nonzero(buffer.grad_data) == 0


@cuda_only
def test_non_ddp_chunk_moves_every_parameter():
    chunk = torch.nn.Linear(512, 256, bias=False).cuda()
    weight = chunk.weight.data.clone()
    host = HostState()
    resident = _allocated()
    released = offload_to_cpu([chunk], None, host, keep_gradients=False)
    assert released == weight.numel() * weight.element_size() and resident - _allocated() == released
    assert all(p.device.type == "cpu" for p in chunk.parameters())
    reload_to_cuda([chunk], host)
    assert _allocated() == resident
    assert chunk.weight.device.type == "cuda" and torch.equal(chunk.weight.data, weight)


@cuda_only
def test_non_distributed_mixed_precision_masters_move_too():
    # Float16OptimizerWithFloat16Params keeps its fp32 masters in fp32_from_float16_groups.
    master = torch.randn(MASTER_NUMEL, device="cuda")
    state = {master: {"exp_avg": torch.zeros_like(master)}}
    optimizer = SimpleNamespace(fp32_from_float16_groups=[[master]], optimizer=SimpleNamespace(state=state))
    host = HostState()
    resident = _allocated()
    released = offload_to_cpu([], optimizer, host, keep_gradients=False)
    assert master.device.type == "cpu" and released == 2 * master.numel() * master.element_size()
    assert resident - _allocated() == released
    reload_to_cuda([], host)
    assert master.device.type == "cuda" and _allocated() == resident


@pytest.mark.parametrize(
    ("section", "flag"),
    [
        ("ddp", "overlap_param_gather"),
        ("optimizer", "overlap_param_gather"),
        ("ddp", "reuse_grad_buf_for_mxfp8_param_ag"),
        ("ddp", "nccl_ub"),
        ("optimizer", "optimizer_cpu_offload"),
        ("optimizer", "use_precision_aware_optimizer"),
    ],
)
def test_check_movable_rejects_each_unsupported_mode(section, flag):
    recipe = SimpleNamespace(ddp=SimpleNamespace(), optimizer=SimpleNamespace())
    check_movable(recipe, None)
    setattr(getattr(recipe, section), flag, True)
    with pytest.raises(NotImplementedError, match=flag):
        check_movable(recipe, None)
