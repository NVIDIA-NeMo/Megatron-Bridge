"""Real Gloo row routing and checkpoint descriptors, without CUDA/TE dependencies."""

import datetime
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "src/megatron/bridge/models/qwen/modeling_qwen4_exp/ep_ngram_embedding.py"


def _module():
    spec = importlib.util.spec_from_file_location("ep_ngram_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_imports():
    """Load real Core checkpoint code without its unrelated CUDA package initializer."""
    try:
        importlib.import_module("megatron.core.dist_checkpointing.mapping")
        return
    except ModuleNotFoundError:
        if "megatron.core" in sys.modules:
            # Never replace a live Core package or conceal its incomplete installation.
            raise
    root = Path(os.environ.get("MCORE_TEST_ROOT", ROOT / "3rdparty/Megatron-LM"))
    if not (root / "megatron/core/dist_checkpointing/mapping.py").exists():
        raise RuntimeError("Initialize the Core submodule or set MCORE_TEST_ROOT for checkpoint metadata tests")
    for name, relative in [
        ("megatron", "megatron"),
        ("megatron.core", "megatron/core"),
        ("megatron.core.dist_checkpointing", "megatron/core/dist_checkpointing"),
    ]:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root / relative)]
            sys.modules[name] = package


def _config(**overrides):
    return SimpleNamespace(
        **dict(
            expert_tensor_parallel_size=1,
            params_dtype=torch.float32,
            use_cpu_initialization=True,
            perform_initialization=True,
        )
        | overrides
    )


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=datetime.timedelta(seconds=45)
    )
    try:
        _checkpoint_imports()
        cls = _module().FrozenEPNgramEmbedding
        # Folded dense TP2/EP2: the two TP ranks own DIFFERENT row shards.
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3], [0, 2], [1, 3])]
        ep = groups[rank // 2]
        edp = groups[2 + rank % 2]
        torch.manual_seed(781)
        before = torch.get_rng_state().clone()
        model = cls(_config(), 12, 3, init_method=torch.nn.init.normal_, ep_group=ep, expt_dp_group=edp)
        assert torch.equal(torch.get_rng_state(), before)
        assert not model.weight.requires_grad and model.weight.allreduce is False
        assert list(model.state_dict()) == ["weight"]
        initial = [torch.empty_like(model.weight) for _ in range(4)]
        dist.all_gather(initial, model.weight)
        assert torch.equal(initial[0], initial[2]) and torch.equal(initial[1], initial[3])
        assert not torch.equal(initial[0], initial[1])
        for dtype in (torch.float32, torch.bfloat16):
            full = torch.arange(36, dtype=dtype).view(12, 3)
            full[0, 0] = -0.0
            model = cls(
                _config(params_dtype=dtype), 12, 3, init_method=torch.nn.init.normal_, ep_group=ep, expt_dp_group=edp
            )
            with torch.no_grad():
                model.weight.copy_(full[model.row_start : model.row_end])
            ids_cases = [
                torch.tensor([[0, 5, 6, 11], [6, 6, rank, 0]], dtype=torch.int64).t(),
                torch.empty((0, 2), dtype=torch.int64)
                if rank % 2 == 0
                else torch.tensor([[11, 0]], dtype=torch.int64),
                torch.empty((2, 0), dtype=torch.int64),
                torch.tensor(rank, dtype=torch.int64),
                # Caller slices positions AFTER full-sequence hashing; no TP reduction here.
                torch.arange(12, dtype=torch.int64).reshape(1, 6, 2)[:, (rank % 2) * 3 : (rank % 2 + 1) * 3],
            ]
            for ids in ids_cases:
                result = model(ids)
                expected = full[ids]
                assert result.shape == expected.shape and result.dtype == dtype
                assert torch.equal(result.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
                assert not result.requires_grad
                assert torch.equal(model(ids), result)
            # Frozen lookup still supplies the exact operand to trainable dense PLE.
            ids = ids_cases[0]
            dense = torch.nn.Linear(3, 2, bias=False, dtype=dtype)
            output = dense(model(ids))
            output.float().sum().backward()
            expected_grad = full[ids].reshape(-1, 3).sum(0).expand(2, 3)
            assert torch.equal(dense.weight.grad, expected_grad)
            assert model.weight.grad is None
            prefix = "decoder.layers.1.ple.ngram_embedding."
            descriptor = model.sharded_state_dict(prefix)[prefix + "weight"]
            descriptor.validate_metadata_integrity()
            assert descriptor.key == prefix + "weight"
            assert descriptor.global_shape == (12, 3)
            assert descriptor.local_shape == (6, 3)
            assert descriptor.global_offset == (rank % 2 * 6, 0)
            assert descriptor.axis_fragmentations == (2, 1)
            assert descriptor.replica_id == (0, 0, rank // 2)
            assert descriptor.data is model.weight
            stacked = model.sharded_state_dict(prefix, ((0, 1, 4),))[prefix + "weight"]
            assert stacked.global_shape == (4, 12, 3)
            assert stacked.global_offset == (1, rank % 2 * 6, 0)
            rows = [None] * 4
            dist.all_gather_object(
                rows, (descriptor.global_offset, descriptor.replica_id, model.weight.detach().clone())
            )
            rebuilt = torch.cat([r[2] for r in rows if r[1] == (0, 0, 0)])
            assert torch.equal(rebuilt.view(torch.uint8), full.view(torch.uint8))
        # One invalid participant rejects both members of its EP group at the same boundary.
        for ids in (
            torch.tensor([12 if rank % 2 == 0 else 0]),
            torch.tensor([0.0]) if rank % 2 == 0 else torch.tensor([0]),
        ):
            with pytest.raises(ValueError, match="Frozen EP lookup"):
                model(ids)
        model.weight.requires_grad_(rank % 2 == 0)
        with pytest.raises(ValueError, match="Frozen EP lookup"):
            model(torch.tensor([0]))
        model.weight.requires_grad_(False)
        assert torch.equal(model(torch.tensor([0])), full[[0]])
        for config, rows in ((_config(expert_tensor_parallel_size=2), 12), (_config(), 11), (_config(), 0)):
            with pytest.raises(ValueError):
                cls(config, rows, 3, init_method=torch.nn.init.normal_, ep_group=ep, expt_dp_group=edp)
        solo = [dist.new_group([r]) for r in range(4)]
        single = cls(
            _config(), 12, 3, init_method=torch.nn.init.normal_, ep_group=solo[rank], expt_dp_group=dist.group.WORLD
        )
        with torch.no_grad():
            single.weight.copy_(full.float())
        ids = torch.tensor([11, 0, 11])
        assert torch.equal(single(ids), full.float()[ids])
        assert single(torch.empty(0, dtype=torch.int64)).shape == (0, 3)
        with pytest.raises(ValueError, match="prepended"):
            single.sharded_state_dict(sharded_offsets=((1, 0, 2),))
        before = torch.get_rng_state().clone()

        def failed_init(weight):
            torch.nn.init.normal_(weight)
            raise RuntimeError("initialization failure")

        with pytest.raises(RuntimeError, match="initialization failure"):
            cls(_config(), 12, 3, init_method=failed_init, ep_group=ep, expt_dp_group=edp)
        assert torch.equal(torch.get_rng_state(), before)
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_ep_row_routing_and_checkpoint_metadata(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4, join=True)


@pytest.mark.unit
def test_explicit_groups_required():
    with pytest.raises(ValueError, match="explicit initialized"):
        _module().FrozenEPNgramEmbedding(
            _config(), 12, 3, init_method=torch.nn.init.normal_, ep_group=None, expt_dp_group=None
        )
