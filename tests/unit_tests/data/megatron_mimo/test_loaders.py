# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for build_megatron_mimo_data_loaders."""

from types import SimpleNamespace

import pytest
from torch.utils.data import Dataset

from megatron.bridge.data.megatron_mimo.loaders import build_megatron_mimo_data_loaders


class FakeMegatronMIMOProvider:
    def __init__(self, megatron_mimo_parallelism_config, grids=None):
        self.megatron_mimo_parallelism_config = megatron_mimo_parallelism_config
        self._grids = grids


class FakeDataset:
    def __init__(self, size: int):
        self._size = size

    def __len__(self) -> int:
        return self._size


class FakeProvider:
    num_workers = 0
    pin_memory = False
    drop_last = True
    dataloader_type = "single"
    data_sharding = False
    persistent_workers = False

    def __init__(self):
        self.built = False

    def build_datasets(self, context):
        self.built = True
        del context
        return FakeDataset(12), FakeDataset(6), FakeDataset(6)

    def get_collate_fn(self):
        return lambda batch: batch


def _patch_megatron_mimo_provider_class(monkeypatch):
    monkeypatch.setattr(
        "megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MegatronMIMOProvider",
        FakeMegatronMIMOProvider,
    )


def test_build_megatron_mimo_data_loaders_raises_when_model_not_megatron_mimo(monkeypatch):
    _patch_megatron_mimo_provider_class(monkeypatch)
    cfg = SimpleNamespace(model=object(), train=SimpleNamespace(micro_batch_size=2))
    provider = FakeProvider()

    with pytest.raises(ValueError, match="cfg.model must be MegatronMIMOProvider"):
        build_megatron_mimo_data_loaders(
            cfg, train_state=None, megatron_mimo_provider=provider, train_samples=4, valid_samples=2, test_samples=2
        )


def test_build_megatron_mimo_data_loaders_raises_when_parallelism_missing(monkeypatch):
    _patch_megatron_mimo_provider_class(monkeypatch)
    cfg = SimpleNamespace(
        model=FakeMegatronMIMOProvider(megatron_mimo_parallelism_config=None, grids={"llm": object()}),
        train=SimpleNamespace(micro_batch_size=2),
    )
    provider = FakeProvider()

    with pytest.raises(ValueError, match="megatron_mimo_parallelism_config must be set"):
        build_megatron_mimo_data_loaders(
            cfg, train_state=None, megatron_mimo_provider=provider, train_samples=4, valid_samples=2, test_samples=2
        )


def test_build_megatron_mimo_data_loaders_raises_when_grids_missing(monkeypatch):
    _patch_megatron_mimo_provider_class(monkeypatch)
    cfg = SimpleNamespace(
        model=FakeMegatronMIMOProvider(megatron_mimo_parallelism_config=object(), grids=None),
        train=SimpleNamespace(micro_batch_size=2),
    )
    provider = FakeProvider()

    with pytest.raises(ValueError, match="_grids is None"):
        build_megatron_mimo_data_loaders(
            cfg, train_state=None, megatron_mimo_provider=provider, train_samples=4, valid_samples=2, test_samples=2
        )


def _patch_happy_path_dependencies(monkeypatch):
    """Shared monkeypatches for happy-path tests.

    Stubs ``get_megatron_mimo_sampling_info`` (returning dp_rank=1, dp_size=4,
    needs_data=True) and ``build_pretraining_data_loader`` so we can assert on
    the arguments MegatronMIMO's builder passes through to the shared helper.
    """
    _patch_megatron_mimo_provider_class(monkeypatch)

    monkeypatch.setattr(
        "megatron.bridge.data.megatron_mimo.loaders.get_megatron_mimo_sampling_info",
        lambda megatron_mimo_cfg, grids: (1, 4, True),
    )
    monkeypatch.setattr(
        "megatron.bridge.data.megatron_mimo.loaders.print_rank_0",
        lambda *args, **kwargs: None,
    )

    builder_calls = []

    def _fake_build_loader(**kwargs):
        builder_calls.append(kwargs)
        return f"loader-{len(builder_calls)}"

    monkeypatch.setattr(
        "megatron.bridge.data.megatron_mimo.loaders.build_pretraining_data_loader",
        _fake_build_loader,
    )
    return builder_calls


def _make_happy_cfg(micro_batch_size: int = 3):
    fake_grids = {"llm": object()}
    fake_parallelism_config = SimpleNamespace(
        module_parallelisms={"llm": SimpleNamespace(data_parallel_size=1)},
    )
    return SimpleNamespace(
        model=FakeMegatronMIMOProvider(megatron_mimo_parallelism_config=fake_parallelism_config, grids=fake_grids),
        train=SimpleNamespace(micro_batch_size=micro_batch_size),
    )


def test_build_megatron_mimo_data_loaders_happy_path(monkeypatch):
    builder_calls = _patch_happy_path_dependencies(monkeypatch)
    cfg = _make_happy_cfg(micro_batch_size=3)
    provider = FakeProvider()

    train_state = SimpleNamespace(consumed_train_samples=0)
    train_loader, valid_loader, test_loader = build_megatron_mimo_data_loaders(
        cfg,
        train_state=train_state,
        megatron_mimo_provider=provider,
        train_samples=10,
        valid_samples=4,
        test_samples=2,
    )

    assert provider.built is True
    assert (train_loader, valid_loader, test_loader) == ("loader-1", "loader-2", "loader-3")
    assert len(builder_calls) == 3
    # sampler_dp_rank/sampler_dp_size from the mocked get_megatron_mimo_sampling_info.
    assert all(c["data_parallel_rank"] == 1 for c in builder_calls)
    assert all(c["data_parallel_size"] == 4 for c in builder_calls)
    assert all(c["micro_batch_size"] == 3 for c in builder_calls)
    # Provider fields forwarded verbatim.
    assert all(c["dataloader_type"] == "single" for c in builder_calls)
    assert all(c["drop_last"] is True for c in builder_calls)
    assert all(c["num_workers"] == 0 for c in builder_calls)
    assert all(c["pin_memory"] is False for c in builder_calls)


def test_build_megatron_mimo_data_loaders_wires_consumed_samples_for_resume(monkeypatch):
    """Regression test for issue #11: train loader must receive
    train_state.consumed_train_samples so data resumes after checkpoint load;
    valid/test loaders always start from 0.
    """
    builder_calls = _patch_happy_path_dependencies(monkeypatch)
    cfg = _make_happy_cfg(micro_batch_size=2)
    provider = FakeProvider()

    train_state = SimpleNamespace(consumed_train_samples=50000)
    build_megatron_mimo_data_loaders(
        cfg,
        train_state=train_state,
        megatron_mimo_provider=provider,
        train_samples=10,
        valid_samples=4,
        test_samples=2,
    )

    # Train loader gets resume offset; valid/test always start from 0.
    assert builder_calls[0]["consumed_samples"] == 50000
    assert builder_calls[1]["consumed_samples"] == 0
    assert builder_calls[2]["consumed_samples"] == 0


class IndexDataset(Dataset):
    """Dataset that returns its own index so tests can trace which samples got picked."""

    def __init__(self, size: int):
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> int:
        return idx


class IndexDatasetProvider:
    """Provider that yields IndexDataset for train/valid/test.

    Used by the end-to-end sampler test below (no ``build_pretraining_data_loader``
    monkeypatch) so the real ``MegatronPretrainingSampler`` is exercised.
    """

    num_workers = 0
    pin_memory = False
    drop_last = True
    dataloader_type = "single"
    data_sharding = False
    persistent_workers = False

    def __init__(self, train_size: int, valid_size: int = 0, test_size: int = 0):
        self._train_size = train_size
        self._valid_size = valid_size
        self._test_size = test_size

    def build_datasets(self, context):
        del context

        def _maybe(sz):
            return IndexDataset(sz) if sz > 0 else None

        return _maybe(self._train_size), _maybe(self._valid_size), _maybe(self._test_size)

    def get_collate_fn(self):
        # Identity collate: DataLoader will still stack via default_collate, but we
        # only read batch contents as Python-list-of-ints.
        return lambda batch: batch


def _make_real_loader_cfg(monkeypatch, *, micro_batch_size: int, sampler_dp_rank: int = 0, sampler_dp_size: int = 1):
    """Shared setup for end-to-end loader tests (no build_pretraining_data_loader mock)."""
    _patch_megatron_mimo_provider_class(monkeypatch)
    monkeypatch.setattr(
        "megatron.bridge.data.megatron_mimo.loaders.get_megatron_mimo_sampling_info",
        lambda megatron_mimo_cfg, grids: (sampler_dp_rank, sampler_dp_size, True),
    )
    monkeypatch.setattr(
        "megatron.bridge.data.megatron_mimo.loaders.print_rank_0",
        lambda *args, **kwargs: None,
    )
    parallelism_config = SimpleNamespace(
        module_parallelisms={"llm": SimpleNamespace(data_parallel_size=1)},
    )
    return SimpleNamespace(
        model=FakeMegatronMIMOProvider(megatron_mimo_parallelism_config=parallelism_config, grids={"llm": object()}),
        train=SimpleNamespace(micro_batch_size=micro_batch_size),
    )


def test_train_loader_starts_from_consumed_samples_end_to_end(monkeypatch):
    """Issue #11 end-to-end: real MegatronPretrainingSampler skips already-seen samples.

    No monkeypatch on build_pretraining_data_loader — we want to exercise the real
    sampler selection (dataloader_type="single" → MegatronPretrainingSampler) and
    verify that when the checkpoint is loaded with consumed_train_samples=K, the
    first batch returned starts at sample index K (not 0).
    """
    micro_batch_size = 4
    train_size = 32
    consumed = 12  # pretend 3 iterations already ran before the crash
    cfg = _make_real_loader_cfg(monkeypatch, micro_batch_size=micro_batch_size)
    provider = IndexDatasetProvider(train_size=train_size)
    train_state = SimpleNamespace(consumed_train_samples=consumed)

    train_loader, _, _ = build_megatron_mimo_data_loaders(
        cfg,
        train_state=train_state,
        megatron_mimo_provider=provider,
        train_samples=train_size,
        valid_samples=0,
        test_samples=0,
    )

    # Collect the first batch the resumed loader produces.
    first_batch = next(iter(train_loader))
    expected_indices = list(range(consumed, consumed + micro_batch_size))
    assert first_batch == expected_indices, (
        f"Resumed loader should start at consumed_samples={consumed}, "
        f"got first batch {first_batch} (expected {expected_indices})"
    )


def test_valid_and_test_loaders_always_start_from_zero_end_to_end(monkeypatch):
    """Issue #11 end-to-end: valid/test loaders must start at 0 even when the train
    checkpoint carries a non-zero consumed_train_samples.
    """
    micro_batch_size = 2
    cfg = _make_real_loader_cfg(monkeypatch, micro_batch_size=micro_batch_size)
    provider = IndexDatasetProvider(train_size=16, valid_size=8, test_size=8)
    train_state = SimpleNamespace(consumed_train_samples=10)

    _, valid_loader, test_loader = build_megatron_mimo_data_loaders(
        cfg,
        train_state=train_state,
        megatron_mimo_provider=provider,
        train_samples=16,
        valid_samples=8,
        test_samples=8,
    )

    assert next(iter(valid_loader)) == [0, 1]
    assert next(iter(test_loader)) == [0, 1]


def test_build_megatron_mimo_data_loaders_forwards_dataloader_type(monkeypatch):
    """Provider's dataloader_type (e.g., "cyclic" for shuffled training)
    is forwarded to build_pretraining_data_loader unchanged.
    """
    builder_calls = _patch_happy_path_dependencies(monkeypatch)
    cfg = _make_happy_cfg()
    provider = FakeProvider()
    provider.dataloader_type = "cyclic"

    build_megatron_mimo_data_loaders(
        cfg,
        train_state=SimpleNamespace(consumed_train_samples=0),
        megatron_mimo_provider=provider,
        train_samples=10,
        valid_samples=4,
        test_samples=2,
    )

    assert all(c["dataloader_type"] == "cyclic" for c in builder_calls)


def test_build_megatron_mimo_data_loaders_skips_non_data_ranks(monkeypatch):
    _patch_megatron_mimo_provider_class(monkeypatch)
    cfg = SimpleNamespace(
        model=FakeMegatronMIMOProvider(
            megatron_mimo_parallelism_config=SimpleNamespace(
                module_parallelisms={"llm": SimpleNamespace(data_parallel_size=1)},
            ),
            grids={"llm": object()},
        ),
        train=SimpleNamespace(micro_batch_size=2),
    )
    provider = FakeProvider()
    monkeypatch.setattr(
        "megatron.bridge.data.megatron_mimo.loaders.get_megatron_mimo_sampling_info",
        lambda megatron_mimo_cfg, grids: (0, 1, False),
    )

    monkeypatch.setattr(
        "megatron.bridge.data.megatron_mimo.loaders.print_rank_0",
        lambda *args, **kwargs: None,
    )

    train_loader, valid_loader, test_loader = build_megatron_mimo_data_loaders(
        cfg,
        train_state=None,
        megatron_mimo_provider=provider,
        train_samples=10,
        valid_samples=4,
        test_samples=2,
    )

    assert (train_loader, valid_loader, test_loader) == (None, None, None)
    assert provider.built is False
