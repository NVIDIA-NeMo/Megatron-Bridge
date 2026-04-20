# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for build_omni_modal_data_loaders."""

from types import SimpleNamespace

import pytest

from megatron.bridge.data.omni_modal.loaders import build_omni_modal_data_loaders


class FakeOmniModalProvider:
    def __init__(self, omni_modal_parallelism_config, grids=None):
        self.omni_modal_parallelism_config = omni_modal_parallelism_config
        self._grids = grids


class FakeProvider:
    num_workers = 0
    pin_memory = False
    drop_last = True

    def __init__(self):
        self.built = False

    def build_datasets(self, context):
        self.built = True
        del context
        return [1, 2], [3], [4]

    def get_collate_fn(self):
        return lambda batch: batch


def _patch_omni_modal_provider_class(monkeypatch):
    monkeypatch.setattr(
        "megatron.bridge.models.omni_modal.omni_modal_provider.OmniModalProvider",
        FakeOmniModalProvider,
    )


def test_build_omni_modal_data_loaders_raises_when_model_not_omni_modal(monkeypatch):
    _patch_omni_modal_provider_class(monkeypatch)
    cfg = SimpleNamespace(model=object(), train=SimpleNamespace(micro_batch_size=2))
    provider = FakeProvider()

    with pytest.raises(ValueError, match="cfg.model must be OmniModalProvider"):
        build_omni_modal_data_loaders(
            cfg, train_state=None, omni_modal_provider=provider, train_samples=4, valid_samples=2, test_samples=2
        )


def test_build_omni_modal_data_loaders_raises_when_parallelism_missing(monkeypatch):
    _patch_omni_modal_provider_class(monkeypatch)
    cfg = SimpleNamespace(
        model=FakeOmniModalProvider(omni_modal_parallelism_config=None, grids={"llm": object()}),
        train=SimpleNamespace(micro_batch_size=2),
    )
    provider = FakeProvider()

    with pytest.raises(ValueError, match="omni_modal_parallelism_config must be set"):
        build_omni_modal_data_loaders(
            cfg, train_state=None, omni_modal_provider=provider, train_samples=4, valid_samples=2, test_samples=2
        )


def test_build_omni_modal_data_loaders_raises_when_grids_missing(monkeypatch):
    _patch_omni_modal_provider_class(monkeypatch)
    cfg = SimpleNamespace(
        model=FakeOmniModalProvider(omni_modal_parallelism_config=object(), grids=None),
        train=SimpleNamespace(micro_batch_size=2),
    )
    provider = FakeProvider()

    with pytest.raises(ValueError, match="_grids is None"):
        build_omni_modal_data_loaders(
            cfg, train_state=None, omni_modal_provider=provider, train_samples=4, valid_samples=2, test_samples=2
        )


def test_build_omni_modal_data_loaders_happy_path(monkeypatch):
    _patch_omni_modal_provider_class(monkeypatch)
    fake_grids = {"llm": object()}
    fake_parallelism_config = SimpleNamespace(
        module_parallelisms={"llm": SimpleNamespace(data_parallel_size=1)},
    )
    cfg = SimpleNamespace(
        model=FakeOmniModalProvider(omni_modal_parallelism_config=fake_parallelism_config, grids=fake_grids),
        train=SimpleNamespace(micro_batch_size=3),
    )
    provider = FakeProvider()

    monkeypatch.setattr(
        "megatron.bridge.data.omni_modal.loaders.get_omni_modal_sampling_info",
        lambda omni_modal_cfg, grids: (1, 4, True),
    )

    monkeypatch.setattr(
        "megatron.bridge.data.omni_modal.loaders.print_rank_0",
        lambda *args, **kwargs: None,
    )

    sampler_calls = []

    class _Sampler:
        def __init__(self, dataset, num_replicas, rank, shuffle):
            sampler_calls.append(
                {
                    "dataset": dataset,
                    "num_replicas": num_replicas,
                    "rank": rank,
                    "shuffle": shuffle,
                }
            )

    monkeypatch.setattr("megatron.bridge.data.omni_modal.loaders.torch.utils.data.DistributedSampler", _Sampler)

    loader_calls = []

    def _fake_dataloader(
        dataset,
        batch_size,
        sampler,
        num_workers,
        collate_fn,
        pin_memory,
        drop_last,
    ):
        loader_calls.append(
            {
                "dataset": dataset,
                "batch_size": batch_size,
                "sampler": sampler,
                "num_workers": num_workers,
                "collate_fn": collate_fn,
                "pin_memory": pin_memory,
                "drop_last": drop_last,
            }
        )
        return f"loader-{len(loader_calls)}"

    monkeypatch.setattr("megatron.bridge.data.omni_modal.loaders.DataLoader", _fake_dataloader)

    train_loader, valid_loader, test_loader = build_omni_modal_data_loaders(
        cfg,
        train_state=None,
        omni_modal_provider=provider,
        train_samples=10,
        valid_samples=4,
        test_samples=2,
    )

    assert provider.built is True
    assert (train_loader, valid_loader, test_loader) == ("loader-1", "loader-2", "loader-3")
    assert len(sampler_calls) == 3
    assert sampler_calls[0]["shuffle"] is True
    assert sampler_calls[1]["shuffle"] is False
    assert sampler_calls[2]["shuffle"] is False
    assert all(call["num_replicas"] == 4 for call in sampler_calls)
    assert all(call["rank"] == 1 for call in sampler_calls)
    assert len(loader_calls) == 3
    assert all(call["batch_size"] == 3 for call in loader_calls)
    assert all(call["drop_last"] is True for call in loader_calls)


def test_build_omni_modal_data_loaders_skips_non_data_ranks(monkeypatch):
    _patch_omni_modal_provider_class(monkeypatch)
    cfg = SimpleNamespace(
        model=FakeOmniModalProvider(
            omni_modal_parallelism_config=SimpleNamespace(
                module_parallelisms={"llm": SimpleNamespace(data_parallel_size=1)},
            ),
            grids={"llm": object()},
        ),
        train=SimpleNamespace(micro_batch_size=2),
    )
    provider = FakeProvider()
    monkeypatch.setattr(
        "megatron.bridge.data.omni_modal.loaders.get_omni_modal_sampling_info",
        lambda omni_modal_cfg, grids: (0, 1, False),
    )

    monkeypatch.setattr(
        "megatron.bridge.data.omni_modal.loaders.print_rank_0",
        lambda *args, **kwargs: None,
    )

    train_loader, valid_loader, test_loader = build_omni_modal_data_loaders(
        cfg,
        train_state=None,
        omni_modal_provider=provider,
        train_samples=10,
        valid_samples=4,
        test_samples=2,
    )

    assert (train_loader, valid_loader, test_loader) == (None, None, None)
    assert provider.built is False
