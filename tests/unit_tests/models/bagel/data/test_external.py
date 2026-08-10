from unittest.mock import Mock

import pytest
import torch
from megatron.core.rerun_state_machine import RerunDataIterator

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.models.bagel.data.dataset import _restore_loader
from megatron.bridge.models.bagel.data.external import BagelExternalLoader, BagelMegatronMIMODatasetProvider


pytestmark = pytest.mark.unit


class _StatefulRows:
    def __init__(self) -> None:
        self.position = 0

    def save_state_rank(self) -> dict[str, int]:
        return {"position": self.position}

    def restore_state_rank(self, state: dict[str, int]) -> None:
        self.position = state["position"]


class _StatefulBatches:
    def __init__(self, rows: _StatefulRows) -> None:
        self.rows = rows
        self.position = 0

    def __iter__(self) -> "_StatefulBatches":
        return self

    def __next__(self) -> dict[str, object]:
        batch = {"step": self.position, "row": self.rows.position}
        self.position += 1
        self.rows.position += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = state["position"]


def test_bagel_external_loader_bypasses_megatron_sampler_and_collate() -> None:
    batches = [{"step": 0}, {"step": 1}]
    loader = BagelExternalLoader(iter(batches), length=2)
    provider = BagelMegatronMIMODatasetProvider(train_loader=loader)
    train, valid, test = provider.build_datasets(DatasetBuildContext(train_samples=2, valid_samples=0, test_samples=0))

    external = build_pretraining_data_loader(
        train,
        consumed_samples=0,
        dataloader_type="external",
        micro_batch_size=1,
        num_workers=0,
        data_sharding=False,
    )
    rerun = RerunDataIterator(external)

    assert external is loader
    assert len(loader) == 2
    assert next(rerun) is batches[0]
    assert next(rerun) is batches[1]
    assert valid is None
    assert test is None
    with pytest.raises(RuntimeError, match="must not be collated"):
        provider.get_collate_fn()([])


def test_bagel_external_loader_restores_reader_and_packer_state() -> None:
    rows = _StatefulRows()
    loader = BagelExternalLoader(_StatefulBatches(rows), length=3, stateful_loaders=[rows])
    assert next(loader) == {"step": 0, "row": 0}
    state = loader.save_state()
    expected = next(loader)

    restored_rows = _StatefulRows()
    restored = BagelExternalLoader(
        _StatefulBatches(restored_rows),
        length=3,
        stateful_loaders=[restored_rows],
    )
    restored.restore_state(state)

    assert next(restored) == expected


def test_bagel_dataset_restores_latest_training_dataloader_state(tmp_path) -> None:
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("0", encoding="utf-8")
    loader = Mock()
    _restore_loader(loader, str(tmp_path), 3)
    loader.restore_state.assert_not_called()

    tracker.write_text("2", encoding="utf-8")
    state_path = tmp_path / "iter_0000002" / "train_dataloader_dprank003.pt"
    state_path.parent.mkdir()
    torch.save({"dataloader_state_dict": {"position": 2}}, state_path)
    _restore_loader(loader, str(tmp_path), 3)
    loader.restore_state.assert_called_once_with({"position": 2})
