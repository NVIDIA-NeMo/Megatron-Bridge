import pytest
from megatron.core.rerun_state_machine import RerunDataIterator

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.models.bagel.data.external import BagelExternalLoader, BagelMegatronMIMODatasetProvider


pytestmark = pytest.mark.unit


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
