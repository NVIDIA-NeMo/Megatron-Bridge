import importlib

import pytest
from megatron.core.distributed.fsdp import mcore_fsdp_adapter

from megatron.bridge.utils import mcore_fsdp
from megatron.bridge.utils.import_utils import UnavailableError


@pytest.fixture
def adapter_names(monkeypatch):
    with monkeypatch.context() as patch:
        yield patch
    importlib.reload(mcore_fsdp)


@pytest.mark.unit
def test_legacy_fsdp_keeps_v1_identity_and_rejects_v2(adapter_names):
    class LegacyFSDP:
        pass

    adapter_names.delattr(mcore_fsdp_adapter, "FullyShardedDataParallelV1", raising=False)
    adapter_names.delattr(mcore_fsdp_adapter, "FullyShardedDataParallelV2", raising=False)
    adapter_names.setattr(mcore_fsdp_adapter, "FullyShardedDataParallel", LegacyFSDP, raising=False)
    wrappers = importlib.reload(mcore_fsdp)

    assert wrappers.FullyShardedDataParallelV1 is LegacyFSDP
    assert isinstance(LegacyFSDP(), wrappers.FullyShardedDataParallelV1)
    assert not isinstance(LegacyFSDP(), wrappers.FullyShardedDataParallelV2)
    with pytest.raises(UnavailableError, match="Megatron-FSDP v2 requires"):
        wrappers.FullyShardedDataParallelV2()


@pytest.mark.unit
def test_modern_fsdp_wrappers_preserve_class_identity(adapter_names):
    class FSDPv1:
        pass

    class FSDPv2:
        pass

    adapter_names.setattr(mcore_fsdp_adapter, "FullyShardedDataParallelV1", FSDPv1, raising=False)
    adapter_names.setattr(mcore_fsdp_adapter, "FullyShardedDataParallelV2", FSDPv2, raising=False)
    wrappers = importlib.reload(mcore_fsdp)

    assert wrappers.FullyShardedDataParallelV1 is FSDPv1
    assert wrappers.FullyShardedDataParallelV2 is FSDPv2
    assert not isinstance(FSDPv1(), wrappers.FullyShardedDataParallelV2)
    assert isinstance(FSDPv2(), wrappers.FullyShardedDataParallelV2)
