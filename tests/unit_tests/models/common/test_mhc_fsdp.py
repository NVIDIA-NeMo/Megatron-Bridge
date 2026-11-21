import pytest
from megatron.training.models.dist_utils import _ddp_wrap

from megatron.bridge.models.common import unimodal


@pytest.mark.unit
def test_mhc_wrapping_uses_core_entry_point():
    """Core owns mHC unit collection, expert sharding, and replay boundaries."""
    assert unimodal._ddp_wrap is _ddp_wrap
