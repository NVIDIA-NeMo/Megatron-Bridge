# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Muse Spark Open reuses the Muse family multimodal SFT collator."""

import pytest

from megatron.bridge.data.collators.registry import resolve_model_collate
from megatron.bridge.models.muse_glimmer.data.collate_fn import muse_glimmer_collate_fn


pytestmark = pytest.mark.unit


def test_muse_spark_processor_resolves_muse_family_collator():
    assert resolve_model_collate("MuseSparkProcessor") is muse_glimmer_collate_fn
