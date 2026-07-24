# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import importlib

import pytest


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "module_name",
    [
        "megatron.bridge.data.builders.direct_hf_sft_dataset",
        "megatron.bridge.data.builders.gpt_sft_dataset",
        "megatron.bridge.data.datasets.sft",
        "megatron.bridge.data.hf_datasets",
        "megatron.bridge.data.hf_source",
        "megatron.bridge.data.sources.local_conversation",
        "megatron.bridge.data.vlm_datasets",
    ],
)
def test_removed_internal_data_modules_have_no_compatibility_shims(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
