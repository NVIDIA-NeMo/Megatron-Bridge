# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for examples/models/flux/inference_flux.py."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from argparse import Namespace
from unittest.mock import MagicMock

import pytest


pytestmark = [pytest.mark.unit]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CLI_PATH = _REPO_ROOT / "examples" / "models" / "flux" / "inference_flux.py"


@pytest.fixture(scope="module")
def cli():
    """Load the FLUX inference script as a module under a stable test name."""
    spec = importlib.util.spec_from_file_location("flux_inference_under_test", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_main_passes_base_seed_to_pipeline_generator(cli, monkeypatch, tmp_path):
    """The documented base seed controls the generator used for latent sampling."""
    generation_kwargs = {}

    class FakeGenerator:
        def manual_seed(self, seed):
            self.seed = seed
            return self

        def initial_seed(self):
            return self.seed

    class FakePipeline:
        device = "cuda:0"

        def __init__(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            generation_kwargs.update(kwargs)
            return []

    args = Namespace(
        flux_ckpt="/fake/flux",
        vae_ckpt="/fake/vae",
        t5_version="fake-t5",
        clip_version="fake-clip",
        prompts=["a reproducible image"],
        height=64,
        width=64,
        num_inference_steps=1,
        guidance_scale=0.0,
        output_path=str(tmp_path),
        base_seed=17,
    )
    fake_flux_module = types.ModuleType("megatron.bridge.diffusion.models.flux")
    fake_flux_module.FluxInferencePipeline = FakePipeline
    generator_factory = MagicMock(return_value=FakeGenerator())

    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(cli.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(cli.torch, "Generator", generator_factory)
    monkeypatch.setitem(sys.modules, "megatron.bridge.diffusion.models.flux", fake_flux_module)

    cli.main()

    generator_factory.assert_called_once_with(device="cuda:0")
    assert generation_kwargs["generator"].initial_seed() == 17
