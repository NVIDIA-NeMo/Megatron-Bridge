# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the builder-backed Sarvam MoE model config."""

from unittest.mock import patch

import pytest
from transformers import PretrainedConfig

from megatron.bridge.models.sarvam.sarvam_config import SarvamMoEModelConfig, _get_sarvam_moe_pipeline_layout
from megatron.bridge.models.sarvam.sarvam_moe_bridge import SarvamMoEBridge
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.model_load_save import build_and_load_model, load_model_config


def _model_config(**overrides) -> SarvamMoEModelConfig:
    """Map a Sarvam 30B-shaped Hugging Face config to its builder config."""
    values = dict(
        architectures=["SarvamMoEForCausalLM"],
        num_hidden_layers=19,
        hidden_size=4096,
        intermediate_size=8192,
        moe_intermediate_size=1024,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=64,
        num_experts=128,
        num_experts_per_tok=6,
        num_shared_experts=1,
        first_k_dense_replace=1,
        vocab_size=262144,
        max_position_embeddings=131072,
        rope_theta=8_000_000,
        torch_dtype="float32",
    )
    values.update(overrides)
    return SarvamMoEBridge().hf_config_to_model_config(PretrainedConfig(**values))


def _save_run_config(model_config: SarvamMoEModelConfig, directory) -> None:
    """Write the model config as a checkpoint's run_config.yaml."""
    ConfigContainer(
        model=model_config,
        train=None,
        optimizer=None,
        scheduler=None,
        dataset=None,
        logger=None,
        tokenizer=None,
        checkpoint=None,
    ).to_yaml(str(directory / "run_config.yaml"))


class TestSarvamMoEModelConfigFinalize:
    """SarvamMoEModelConfig.finalize() applies the provider's uneven pipeline layouts."""

    @pytest.mark.parametrize("pp", [2, 4, 8])
    def test_finalize_sets_pipeline_layout_when_pp_uneven(self, pp):
        model_config = _model_config()
        model_config.pipeline_model_parallel_size = pp

        model_config.finalize()

        assert model_config.pipeline_model_parallel_layout.input_data == _get_sarvam_moe_pipeline_layout(pp)
        assert model_config.num_layers_in_first_pipeline_stage is None

    def test_finalize_no_change_when_pp_is_one(self):
        model_config = _model_config()

        model_config.finalize()

        assert model_config.pipeline_model_parallel_layout is None

    def test_finalize_no_change_when_pp_evenly_divides(self):
        model_config = _model_config(num_hidden_layers=20)
        model_config.pipeline_model_parallel_size = 4

        model_config.finalize()

        assert model_config.pipeline_model_parallel_layout is None

    def test_finalize_raises_for_unsupported_uneven_pp(self):
        model_config = _model_config()
        model_config.pipeline_model_parallel_size = 3

        with pytest.raises(ValueError, match="Unsupported PP size 3"):
            model_config.finalize()

    def test_finalize_respects_explicit_pipeline_layout(self):
        custom_layout = [["embedding"] + ["decoder"] * 7, ["decoder"] * 6, ["decoder"] * 6 + ["loss"]]
        model_config = _model_config()
        model_config.pipeline_model_parallel_size = 3
        model_config.pipeline_model_parallel_layout = custom_layout

        model_config.finalize()

        assert model_config.pipeline_model_parallel_layout.input_data == custom_layout


class TestSarvamMoEModelConfigCheckpoint:
    """Saved run configs restore the Sarvam MoE builder config and its pipeline layout."""

    @pytest.mark.parametrize("pp", [1, 2])
    def test_run_config_round_trip(self, tmp_path, pp):
        model_config = _model_config()
        model_config.pipeline_model_parallel_size = pp
        model_config.finalize()
        expected_layout = _get_sarvam_moe_pipeline_layout(pp)

        _save_run_config(model_config, tmp_path)
        loaded, _ = load_model_config(str(tmp_path))

        assert type(loaded) is SarvamMoEModelConfig
        assert loaded.pipeline_model_parallel_layout == expected_layout
        assert loaded.moe_layer_freq == model_config.moe_layer_freq
        loaded.finalize()
        if expected_layout is not None:
            assert loaded.pipeline_model_parallel_layout.input_data == expected_layout

    @pytest.mark.parametrize("saved_cpu_init", [True, False])
    @pytest.mark.parametrize("use_cpu_init", [False, True])
    def test_load_builds_on_requested_device(self, tmp_path, saved_cpu_init, use_cpu_init):
        """The load request, not the flag a CPU import saved, decides where the model is built."""
        model_config = _model_config()
        # AutoBridge.import_ckpt() builds on CPU and saves use_cpu_initialization=True.
        model_config.transformer.use_cpu_initialization = saved_cpu_init
        _save_run_config(model_config, tmp_path)
        loaded, _ = load_model_config(str(tmp_path))
        assert loaded.transformer.use_cpu_initialization is saved_cpu_init

        model = [object()]
        cpu_init_at_build = []

        def build_distributed_models(*args, **kwargs):
            cpu_init_at_build.append(loaded.transformer.use_cpu_initialization)
            return model

        with (
            patch.object(loaded.get_builder_cls(), "build_distributed_models", build_distributed_models),
            patch("megatron.bridge.training.model_load_save._get_or_initialize_pg_collection"),
            patch("megatron.bridge.training.checkpointing._load_model_weights_from_checkpoint"),
        ):
            result = build_and_load_model(
                str(tmp_path), loaded, use_cpu_init=use_cpu_init, skip_temp_dist_context=True
            )

        assert result is model
        assert cpu_init_at_build == [use_cpu_init]
