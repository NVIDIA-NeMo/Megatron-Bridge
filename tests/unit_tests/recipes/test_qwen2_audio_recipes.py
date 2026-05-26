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

#
# Test purpose:
# - Cover the previously untested qwen2_audio finetune recipe (issue #3177).
# - Monkeypatch AutoBridge to avoid HF Hub I/O.
# - Verify the entry point's full-SFT vs PEFT branch (default lr selection).
# - Sanity-check parallelism, dataset provider type, freeze flags, and
#   PEFT scheme propagation.
#

import importlib

import pytest


_qwen2_audio_module = importlib.import_module("megatron.bridge.recipes.qwen2_audio.qwen2_audio")


class _FakeAudioModelCfg:
    """Fake provider returned by AutoBridge.to_megatron_provider.

    The recipe sets a number of attributes on the provider (parallelism,
    freeze flags, seq_length). This stub mirrors the attribute surface so
    the recipe can mutate it without errors.
    """

    def __init__(self):
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.sequence_parallel = False
        self.freeze_language_model = False
        self.freeze_audio_model = False
        self.freeze_audio_projection = False
        self.seq_length = 4096
        # Recipes may interrogate vocab_size for tokenizer wiring; the
        # qwen2_audio recipe uses DEFAULT_NULL_TOKENIZER_VOCAB_SIZE so this
        # is only here to keep stub-shape close to the real provider.
        self.vocab_size = 152000

    def finalize(self):
        return None


class _FakeAutoBridge:
    """AutoBridge stub that bypasses HF Hub network access."""

    @classmethod
    def from_hf_pretrained(cls, *args, **kwargs):
        return cls()

    def to_megatron_provider(self, *args, **kwargs):
        return _FakeAudioModelCfg()


@pytest.fixture(autouse=True)
def _patch_autobridge(monkeypatch):
    """Monkeypatch AutoBridge in the qwen2_audio recipe module to avoid HF I/O."""
    monkeypatch.setattr(_qwen2_audio_module, "AutoBridge", _FakeAutoBridge)


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None

    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1


class TestQwen2AudioFinetuneConfig:
    """Test cases for qwen2_audio_7b_finetune_config."""

    def test_finetune_config_basic_structure(self):
        """Default finetune config is a valid ConfigContainer."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config()
        _assert_basic_config(cfg)

    def test_finetune_config_default_parallelism(self):
        """Default parallelism is single-GPU (TP=1, PP=1, no CP/SP)."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config()

        assert cfg.model.tensor_model_parallel_size == 1
        assert cfg.model.pipeline_model_parallel_size == 1
        assert cfg.model.context_parallel_size == 1
        assert cfg.model.virtual_pipeline_model_parallel_size is None
        assert cfg.model.sequence_parallel is False

    def test_finetune_config_default_training_settings(self):
        """Default training settings match the recipe contract."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config()

        assert cfg.train.train_iters == 2000
        assert cfg.train.global_batch_size == 32
        assert cfg.train.micro_batch_size == 1
        assert cfg.train.manual_gc is True
        assert cfg.train.manual_gc_interval == 100
        assert cfg.train.manual_gc_eval == 100
        assert cfg.validation.eval_interval == 500
        # eval_iters is hard-coded to 0 in the recipe
        assert cfg.validation.eval_iters == 0

    def test_finetune_config_default_seq_length_propagates(self):
        """seq_length flows from kwargs into both model_cfg and dataset."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config()

        assert cfg.model.seq_length == 4096
        assert cfg.dataset.seq_length == 4096

    def test_finetune_config_default_freeze_flags(self):
        """No components are frozen by default."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config()

        assert cfg.model.freeze_language_model is False
        assert cfg.model.freeze_audio_model is False
        assert cfg.model.freeze_audio_projection is False

    def test_finetune_config_uses_null_tokenizer(self):
        """VLM/audio recipes use NullTokenizer; the processor handles tokenization."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config()

        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"

    def test_finetune_config_uses_hf_conversation_provider(self):
        """Dataset is HFDatasetConversationProvider with the audio maker."""
        from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config()

        assert isinstance(cfg.dataset, HFDatasetConversationProvider)
        assert cfg.dataset.maker_name == "make_default_audio_dataset"
        assert cfg.dataset.hf_processor_path == "Qwen/Qwen2-Audio-7B-Instruct"
        # Default split selection — train uses train, val uses dev.
        assert cfg.dataset.maker_kwargs["subset"] == "train"
        assert cfg.dataset.val_maker_kwargs["subset"] == "dev"

    def test_finetune_config_full_sft_uses_low_lr(self):
        """When peft is None (full SFT), the entry point picks lr=5e-6."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config()

        assert cfg.peft is None
        # Cosine annealing scheduler exposes max_lr / min_lr.
        assert cfg.optimizer.lr == pytest.approx(5e-6)

    def test_finetune_config_peft_none_string_treated_as_full_sft(self):
        """peft='none' (string) selects the full-SFT lr path."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config(peft="none")

        # default_peft_config returns None for 'none' string.
        assert cfg.peft is None
        assert cfg.optimizer.lr == pytest.approx(5e-6)

    def test_finetune_config_lora_uses_higher_lr(self):
        """When peft='lora', the entry point picks lr=1e-4."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config(peft="lora")

        assert cfg.peft is not None
        assert cfg.optimizer.lr == pytest.approx(1e-4)

    def test_finetune_config_dora_attached(self):
        """peft='dora' attaches a DoRA config object."""
        from megatron.bridge.peft.dora import DoRA

        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config(peft="dora")

        assert isinstance(cfg.peft, DoRA)

    def test_finetune_config_unknown_peft_raises(self):
        """Invalid PEFT scheme strings raise a clear error."""
        with pytest.raises(ValueError, match="Unknown PEFT scheme"):
            _qwen2_audio_module.qwen2_audio_7b_finetune_config(peft="not-a-scheme")

    def test_finetune_config_explicit_finetune_lr_wins(self):
        """User-supplied finetune_lr overrides the entry point's default."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config(finetune_lr=2.5e-5)

        assert cfg.optimizer.lr == pytest.approx(2.5e-5)

    def test_finetune_config_freeze_flags_propagate(self):
        """User-supplied freeze flags pass through to the provider config."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config(
            freeze_language_model=True,
            freeze_audio_model=True,
            freeze_audio_projection=True,
        )

        assert cfg.model.freeze_language_model is True
        assert cfg.model.freeze_audio_model is True
        assert cfg.model.freeze_audio_projection is True

    def test_finetune_config_parallelism_overrides_apply(self):
        """Parallelism overrides flow into model_cfg as set."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=4,
            context_parallel_size=2,
            sequence_parallel=True,
        )

        assert cfg.model.tensor_model_parallel_size == 2
        assert cfg.model.pipeline_model_parallel_size == 4
        assert cfg.model.context_parallel_size == 2
        assert cfg.model.sequence_parallel is True

    def test_finetune_config_seq_length_override_propagates(self):
        """A non-default seq_length flows into both model_cfg and dataset."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config(seq_length=8192)

        assert cfg.model.seq_length == 8192
        assert cfg.dataset.seq_length == 8192

    def test_finetune_config_ddp_settings(self):
        """DDP defaults match the recipe's documented settings."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config()

        assert cfg.ddp.use_distributed_optimizer is True
        assert cfg.ddp.overlap_grad_reduce is False
        assert cfg.ddp.overlap_param_gather is False
        assert cfg.ddp.grad_reduce_in_fp32 is True
        assert cfg.ddp.average_in_collective is True
        assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"
        assert cfg.ddp.check_for_nan_in_grad is True

    def test_finetune_config_checkpoint_defaults(self):
        """Checkpoint config uses torch_dist with parallel save."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config()

        assert cfg.checkpoint.save_interval == 200
        assert cfg.checkpoint.ckpt_format == "torch_dist"
        assert cfg.checkpoint.fully_parallel_save is True

    def test_finetune_config_rng_seed(self):
        """RNG seed is the recipe-default 1234."""
        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config()

        assert cfg.rng.seed == 1234

    def test_finetune_config_test_split_override(self):
        """val/test maker overrides flow through to the dataset provider."""
        custom_test = {"subset": "test", "split": "test"}

        cfg = _qwen2_audio_module.qwen2_audio_7b_finetune_config(
            test_maker_kwargs=custom_test,
        )

        assert cfg.dataset.test_maker_kwargs == custom_test
