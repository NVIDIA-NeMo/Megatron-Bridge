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
# - The `vanilla_gpt_pretrain_config` recipe is the documented baseline used
#   for MLM <-> Megatron-Bridge correlation testing, so it must keep its
#   "bare Megatron-LM defaults" shape: GPTModelProvider with no overrides,
#   NullTokenizer, no distributed optimizer, no overlap_grad_reduce, etc.
# - This test file locks those invariants in. It does not need any provider
#   monkeypatching: vanilla_gpt builds GPTModelProvider() directly and never
#   touches HuggingFace.
#

from megatron.bridge.recipes.gpt import vanilla_gpt_pretrain_config
from megatron.bridge.training.config import ConfigContainer


class TestVanillaGptPretrainConfig:
    def test_returns_config_container(self):
        cfg = vanilla_gpt_pretrain_config()
        assert isinstance(cfg, ConfigContainer)

    def test_all_top_level_sections_populated(self):
        cfg = vanilla_gpt_pretrain_config()
        # Every required subsection must be present.
        assert cfg.model is not None
        assert cfg.train is not None
        assert cfg.validation is not None
        assert cfg.optimizer is not None
        assert cfg.scheduler is not None
        assert cfg.ddp is not None
        assert cfg.dataset is not None
        assert cfg.logger is not None
        assert cfg.tokenizer is not None
        assert cfg.checkpoint is not None
        assert cfg.rng is not None
        assert cfg.dist is not None

    def test_uses_null_tokenizer_by_default(self):
        # vanilla_gpt is intentionally tokenizer-free so it can run without
        # any HF account. NullTokenizer + a known vocab_size is the contract.
        cfg = vanilla_gpt_pretrain_config()
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
        assert cfg.tokenizer.vocab_size is not None
        assert cfg.tokenizer.vocab_size > 0

    def test_training_defaults(self):
        cfg = vanilla_gpt_pretrain_config()
        # These three knobs are part of the published example in the
        # module docstring — locking them in protects external users.
        assert cfg.train.train_iters == 300000
        assert cfg.train.global_batch_size == 32
        assert cfg.train.micro_batch_size == 2

    def test_validation_defaults(self):
        cfg = vanilla_gpt_pretrain_config()
        assert cfg.validation.eval_interval == 500
        assert cfg.validation.eval_iters == 32

    def test_dataset_defaults_are_megatron_lm_aligned(self):
        # vanilla_gpt's purpose is parity with Megatron-LM pretrain_gpt.py
        # defaults. Lock in the dataset knobs that drive sequence shape
        # and dataloader behavior.
        cfg = vanilla_gpt_pretrain_config()
        assert cfg.dataset.random_seed == 1234
        assert cfg.dataset.sequence_length == 1024
        assert cfg.dataset.blend is None
        assert cfg.dataset.blend_per_split is None
        assert cfg.dataset.split == "9999,8,2"
        assert cfg.dataset.dataloader_type == "single"
        assert cfg.dataset.reset_position_ids is False
        assert cfg.dataset.reset_attention_mask is False
        assert cfg.dataset.eod_mask_loss is False

    def test_ddp_defaults_match_pretrain_gpt(self):
        # The point of vanilla_gpt is "no hidden distributed-optimizer or
        # comm-overlap surprises." Lock that in.
        cfg = vanilla_gpt_pretrain_config()
        assert cfg.ddp.check_for_nan_in_grad is True
        assert cfg.ddp.grad_reduce_in_fp32 is True
        assert cfg.ddp.overlap_grad_reduce is False
        assert cfg.ddp.overlap_param_gather is False
        assert cfg.ddp.use_distributed_optimizer is False

    def test_mixed_precision_default(self):
        cfg = vanilla_gpt_pretrain_config()
        # bf16_mixed is the published baseline.
        assert cfg.mixed_precision == "bf16_mixed"

    def test_checkpoint_format_and_interval(self):
        cfg = vanilla_gpt_pretrain_config()
        assert cfg.checkpoint.ckpt_format == "torch_dist"
        assert cfg.checkpoint.save_interval == 500

    def test_rng_seed(self):
        cfg = vanilla_gpt_pretrain_config()
        assert cfg.rng.seed == 1234

    def test_logger_interval(self):
        cfg = vanilla_gpt_pretrain_config()
        assert cfg.logger.log_interval == 10

    def test_two_invocations_are_independent(self):
        # Each call should return a fresh ConfigContainer — important so a
        # caller mutating one config never silently affects the next.
        cfg_a = vanilla_gpt_pretrain_config()
        cfg_b = vanilla_gpt_pretrain_config()
        assert cfg_a is not cfg_b
        assert cfg_a.train is not cfg_b.train

    def test_optimizer_lr_warmup_and_decay(self):
        # The recipe assembles its optimizer via
        # distributed_fused_adam_with_cosine_annealing with explicit lr knobs.
        # Lock in the published warmup/peak/min values.
        cfg = vanilla_gpt_pretrain_config()
        assert cfg.scheduler.lr_warmup_iters == 500
        assert cfg.optimizer.lr == 3e-4
        assert cfg.optimizer.min_lr == 3e-5
