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

"""End-to-end DPO: score the reference offline, then train the policy against the artifact."""

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import DPODatasetConfig, HFDatasetSourceConfig
from megatron.bridge.data.datasets.preference import load_ref_logprobs
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.dpo import DPOLossConfig
from megatron.bridge.training.dpo_step import forward_step
from megatron.bridge.training.dpo_train import dpo_train
from tests.functional_tests.fixture_utils import get_test_data_root
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
STAGED_TOKENIZER = "megatron_bridge/tokenizers/meta-llama/Llama-3.1-8B-Instruct"
FALLBACK_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"

NUM_PAIRS = 16
SEQ_LENGTH = 128
TRAIN_ITERS = 4
GLOBAL_BATCH_ROWS = 8  # 4 pairs per step


def _tokenizer_path() -> str:
    staged = get_test_data_root() / STAGED_TOKENIZER
    return str(staged) if staged.is_dir() else FALLBACK_TOKENIZER


def _load_scorer_module():
    script_path = REPO_ROOT / "scripts" / "dpo" / "score_reference_logprobs.py"
    spec = importlib.util.spec_from_file_location("score_reference_logprobs", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_reference_model(model_dir: Path) -> None:
    """A two-layer Llama with the instruct tokenizer's chat template, saved in HF format."""
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path())
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=SEQ_LENGTH,
        rope_theta=500_000.0,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        torch_dtype=torch.bfloat16,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)


def _write_pairs(data_path: Path) -> None:
    with open(data_path, "w", encoding="utf-8") as output_file:
        for index in range(NUM_PAIRS):
            prompt = {"role": "user", "content": f"Question {index}: what is {index} plus {index}?"}
            row = {
                "chosen": [prompt, {"role": "assistant", "content": f"{index} plus {index} is {2 * index}."}],
                "rejected": [prompt, {"role": "assistant", "content": "I would rather not say."}],
            }
            output_file.write(json.dumps(row) + "\n")


class _FirstStepMetrics(Callback):
    """Keeps the reduced metrics of the first optimizer step."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] | None = None

    def on_train_step_end(self, context) -> None:
        if self.metrics is None:
            self.metrics = {key: float(value) for key, value in context.loss_dict.items()}


def _dpo_config(
    model_dir: Path, data_path: Path, artifact_dir: Path, checkpoint_dir: Path, tensorboard_dir: Path
) -> ConfigContainer:
    model = AutoBridge.from_hf_pretrained(str(model_dir)).to_megatron_provider(load_weights=False)
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.context_parallel_size = 1
    model.sequence_parallel = False
    model.seq_length = SEQ_LENGTH
    model.calculate_per_token_loss = True
    model.cross_entropy_loss_fusion = False
    model.mtp_num_layers = None

    return ConfigContainer(
        model=model,
        train=TrainingConfig(
            train_iters=TRAIN_ITERS,
            global_batch_size=GLOBAL_BATCH_ROWS,
            micro_batch_size=2,
            exit_signal_handler=True,
        ),
        validation=ValidationConfig(eval_interval=TRAIN_ITERS, eval_iters=0),
        optimizer=OptimizerConfig(
            optimizer="adam",
            bf16=True,
            fp16=False,
            lr=1e-6,
            min_lr=0.0,
            weight_decay=0.0,
            clip_grad=1.0,
            use_distributed_optimizer=True,
        ),
        scheduler=SchedulerConfig(
            start_weight_decay=0.0,
            end_weight_decay=0.0,
            weight_decay_incr_style="constant",
            lr_decay_style="constant",
            lr_warmup_iters=0,
            lr_decay_iters=TRAIN_ITERS,
            override_opt_param_scheduler=True,
        ),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=False,
            use_distributed_optimizer=True,
        ),
        dataset=DPODatasetConfig(
            seq_length=SEQ_LENGTH,
            source=HFDatasetSourceConfig(
                path_or_dataset="json", load_kwargs={"data_files": str(data_path)}, split="train"
            ),
            ref_artifact=str(artifact_dir),
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        ),
        dpo=DPOLossConfig(),
        logger=LoggerConfig(log_interval=1, tensorboard_dir=str(tensorboard_dir)),
        tokenizer=TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=str(model_dir)),
        checkpoint=CheckpointConfig(
            save_interval=TRAIN_ITERS,
            save=str(checkpoint_dir),
            load=str(checkpoint_dir),
            pretrained_checkpoint=str(model_dir),
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=False,
        ),
        rng=RNGConfig(seed=1234),
        mixed_precision="bf16_mixed",
    )


class TestDPOTraining:
    """Score with the offline scorer, train with ``dpo_train`` on two data-parallel ranks."""

    @pytest.mark.run_only_on("GPU")
    def test_score_reference_then_train_policy(self, tmp_path):
        initialize_distributed()
        shared_base_dir = Path(broadcast_path(tmp_path))
        model_dir = shared_base_dir / "reference_hf"
        data_path = shared_base_dir / "pairs.jsonl"
        artifact_dir = shared_base_dir / "ref_logprobs"
        checkpoint_dir = shared_base_dir / "checkpoints"
        tensorboard_dir = shared_base_dir / "tb_logs"

        if torch.distributed.get_rank() == 0:
            _write_reference_model(model_dir)
            _write_pairs(data_path)
        torch.distributed.barrier()

        try:
            scorer = _load_scorer_module()
            scorer.ReferenceLogprobScorer(
                _dpo_config(model_dir, data_path, artifact_dir, checkpoint_dir, tensorboard_dir),
                scorer.ScoringOptions(output=str(artifact_dir), micro_batch_size=4),
            ).run()
            torch.distributed.barrier()
            assert len(load_ref_logprobs(str(artifact_dir), expected_num_pairs=NUM_PAIRS)) == NUM_PAIRS

            first_step = _FirstStepMetrics()
            dpo_train(
                _dpo_config(model_dir, data_path, artifact_dir, checkpoint_dir, tensorboard_dir),
                forward_step,
                callbacks=[first_step],
            )

            # Before the first update the policy is the reference, so every margin is noise around 0 and
            # the preference loss is -log sigmoid(0) = ln 2. Every pair is well-formed, so none is a stub.
            assert first_step.metrics is not None
            assert first_step.metrics["live fraction"] == pytest.approx(1.0)
            assert first_step.metrics["preference loss"] == pytest.approx(math.log(2), abs=0.05)
            verify_checkpoint_files(str(checkpoint_dir), TRAIN_ITERS, ckpt_format="torch_dist")
        finally:
            clear_directories(str(shared_base_dir))
