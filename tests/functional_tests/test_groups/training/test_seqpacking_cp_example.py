# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import faulthandler
import logging
import os
import socket
import sys
import time
import traceback

import pytest
import torch

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.recipes.llama.llama3 import llama32_1b_sft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


logger = logging.getLogger(__name__)


def _diag_fields(**extra: object) -> str:
    rank = os.getenv("RANK", "unset")
    local_rank = os.getenv("LOCAL_RANK", "unset")
    world_size = os.getenv("WORLD_SIZE", "unset")
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if initialized:
        rank = str(torch.distributed.get_rank())
        world_size = str(torch.distributed.get_world_size())

    fields = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "initialized": initialized,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "time": f"{time.time():.3f}",
    }
    fields.update(extra)
    return " ".join(f"{key}={value}" for key, value in fields.items())


def _diag(event: str, **extra: object) -> None:
    message = f"[SEQPACK_CP_DIAG] {event} {_diag_fields(**extra)}\n"
    sys.stderr.write(message)
    sys.stderr.flush()
    logger.warning(message.rstrip())


def _diag_env() -> None:
    env_keys = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
        "GHA_RUNNER",
        "GITHUB_RUN_ID",
        "HF_HOME",
        "NEMO_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TORCH_DISTRIBUTED_DEBUG",
        "NCCL_DEBUG",
        "NCCL_NVLS_ENABLE",
        "TORCH_NCCL_AVOID_RECORD_STREAMS",
        "PYTEST_CURRENT_TEST",
    )
    env = {key: os.getenv(key, "unset") for key in env_keys}
    _diag("env", **env)


class TestPeftSftExample:
    """Run the PEFT SFT example as a functional test with packed sequences + CP."""

    @pytest.mark.run_only_on("GPU")
    def test_sft_example_runs_with_cp_and_packing(self, tmp_path):
        caught_exception = None
        shared_dir = None
        os.environ["MBRIDGE_SEQPACK_CP_DIAG"] = "1"
        faulthandler.dump_traceback_later(180, repeat=True, file=sys.stderr)
        try:
            _diag("test_start", tmp_path=tmp_path)
            _diag_env()

            _diag("before_importorskip_transformer_engine")
            pytest.importorskip("transformer_engine_torch")
            _diag("after_importorskip_transformer_engine")

            _diag("before_initialize_distributed")
            initialize_distributed()
            _diag(
                "after_initialize_distributed",
                cuda_device_count=torch.cuda.device_count(),
                current_device=torch.cuda.current_device() if torch.cuda.is_available() else "none",
            )

            if torch.distributed.get_world_size() < 2:
                _diag("skip_world_size_too_small")
                pytest.skip("requires >=2 GPUs for context_parallel_size=2")

            _diag("before_broadcast_path", tmp_path=tmp_path)
            shared_dir = broadcast_path(tmp_path)
            _diag("after_broadcast_path", shared_dir=shared_dir)

            checkpoint_dir = os.path.join(shared_dir, "checkpoints")
            tensorboard_dir = os.path.join(shared_dir, "tensorboard")

            if torch.distributed.get_rank() == 0:
                _diag("before_makedirs", checkpoint_dir=checkpoint_dir, tensorboard_dir=tensorboard_dir)
                os.makedirs(checkpoint_dir, exist_ok=True)
                os.makedirs(tensorboard_dir, exist_ok=True)
                _diag("after_makedirs", checkpoint_dir=checkpoint_dir, tensorboard_dir=tensorboard_dir)

            _diag("before_initial_barrier", shared_dir=shared_dir)
            torch.distributed.barrier()
            _diag("after_initial_barrier", shared_dir=shared_dir)

            _diag("before_config")
            cfg = llama32_1b_sft_config()
            cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
            cfg.tokenizer.tokenizer_model = "meta-llama/Llama-3.2-1B"
            cfg.model.calculate_per_token_loss = True
            cfg.ddp.average_in_collective = False

            # Keep the world-size math simple: tp=1, pp=1, cp=2 -> dp derived from env.
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 2

            # Small, fast run
            cfg.train.train_iters = 2
            cfg.train.global_batch_size = 2
            cfg.train.micro_batch_size = 1
            cfg.validation.eval_interval = 1
            cfg.validation.eval_iters = 0
            cfg.scheduler.lr_warmup_iters = 0
            cfg.logger.log_interval = 1
            cfg.logger.tensorboard_dir = tensorboard_dir

            # Use a small packed SQuAD dataset to exercise THD/context-parallel slicing
            cfg.dataset = HFDatasetConfig(
                dataset_name="rajpurkar/squad",
                process_example_fn=process_squad_example,
                seq_length=256,
                dataloader_type="batch",
                num_workers=1,
                do_validation=False,
                do_test=False,
                val_proportion=None,
                dataset_kwargs={"pad_to_max_length": True},
                max_train_samples=16,
                packed_sequence_specs=PackedSequenceSpecs(
                    packed_sequence_size=512,
                    tokenizer_model_name="meta-llama/Llama-3.2-1B",
                    pad_seq_to_mult=cfg.model.context_parallel_size * 2,
                ),
                rewrite=False,
            )

            cfg.model.seq_length = 256
            cfg.checkpoint.save_interval = cfg.train.train_iters
            cfg.checkpoint.save = checkpoint_dir
            cfg.checkpoint.pretrained_checkpoint = None
            _diag(
                "after_config",
                seq_length=cfg.model.seq_length,
                train_iters=cfg.train.train_iters,
                global_batch_size=cfg.train.global_batch_size,
                micro_batch_size=cfg.train.micro_batch_size,
                context_parallel_size=cfg.model.context_parallel_size,
                dataset_name=cfg.dataset.dataset_name,
                checkpoint_dir=checkpoint_dir,
            )

            _diag("before_finetune", shared_dir=shared_dir)
            finetune(cfg, forward_step)
            _diag("after_finetune", checkpoint_dir=checkpoint_dir)

            _diag("before_verify_checkpoint", checkpoint_dir=checkpoint_dir)
            verify_checkpoint_files(
                checkpoint_dir,
                cfg.train.train_iters,
                ckpt_format=cfg.checkpoint.ckpt_format,
                storage_writers_per_rank=cfg.checkpoint.storage_writers_per_rank,
            )
            _diag("after_verify_checkpoint", checkpoint_dir=checkpoint_dir)
        except BaseException as exc:
            caught_exception = exc
            _diag("caught_exception", exc_type=type(exc).__name__, exc=repr(exc))
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            _diag(
                "finally_enter",
                shared_dir=shared_dir,
                caught_exception=type(caught_exception).__name__ if caught_exception is not None else "none",
            )
            if shared_dir is not None and caught_exception is None:
                clear_directories(shared_dir, debug_label="seqpacking_cp")
            elif shared_dir is not None:
                _diag("skip_distributed_cleanup_after_exception", shared_dir=shared_dir)
            _diag("finally_exit", shared_dir=shared_dir)
            faulthandler.cancel_dump_traceback_later()
            os.environ.pop("MBRIDGE_SEQPACK_CP_DIAG", None)
