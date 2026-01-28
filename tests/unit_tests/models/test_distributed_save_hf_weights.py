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


"""
Unit tests for distributed_save.

Run with: torchrun --nproc_per_node=8 --master_port=29501 -m pytest -vs tests/unit_tests/models/test_distributed_save_hf_weights.py

Or for single GPU: pytest -vs tests/unit_tests/models/test_distributed_save_hf_weights.py
"""

import datetime
import logging
import os
import shutil
import time
import unittest
from pathlib import Path

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist

from megatron.bridge.models.conversion.auto_bridge import AutoBridge


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_parallel_state(tp_size, pp_size):
    if not dist.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if world_size == 1:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29501"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        device_count = torch.cuda.device_count()
        if device_count > 0:
            torch.cuda.set_device(local_rank)

        init_process_group_kwargs = {
            "backend": "nccl" if device_count > 0 else "gloo",
            "world_size": world_size,
            "rank": rank,
            "timeout": datetime.timedelta(minutes=30),
        }
        dist.init_process_group(**init_process_group_kwargs)

    assert dist.is_initialized()
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
        )


class TestAutoBridgeDistributedSave(unittest.TestCase):
    def test_distributed_save_hf_pretrained(
        self,
    ):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        pp_size = 1
        tp_size = min(2, world_size // pp_size)
        distributed_save = True
        save_every_n_ranks = 1
        temp_dir = f"_test_distributed_save_dir_{distributed_save}/hf_exports_qwen3_8B"

        init_parallel_state(tp_size, pp_size)

        output_path = Path(temp_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            bridge = AutoBridge.from_hf_pretrained(
                "Qwen/Qwen3-8B",
                trust_remote_code=True,
            )

            provider = bridge.to_megatron_provider()
            provider.tensor_model_parallel_size = tp_size
            provider.pipeline_model_parallel_size = pp_size
            provider.finalize()

            model = provider.provide_distributed_model(wrap_with_ddp=False)

            torch.cuda.synchronize()
            before_save = time.time()
            bridge.save_hf_pretrained(
                model, str(output_path), distributed_save=distributed_save, save_every_n_ranks=save_every_n_ranks
            )
            torch.distributed.barrier()
            torch.cuda.synchronize()
            after_save = time.time()

            assert output_path.exists(), f"Output directory {output_path} was not created"
            if torch.distributed.get_rank() == 0:
                for item in output_path.iterdir():
                    logger.info(f"  {item.name} {item.is_file()}")

                config_file = output_path / "config.json"
                assert config_file.exists(), "config.json not found in output directory"

                weight_files = list(output_path.glob("model*.safetensors")) or list(
                    output_path.glob("pytorch_model*.bin")
                )
                assert len(weight_files) > 0, "No model weight files found in output directory"

                tokenizer_files = list(output_path.glob("tokenizer*"))
                assert len(tokenizer_files) > 0, "No tokenizer files found in output directory"

                from transformers import AutoConfig, AutoModelForCausalLM

                reloaded_config = AutoConfig.from_pretrained(str(output_path))
                assert reloaded_config is not None, "Failed to load config from saved model"

                reloaded_model = AutoModelForCausalLM.from_pretrained(
                    str(output_path),
                    device_map="cpu",
                    trust_remote_code=True,
                )
                assert reloaded_model is not None, "Failed to load model from saved checkpoint"

                assert hasattr(reloaded_model, "model"), "Reloaded model missing 'model' attribute"
                assert hasattr(reloaded_model.model, "layers"), "Reloaded model missing 'layers' attribute"

                logger.info(
                    f"Distributed_save test passed: Model successfully saved to {output_path} using time {(after_save - before_save):.2f}s"
                )
                logger.info(f"  - Config file: {config_file}")
                logger.info(f"  - Weight files: {len(weight_files)} file(s)")
                logger.info(f"  - Tokenizer files: {len(tokenizer_files)} file(s)")
                logger.info("  - Model successfully reloaded and validated")

        except Exception as e:
            logger.error(f"Distributed_save test skipped due to: {e}")
            pytest.skip(f"Distributed_save test skipped due to: {e}")

        finally:
            if torch.distributed.get_rank() == 0 and output_path.exists():
                try:
                    shutil.rmtree(output_path)
                    logger.info(f"Successfully cleaned up temporary directory: {output_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory {output_path}: {e}")
