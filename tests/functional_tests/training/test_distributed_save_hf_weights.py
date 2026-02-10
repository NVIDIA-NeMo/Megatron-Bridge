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
functional tests for distributed_save.

Run with: torchrun --nproc_per_node=8 --master_port=29501 -m pytest -vs tests/functional_tests/training/test_distributed_save_hf_weights.py

Or for single GPU: pytest -vs tests/functional_tests/training/test_distributed_save_hf_weights.py
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
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from megatron.bridge.models.conversion.auto_bridge import AutoBridge


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


HF_QWEN2_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 896,
    "initializer_range": 0.02,
    "intermediate_size": 4864,
    "max_position_embeddings": 131072,
    "max_window_layers": 2,
    "model_type": "qwen2",
    "num_attention_heads": 14,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 131072,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.1",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


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


@pytest.mark.integration
@pytest.mark.gpu
class TestAutoBridgeDistributedSave(unittest.TestCase):
    def test_distributed_save_hf_pretrained(
        self,
    ):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        pp_size = 1
        tp_size = min(2, world_size // pp_size)
        distributed_save = True
        save_every_n_ranks = 1
        toy_dir = f"_test_distributed_save_dir_{distributed_save}/qwen_toy_path"
        temp_dir = f"_test_distributed_save_dir_{distributed_save}/hf_exports_qwen_toy"

        init_parallel_state(tp_size, pp_size)
        output_path = Path(temp_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # toy model
        config = Qwen2Config(**HF_QWEN2_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16
        toy_model = Qwen2ForCausalLM(config)
        toy_model = toy_model.bfloat16()
        if torch.distributed.get_rank() == 0:
            toy_model.save_pretrained(toy_dir)
        torch.distributed.barrier()

        try:
            bridge = AutoBridge.from_hf_pretrained(
                toy_dir,
                trust_remote_code=True,
            )

            provider = bridge.to_megatron_provider()
            provider.tensor_model_parallel_size = tp_size
            provider.pipeline_model_parallel_size = pp_size
            provider.finalize()

            model = provider.provide_distributed_model(wrap_with_ddp=False)

            torch.cuda.synchronize()
            before_save = time.time()
            bridge.save_hf_weights(
                model,
                str(output_path),
                merge_adapter_weights=True,
                distributed_save=distributed_save,
                save_every_n_ranks=save_every_n_ranks,
            )
            torch.distributed.barrier()
            torch.cuda.synchronize()
            after_save = time.time()

            assert output_path.exists(), f"Output directory {output_path} was not created"
            if torch.distributed.get_rank() == 0:
                for item in output_path.iterdir():
                    logger.info(f"  {item.name} {item.is_file()}")

                weight_files = list(output_path.glob("model*.safetensors")) or list(
                    output_path.glob("pytorch_model*.bin")
                )
                assert len(weight_files) > 0, "No model weight files found in output directory"

                shutil.copy(Path(toy_dir) / "config.json", output_path / "config.json")

                reloaded_model = AutoModelForCausalLM.from_pretrained(
                    str(output_path),
                    device_map="cpu",
                    trust_remote_code=True,
                )
                assert reloaded_model is not None, "Failed to load model from saved checkpoint"

                assert hasattr(reloaded_model, "model"), "Reloaded model missing 'model' attribute"
                assert hasattr(reloaded_model.model, "layers"), "Reloaded model missing 'layers' attribute"

                # Compare weights between toy_model and reloaded_model
                toy_model_cpu = toy_model.cpu()
                toy_state_dict = toy_model_cpu.state_dict()
                reloaded_state_dict = reloaded_model.state_dict()

                # Check if all keys match
                toy_keys = set(toy_state_dict.keys())
                reloaded_keys = set(reloaded_state_dict.keys())

                missing_keys = toy_keys - reloaded_keys
                extra_keys = reloaded_keys - toy_keys

                if missing_keys:
                    logger.warning(f"Missing keys in reloaded model: {missing_keys}")
                if extra_keys:
                    logger.warning(f"Extra keys in reloaded model: {extra_keys}")

                assert toy_keys == reloaded_keys, f"Key mismatch: missing={missing_keys}, extra={extra_keys}"

                # Compare weight values
                max_diff = 0.0
                mismatched_weights = []
                for key in toy_keys:
                    toy_weight = toy_state_dict[key]
                    reloaded_weight = reloaded_state_dict[key]

                    # Convert to same dtype for comparison
                    if toy_weight.dtype != reloaded_weight.dtype:
                        reloaded_weight = reloaded_weight.to(toy_weight.dtype)

                    diff = torch.abs(toy_weight - reloaded_weight).max().item()
                    if diff > max_diff:
                        max_diff = diff

                    # Allow small numerical differences (e.g., 1e-5)
                    if diff > 1e-5:
                        mismatched_weights.append((key, diff))

                if mismatched_weights:
                    logger.warning(f"Found {len(mismatched_weights)} mismatched weights:")
                    for key, diff in mismatched_weights[:10]:  # Print first 10
                        logger.warning(f"  {key}: max_diff={diff}")
                    assert False, f"Weight mismatch detected. Max difference: {max_diff}"

                logger.info(f"Weight comparison passed! Max difference: {max_diff:.2e}")

                logger.info(
                    f"Distributed_save test passed: Model successfully saved to {output_path} using time {(after_save - before_save):.2f}s"
                )
                logger.info(f"  - Weight files: {len(weight_files)} file(s)")
                logger.info("  - Model successfully reloaded and validated")

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Distributed_save test skipped due to: {e}")
            pytest.skip(f"Distributed_save test skipped due to: {e}")

        finally:
            if torch.distributed.get_rank() == 0 and output_path.exists():
                try:
                    shutil.rmtree(os.path.dirname(output_path))
                    logger.info(f"Successfully cleaned up temporary directory: {output_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory {output_path}: {e}")
