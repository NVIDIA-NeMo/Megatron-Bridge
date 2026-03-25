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

"""Functional test: HFEncoderVLMTaskEncoder with a toy HF-encoder VLM model.

Exercises the full pipeline:
  1. HFEncoderVLMTaskEncoder.encode_sample() — encoding ChatML samples with images
  2. HFEncoderVLMTaskEncoder.batch() — padding and collating
  3. HFEncoderVLMTaskEncoder.encode_batch() — wrapping visual tensors in GenericVisualInputs
  4. vlm_step.forward_step — consuming the batch dict during training
  5. pretrain — running 2 iterations end-to-end with checkpoint verification
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pytest
import torch

from megatron.bridge.data.energon.hf_encoder_task_encoder import HFEncoderVLMTaskEncoder
from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


def _make_chatml_sample(key, conversation, imgs=None, videos=None):
    """Create a ChatMLSample compatible with any megatron-energon Sample version."""
    import dataclasses

    base_fields = {f.name for f in dataclasses.fields(ChatMLSample)}
    kwargs = {"conversation": conversation}
    if "__key__" in base_fields:
        kwargs["__key__"] = key
    if "__subflavors__" in base_fields:
        kwargs["__subflavors__"] = {}
    if "__restore_key__" in base_fields:
        kwargs["__restore_key__"] = ()
    if "__subflavor__" in base_fields:
        kwargs["__subflavor__"] = None
    if imgs is not None:
        kwargs["imgs"] = imgs
    if videos is not None:
        kwargs["videos"] = videos
    return ChatMLSample(**kwargs)


@dataclass(kw_only=True)
class MockEnergonHFEncoderProvider(DatasetProvider):
    """DatasetProvider that exercises HFEncoderVLMTaskEncoder with synthetic data.

    Creates synthetic ChatML conversations with random images, encodes them
    through the full HFEncoderVLMTaskEncoder pipeline (encode_sample → batch →
    encode_batch), and returns infinite iterators over the resulting batch dicts.
    """

    seq_length: int
    hf_processor_path: str
    micro_batch_size: int = 1
    dataloader_type: str = "external"
    skip_getting_attention_mask_from_dataset: bool = True

    def _make_samples(self, num_samples: int = 4) -> list[ChatMLSample]:
        """Generate synthetic ChatMLSamples with small random images."""
        rng = np.random.default_rng(42)
        samples = []
        for i in range(num_samples):
            img_tensor = torch.from_numpy(rng.random((3, 64, 64), dtype=np.float32))
            conversation = json.dumps(
                [
                    {"role": "user", "content": "<image> Describe this image."},
                    {"role": "assistant", "content": f"This is a synthetic test image number {i}."},
                ]
            )
            sample = _make_chatml_sample(
                key=f"sample_{i}",
                conversation=conversation,
                imgs=[img_tensor],
            )
            samples.append(sample)
        return samples

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        from transformers import AutoProcessor

        from megatron.bridge.models.hf_pretrained.utils import is_safe_repo

        processor = AutoProcessor.from_pretrained(
            self.hf_processor_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=self.hf_processor_path,
            ),
        )
        task_encoder = HFEncoderVLMTaskEncoder(
            processor=processor,
            seq_length=self.seq_length,
            visual_keys=("pixel_values",),
        )

        samples = self._make_samples(num_samples=max(self.micro_batch_size * 2, 4))

        encoded = [task_encoder.encode_sample(s) for s in samples]
        batch = task_encoder.batch(encoded[: self.micro_batch_size])
        batch_dict = task_encoder.encode_batch(batch)

        def _infinite_iter():
            while True:
                yield batch_dict

        return _infinite_iter(), _infinite_iter(), None


class TestHFEncoderEnergonVLMTraining:
    """End-to-end test: HF encoder task encoder → VLM training for 2 iterations."""

    @pytest.mark.run_only_on("GPU")
    def test_hf_encoder_vlm_pretrain_2_iters(self, tmp_path):
        """Train a Gemma3-VL 4B model (random weights) for 2 iterations using
        batches produced by HFEncoderVLMTaskEncoder with GenericVisualInputs.
        """
        pytest.importorskip("transformer_engine_torch")

        from megatron.bridge.recipes.gemma3_vl.gemma3_vl import gemma3_vl_4b_sft_config
        from megatron.bridge.training.pretrain import pretrain
        from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step
        from tests.functional_tests.utils import (
            broadcast_path,
            clear_directories,
            initialize_distributed,
            verify_checkpoint_files,
        )

        initialize_distributed()

        shared_dir = broadcast_path(tmp_path)
        checkpoint_dir = os.path.join(shared_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)
        torch.distributed.barrier()

        cfg = gemma3_vl_4b_sft_config()

        test_seq_length = 512
        cfg.model.seq_length = test_seq_length

        cfg.model.tensor_model_parallel_size = 1
        cfg.model.pipeline_model_parallel_size = 1
        cfg.model.context_parallel_size = 1
        cfg.model.sequence_parallel = False
        cfg.model.cp_comm_type = "a2a"

        cfg.train.train_iters = 2
        cfg.train.global_batch_size = 2
        cfg.train.micro_batch_size = 1
        cfg.validation.eval_interval = 1
        cfg.validation.eval_iters = 0
        cfg.scheduler.lr_warmup_iters = 0
        cfg.logger.log_interval = 1
        cfg.logger.tensorboard_dir = tensorboard_dir

        cfg.dataset = MockEnergonHFEncoderProvider(
            seq_length=test_seq_length,
            hf_processor_path="google/gemma-3-4b-it",
            micro_batch_size=cfg.train.micro_batch_size,
        )

        cfg.checkpoint.save_interval = cfg.train.train_iters
        cfg.checkpoint.save = checkpoint_dir
        cfg.checkpoint.load = checkpoint_dir
        cfg.checkpoint.pretrained_checkpoint = None

        try:
            pretrain(cfg, vlm_forward_step)
            verify_checkpoint_files(
                checkpoint_dir,
                cfg.train.train_iters,
                ckpt_format=cfg.checkpoint.ckpt_format,
                storage_writers_per_rank=cfg.checkpoint.storage_writers_per_rank,
            )
        finally:
            clear_directories(shared_dir)
