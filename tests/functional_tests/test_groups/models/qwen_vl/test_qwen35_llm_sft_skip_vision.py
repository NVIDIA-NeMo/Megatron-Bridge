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

"""T2.2: ``add_encoder=False`` keeps vision parameters out of the model.

Builds the Qwen3.5 LLM-only recipe end-to-end on a real HuggingFace checkpoint
(Qwen/Qwen3.5-0.8B, a VLM) and verifies that:

  - the constructed Megatron model has zero ``vision_model.*`` / ``visual.*``
    parameters and zero buffers (so no vision weights are *initialized* on GPU);
  - language-model parameters are present (the recipe is not over-eager);
  - peak GPU memory after model construction is reported.

Run with (single GPU):
    HF_HOME=... HF_PATH=Qwen/Qwen3.5-0.8B \
    uv run python -m torch.distributed.run --nproc_per_node=1 --master_port=29701 \
        tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_sft_skip_vision.py
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist


HF_PATH = os.environ.get("HF_PATH", "Qwen/Qwen3.5-0.8B")


def _is_vision_name(name: str) -> bool:
    return "vision_model" in name or "visual" in name


def main():
    from megatron.bridge.recipes.qwen_vl import qwen35_llm_800m_sft_config
    from megatron.bridge.training.callbacks import Callback, CallbackContext
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.vlm_step import forward_step

    results: dict = {}

    class VisionInventory(Callback):
        """After the model is built, record vision/LM params and buffers and exit."""

        def on_train_step_end(self, context: CallbackContext) -> None:
            if context.state.train_state.step != 0:
                return

            from megatron.core.utils import unwrap_model

            vision_params: list[str] = []
            vision_buffers: list[str] = []
            lm_params: list[str] = []

            for chunk in unwrap_model(context.model):
                for name, _p in chunk.named_parameters():
                    if _is_vision_name(name):
                        vision_params.append(name)
                    else:
                        lm_params.append(name)
                for name, _b in chunk.named_buffers():
                    if _is_vision_name(name):
                        vision_buffers.append(name)

            results["vision_params"] = vision_params
            results["vision_buffers"] = vision_buffers
            results["lm_params"] = lm_params
            results["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

    cfg = qwen35_llm_800m_sft_config(hf_path=HF_PATH)
    cfg.train.train_iters = 1
    cfg.train.log_interval = 1
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1

    _ckpt_dir = tempfile.mkdtemp(prefix="t22_skip_vision_ckpt_")
    cfg.checkpoint.save = _ckpt_dir
    cfg.checkpoint.load = _ckpt_dir

    torch.cuda.reset_peak_memory_stats()
    finetune(cfg, forward_step, callbacks=[VisionInventory()])

    if not dist.is_initialized() or dist.get_rank() == 0:
        vision_params = results.get("vision_params", [])
        vision_buffers = results.get("vision_buffers", [])
        lm_params = results.get("lm_params", [])
        peak_mem_gb = results.get("peak_mem_gb", float("nan"))

        print("\n" + "=" * 70)
        print("T2.2 Vision-tower-not-initialized Check")
        print("=" * 70)
        print(f"  HF model          : {HF_PATH}")
        print(f"  LM params         : {len(lm_params)}")
        print(f"  Vision params     : {len(vision_params)}")
        print(f"  Vision buffers    : {len(vision_buffers)}")
        print(f"  Peak GPU mem (GB) : {peak_mem_gb:.3f}")

        failed = False
        if vision_params:
            print("\n  FAIL — vision parameters were initialized:")
            for n in vision_params[:10]:
                print(f"    {n}")
            failed = True
        if vision_buffers:
            print("\n  FAIL — vision buffers were initialized:")
            for n in vision_buffers[:10]:
                print(f"    {n}")
            failed = True
        if not lm_params:
            print("\n  FAIL — no LM params present (recipe is mis-wired)")
            failed = True

        if failed:
            sys.exit(1)

        print("\n  PASS — only LM weights are present in the Megatron model")
        print("=" * 70)


if __name__ == "__main__":
    main()
