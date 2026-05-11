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

"""T2.5: Vision tower is not instantiated; only LLM params are present.

The recipe sets ``add_encoder=False`` on the provider, so the Megatron
model has no ``vision_model.*`` or ``visual.*`` parameters at all. This test
verifies that and reports peak memory.

Run with (single GPU):
    HF_HOME=... HF_PATH=Qwen/Qwen3.5-0.8B \
    uv run python -m torch.distributed.run --nproc_per_node=1 --master_port=29700 \
        tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_optimizer_state.py
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist


HF_PATH = os.environ.get("HF_PATH", "Qwen/Qwen3.5-0.8B")
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", "3"))
# Peak-memory ceiling for the 0.8B LLM-only path. The reference measurement
# was ~17.3 GB; 25 GB leaves headroom for kernel/dtype drift but still trips
# if vision params or optimizer state for them are accidentally allocated
# (the full Qwen3.5-VL model would push well past 30 GB on the same setup).
PEAK_MEM_GB_CEILING = float(os.environ.get("PEAK_MEM_GB_CEILING", "25.0"))


def main():
    from megatron.bridge.recipes.qwen_vl import qwen35_llm_800m_sft_config
    from megatron.bridge.training.callbacks import Callback, CallbackContext
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.vlm_step import forward_step

    results: dict = {}

    class ParameterChecker(Callback):
        """After the first training step, inspect which params exist and are trainable."""

        def on_train_step_end(self, context: CallbackContext) -> None:
            if context.state.train_state.step != 0:
                return

            from megatron.core.utils import unwrap_model

            vision_present = []
            lm_trainable = []
            lm_frozen = []

            for model_chunk in unwrap_model(context.model):
                for name, param in model_chunk.named_parameters():
                    if "vision_model" in name or "visual" in name:
                        vision_present.append(name)
                    else:
                        if param.requires_grad:
                            lm_trainable.append(name)
                        else:
                            lm_frozen.append(name)

            results["vision_present"] = vision_present
            results["lm_trainable"] = lm_trainable
            results["lm_frozen"] = lm_frozen
            results["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

    cfg = qwen35_llm_800m_sft_config(hf_path=HF_PATH)
    cfg.train.train_iters = TRAIN_ITERS
    cfg.train.log_interval = 1
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1

    # Isolate checkpoint I/O so this test never resumes from a prior run.
    # Point both save and load at a fresh temp dir (load finds nothing → fresh start).
    _ckpt_dir = tempfile.mkdtemp(prefix="t25_optim_ckpt_")
    cfg.checkpoint.save = _ckpt_dir
    cfg.checkpoint.load = _ckpt_dir

    torch.cuda.reset_peak_memory_stats()
    finetune(cfg, forward_step, callbacks=[ParameterChecker()])

    # Only rank-0 reports results.
    if not dist.is_initialized() or dist.get_rank() == 0:
        vision_present = results.get("vision_present", [])
        lm_trainable = results.get("lm_trainable", [])
        lm_frozen = results.get("lm_frozen", [])
        peak_mem_gb = results.get("peak_mem_gb", float("nan"))

        print("\n" + "=" * 70)
        print("T2.5 Parameter Inventory")
        print("=" * 70)
        print(f"  LM params (trainable)      : {len(lm_trainable)}")
        print(f"  LM params (frozen)         : {len(lm_frozen)}")
        print(f"  Vision params (present)    : {len(vision_present)}")
        print(f"  Peak GPU memory (GB)       : {peak_mem_gb:.3f}")

        failed = []

        if vision_present:
            failed.append("vision params are present in the model (should not exist):")
            for n in vision_present[:10]:
                failed.append(f"    {n}")

        if not lm_trainable:
            failed.append("no LM params are trainable (recipe misconfigured)")

        # Regression guard: peak memory must stay under the ceiling. A breach
        # almost always means vision params (or their optimizer state) leaked
        # back into the model.
        if peak_mem_gb != peak_mem_gb:  # NaN — torch.cuda.max_memory_allocated unavailable
            failed.append("peak_mem_gb is NaN; CUDA stats unavailable")
        elif peak_mem_gb > PEAK_MEM_GB_CEILING:
            failed.append(
                f"peak GPU memory {peak_mem_gb:.3f} GB exceeds ceiling {PEAK_MEM_GB_CEILING:.3f} GB — "
                "possible reintroduction of vision params"
            )

        if failed:
            print("\n  FAIL:")
            for msg in failed:
                print(f"    {msg}")
            sys.exit(1)

        print(
            f"\n  PASS — vision tower not instantiated; LM params trainable; "
            f"peak {peak_mem_gb:.3f} GB ≤ {PEAK_MEM_GB_CEILING:.3f} GB ceiling"
        )
        print("=" * 70)


if __name__ == "__main__":
    main()
