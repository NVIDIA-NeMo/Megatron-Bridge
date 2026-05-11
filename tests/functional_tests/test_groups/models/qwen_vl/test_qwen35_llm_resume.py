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

"""T2.7: Checkpoint save and resume for the Qwen3.5 LLM-only recipe.

Two-phase test:
  Phase 1  Train 3 steps. Just before save, perturb the first trainable
           parameter to a sentinel value far outside any normal training
           trajectory. The save captures the perturbed state.
  Phase 2  Resume from that checkpoint. Read the same parameter at
           on_train_start (before any optimizer step in phase 2). The
           sentinel must round-trip — proving the checkpoint loader
           overrode the HF pre-wrap hook.

Assertions:
  - After resume the training loop starts at step PHASE1_ITERS (not 0).
  - The sentinel value written in phase 1 is observed at phase 2 start
    (i.e. the model state came from the checkpoint, not from a fresh HF
    re-load). This guards against the failure mode where the HF
    pre_wrap_hook silently shadows the checkpoint loader.
  - All losses are finite throughout both phases.

Run with (single GPU):
    HF_HOME=... HF_PATH=Qwen/Qwen3.5-0.8B \\
    uv run python -m torch.distributed.run --nproc_per_node=1 --master_port=29711 \\
        tests/functional_tests/test_groups/models/qwen_vl/test_qwen35_llm_resume.py
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist


HF_PATH = os.environ.get("HF_PATH", "Qwen/Qwen3.5-0.8B")
PHASE1_ITERS = int(os.environ.get("PHASE1_ITERS", "3"))
PHASE2_ITERS = int(os.environ.get("PHASE2_ITERS", "2"))

# Value far outside any normal training trajectory (HF init is O(0.1), updates
# are O(1e-4)). If the checkpoint loader runs, this round-trips intact.
SENTINEL_VALUE = -1.2345e6


def _find_first_trainable_param(model_list):
    """Pick a deterministic, trainable parameter present on this rank."""
    for model in model_list:
        for name, p in model.named_parameters():
            if p.requires_grad and p.numel() > 0:
                return name, p
    raise RuntimeError("No trainable parameters found on this rank")


def _make_cfg(hf_path: str, ckpt_dir: str, train_iters: int):
    from megatron.bridge.recipes.qwen_vl import qwen35_llm_800m_sft_config

    cfg = qwen35_llm_800m_sft_config(hf_path=hf_path)
    cfg.train.train_iters = train_iters
    cfg.train.log_interval = 1
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.checkpoint.save = ckpt_dir
    cfg.checkpoint.load = ckpt_dir
    cfg.checkpoint.save_interval = PHASE1_ITERS
    return cfg


def main():
    from megatron.bridge.training.callbacks import Callback, CallbackContext
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.vlm_step import forward_step

    ckpt_dir = tempfile.mkdtemp(prefix="t27_resume_ckpt_")
    phase1_losses: list[float] = []
    phase2_steps: list[int] = []
    phase2_losses: list[float] = []
    phase1_perturbed_param: dict[str, str] = {}  # set by PerturbAtLastStep
    phase2_observed: dict[str, float] = {}  # set by CheckResumedState

    class LossRecorder(Callback):
        def __init__(self, steps_out: list, losses_out: list):
            self._steps = steps_out
            self._losses = losses_out

        def on_train_step_end(self, context: CallbackContext) -> None:
            step = context.state.train_state.step
            loss_dict = context.loss_dict or {}
            loss = loss_dict.get("lm loss", float("nan"))
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            self._steps.append(step)
            self._losses.append(loss)

    class PerturbAtLastStep(Callback):
        """At step PHASE1_ITERS (the last step that fires before save),
        overwrite the first trainable parameter's first element with the
        sentinel. The save_checkpoint_and_time call that follows in the
        train loop captures this perturbed state.
        """

        def __init__(self, target_step: int, recorded_name: dict[str, str]):
            self._target_step = target_step
            self._recorded_name = recorded_name

        def on_train_step_end(self, context: CallbackContext) -> None:
            if context.state.train_state.step != self._target_step:
                return
            name, p = _find_first_trainable_param(context.model)
            with torch.no_grad():
                p.data.view(-1)[0] = SENTINEL_VALUE
            self._recorded_name["name"] = name

    class CheckResumedState(Callback):
        """At the very start of phase 2 (before any optimizer step), read
        the same parameter back and record its first element.
        """

        def __init__(self, observed: dict[str, float]):
            self._observed = observed

        def on_train_start(self, context: CallbackContext) -> None:
            name, p = _find_first_trainable_param(context.model)
            self._observed["name"] = name
            self._observed["value"] = p.data.view(-1)[0].item()

    # Phase 1: fresh start; perturb at last step so save captures sentinel.
    cfg1 = _make_cfg(HF_PATH, ckpt_dir, PHASE1_ITERS)
    phase1_steps: list[int] = []
    finetune(
        cfg1,
        forward_step,
        callbacks=[
            LossRecorder(phase1_steps, phase1_losses),
            PerturbAtLastStep(target_step=PHASE1_ITERS, recorded_name=phase1_perturbed_param),
        ],
    )

    # Phase 2: resume; observe the param at on_train_start before any update.
    cfg2 = _make_cfg(HF_PATH, ckpt_dir, PHASE1_ITERS + PHASE2_ITERS)
    finetune(
        cfg2,
        forward_step,
        callbacks=[
            LossRecorder(phase2_steps, phase2_losses),
            CheckResumedState(observed=phase2_observed),
        ],
    )

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\n" + "=" * 70)
        print("T2.7 Checkpoint Resume Check")
        print("=" * 70)
        print(f"  Phase 1 steps         : {phase1_steps}")
        print(f"  Phase 1 losses        : {[f'{v:.4f}' for v in phase1_losses]}")
        print(f"  Phase 1 perturbed     : {phase1_perturbed_param.get('name')}")
        print(f"  Phase 2 steps         : {phase2_steps}")
        print(f"  Phase 2 losses        : {[f'{v:.4f}' for v in phase2_losses]}")
        print(f"  Phase 2 observed param: {phase2_observed.get('name')}")
        print(f"  Phase 2 observed value: {phase2_observed.get('value')}")
        print(f"  Expected sentinel     : {SENTINEL_VALUE}")

        failed = []

        # Phase 2 must start at step PHASE1_ITERS (not 0)
        if not phase2_steps:
            failed.append("Phase 2 recorded no steps — training did not run after resume")
        elif phase2_steps[0] != PHASE1_ITERS:
            failed.append(
                f"Phase 2 started at step {phase2_steps[0]}, expected {PHASE1_ITERS} (checkpoint not loaded properly)"
            )
        else:
            print(f"\n  PASS  Resume starts at step {phase2_steps[0]} (not 0)")

        # Sentinel round-trip — this is the load-bearing weight identity check.
        observed_name = phase2_observed.get("name")
        observed_value = phase2_observed.get("value")
        if observed_name != phase1_perturbed_param.get("name"):
            failed.append(
                f"Phase 1 perturbed {phase1_perturbed_param.get('name')!r} but phase 2 observed {observed_name!r} — "
                "both phases must target the same parameter for the round-trip check to be valid"
            )
        elif observed_value is None:
            failed.append("Phase 2 did not record a parameter value at on_train_start")
        else:
            # on_train_start fires before any optimizer step in phase 2, so the
            # value should equal the sentinel exactly. Allow a tight tolerance
            # in case of fp16/bf16 round-trip.
            if abs(observed_value - SENTINEL_VALUE) > abs(SENTINEL_VALUE) * 1e-3:
                failed.append(
                    f"Sentinel mismatch: phase 1 wrote {SENTINEL_VALUE} into {observed_name}, "
                    f"phase 2 observed {observed_value} — checkpoint loader did not override HF pre-wrap hook"
                )
            else:
                print(
                    f"  PASS  Sentinel round-trip: wrote {SENTINEL_VALUE} → observed "
                    f"{observed_value} on {observed_name}"
                )

        # All losses must be finite
        all_losses = phase1_losses + phase2_losses
        nan_losses = [v for v in all_losses if not (v == v) or v == float("inf")]
        if nan_losses:
            failed.append(f"Non-finite losses detected: {nan_losses}")
        else:
            print(f"  PASS  All {len(all_losses)} losses are finite")

        if failed:
            print("\n  FAIL:")
            for msg in failed:
                print(f"    {msg}")
            sys.exit(1)

        print("\n  PASS — checkpoint resume works correctly")
        print("=" * 70)


if __name__ == "__main__":
    main()
