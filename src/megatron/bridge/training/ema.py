# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch

from megatron.bridge.training.callbacks import Callback, CheckpointCallbackContext
from megatron.bridge.training.checkpointing import CheckpointType
from megatron.bridge.training.ema_checkpoint import (
    has_ema_state,
    load_ema_user_state,
    save_ema_user_state,
)
from megatron.bridge.utils.common_utils import print_rank_0

class EMACallback(Callback):
    def __init__(self, decay=0.999, start_step=0, store_on_cpu=False, log_interval=10):
        self.decay = decay
        self.start_step = start_step
        self.store_on_cpu = store_on_cpu
        self.log_interval = log_interval

    def _unwrap(self, chunk):
        return getattr(chunk, "module", chunk)

    def _iter_params(self, model_chunks):
        for chunk_idx, chunk in enumerate(model_chunks):
            module = self._unwrap(chunk)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    yield f"chunk{chunk_idx}.{name}", param

    def _materialize_loaded_state(self, context):
        ema_state = context.user_state["ema_state"]
        expected = {name: param for name, param in self._iter_params(context.model)}

        missing = sorted(set(expected.keys()) - set(ema_state.keys()))
        unexpected = sorted(set(ema_state.keys()) - set(expected.keys()))

        if missing or unexpected:
            raise RuntimeError(
                "Loaded EMA state does not match current model shard.\n"
                f"Missing keys: {missing[:10]}\n"
                f"Unexpected keys: {unexpected[:10]}"
                f"| resumed_at_step={context.state.train_state.step} "
            )

        remapped = {}
        tracked_params = 0

        for name, param in expected.items():
            target_device = "cpu" if self.store_on_cpu else param.device
            remapped[name] = ema_state[name].detach().to(device=target_device, dtype=torch.float32)
            tracked_params += remapped[name].numel()

        context.user_state["ema_state"] = remapped
        context.user_state.setdefault("ema_updates", 0)
        context.user_state.setdefault("ema_skipped_iters", 0)

        where = "CPU" if self.store_on_cpu else "same-device"
        print_rank_0(
            f"[EMA] resumed | decay={self.decay} | start_step={self.start_step} "
            f"| storage={where} | tracked_params={tracked_params} "
            f"| updates={context.user_state['ema_updates']} "
            f"| skipped_seen={context.user_state['ema_skipped_iters']}"
        )

    def on_train_start(self, context):
        ckpt_cfg = context.state.cfg.checkpoint
        if getattr(ckpt_cfg, "non_persistent_ckpt_type", None) == "local":
            raise NotImplementedError(
                "EMACallback does not yet support local checkpoints. "
                "Use persistent/global checkpoints or disable EMA."
            )

        if "ema_state" in context.user_state and context.user_state["ema_state"]:
            self._materialize_loaded_state(context)
            return

        ema_state = {}
        num_params = 0

        with torch.no_grad():
            for name, param in self._iter_params(context.model):
                p = param.detach().float().clone()
                if self.store_on_cpu:
                    p = p.cpu()
                ema_state[name] = p
                num_params += p.numel()

        context.user_state["ema_state"] = ema_state
        context.user_state["ema_updates"] = 0
        context.user_state["ema_skipped_iters"] = 0

        where = "CPU" if self.store_on_cpu else "same-device"

        print_rank_0(
            f"[EMA] initialized | decay={self.decay} | start_step={self.start_step} "
            f"| storage={where} | tracked_params={num_params}"
        )

    def on_train_step_end(self, context):
        step = context.state.train_state.step

        if context.skipped_iter:
            context.user_state["ema_skipped_iters"] += 1

            if self.log_interval and step % self.log_interval == 0:
                print_rank_0(
                    f"[EMA] step={step} | skipped update "
                    f"| skipped_seen={context.user_state['ema_skipped_iters']}"
                )
            return

        if step < self.start_step:
            return

        ema_state = context.user_state["ema_state"]

        with torch.no_grad():
            for name, param in self._iter_params(context.model):
                current = param.detach().float()
                if self.store_on_cpu:
                    current = current.cpu()

                ema_state[name].mul_(self.decay).add_(current, alpha=1.0 - self.decay)

        context.user_state["ema_updates"] += 1

        if self.log_interval and step % self.log_interval == 0:
            first_name, first_param = next(iter(self._iter_params(context.model)))
            current = first_param.detach().float()
            ema_param = ema_state[first_name]

            if self.store_on_cpu:
                diff = (current.cpu() - ema_param).abs().mean().item()
            else:
                diff = (current - ema_param.to(current.device)).abs().mean().item()

            print_rank_0(
                f"[EMA] step={step} | updates={context.user_state['ema_updates']} "
                f"| skipped_seen={context.user_state['ema_skipped_iters']} "
                f"| mean_abs_diff={diff:.6e}"
            )

    def on_train_end(self, context):
        print_rank_0(
            f"[EMA] training finished | total_updates={context.user_state.get('ema_updates', 0)} "
            f"| total_skipped={context.user_state.get('ema_skipped_iters', 0)}"
        )
    
    def on_checkpoint_save(self, context: CheckpointCallbackContext) -> None:
        if context.checkpoint_type == CheckpointType.LOCAL:
            return

        if not has_ema_state(context.user_state):
            return

        if context.state.cfg.checkpoint.async_save:
            raise NotImplementedError(
                "EMACallback checkpoint persistence does not yet support async_save=True."
            )

        save_ema_user_state(context.checkpoint_name, context.user_state)
        
    def on_checkpoint_load(self, context: CheckpointCallbackContext) -> None:
        if context.checkpoint_type not in (CheckpointType.GLOBAL, CheckpointType.FSDP_DTENSOR):
            return

        if context.state.cfg.checkpoint.finetune:
            return

        if context.state.train_state.step <= 0:
            return

        restored = load_ema_user_state(context.checkpoint_name, context.user_state)
        if not restored:
            print_rank_0(
                f"No EMA sidecar found in {context.checkpoint_name}; "
                "EMA will be re-initialized on train start."
            )