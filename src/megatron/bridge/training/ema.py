import torch
from megatron.bridge.training.callbacks import Callback
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

        updates = context.user_state.get("ema_updates", 0)
        skips = context.user_state.get("ema_skipped_iters", 0)

        if updates == 0 and context.state.train_state.step > self.start_step:
            inferred_updates = max(0, context.state.train_state.step - self.start_step - skips)
            context.user_state["ema_updates"] = inferred_updates

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