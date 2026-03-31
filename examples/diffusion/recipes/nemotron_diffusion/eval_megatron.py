"""
lm-eval-harness integration for Megatron GPTModel with NemotronDiffusionAttention.

Supports both AR and dLLM evaluation modes via the `--model megatron_dllm` flag.

Usage (single GPU):
    PYTHONPATH=/root/code/Megatron-Bridge/src:/root/code/Megatron-Bridge/examples:$PYTHONPATH \
    HF_ALLOW_CODE_EVAL=1 \
    python examples/diffusion/recipes/nemotron_diffusion/eval_megatron.py \
        --model megatron_dllm \
        --model_args "megatron_load_path=/lustre/fsw/portfolios/coreai/users/snorouzi/megatron_exp/ministral_3b_average_embd/iter_0012500,hf_model_id=/lustre/fsw/portfolios/nvr/users/snorouzi/models/Ministral-3-3B-Base-2512_converted,tokenizer=/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/yongganf/miscs/models/Nemotron-H-8B-Base-8K,mask_token_id=100,eval_mode=dllm,max_new_tokens=256,max_sequence_length=4096,diffusion_steps=256,temperature=0.0,block_length=32,shift_logits=False,neg_entropy=True,denoising_threshold=None,tp=1,pp=1,load_hf_weights=False" \
        --tasks gsm8k_cot \
        --batch_size 1 \
        --num_fewshot 8 \
        --limit 10 \
        --log_samples \
        --output_path /tmp/my_eval_results \
        --confirm_run_unsafe_code

Usage (DP=8, each GPU holds a full model copy):
    PYTHONPATH=/root/code/Megatron-Bridge/src:/root/code/Megatron-Bridge/examples:$PYTHONPATH \
    HF_ALLOW_CODE_EVAL=1 \
    accelerate launch --num_processes 8 examples/diffusion/recipes/nemotron_diffusion/eval_megatron.py \
        --model megatron_dllm \
        --model_args "...,tp=1" \
        --tasks gsm8k_cot --num_fewshot 8 --batch_size 1 --confirm_run_unsafe_code

Usage (TP=2, DP=4 across 8 GPUs):
    PYTHONPATH=/root/code/Megatron-Bridge/src:/root/code/Megatron-Bridge/examples:$PYTHONPATH \
    HF_ALLOW_CODE_EVAL=1 \
    torchrun --nproc_per_node=8 examples/diffusion/recipes/nemotron_diffusion/eval_megatron.py \
        --model megatron_dllm \
        --model_args "...,tp=2" \
        --tasks gsm8k_cot --num_fewshot 8 --batch_size 1 --confirm_run_unsafe_code
"""

import json
import time
import logging
import os
import sys
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union

# Patch antlr4 ATNDeserializer to accept both version 3 (antlr4==4.9 grammar)
# and version 4 (antlr4==4.11), so omegaconf grammars work regardless of installed version.
try:
    from antlr4.atn.ATNDeserializer import ATNDeserializer
    _orig_check = ATNDeserializer.checkVersion
    def _patched_check(self):
        try:
            _orig_check(self)
        except Exception:
            pass  # accept any ATN version
    ATNDeserializer.checkVersion = _patched_check
except Exception:
    pass


import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm import tqdm
from transformers import AutoTokenizer

import lm_eval.models.utils as utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from megatron.bridge import AutoBridge

# Register NemotronDiffusionBridge so AutoBridge uses NemotronDiffusionAttention
import megatron.bridge.diffusion.conversion.nemotron_diffusion.nemotron_diffusion_bridge  # noqa: F401

from diffusion.recipes.nemotron_diffusion.inference_nemotron_diffusion import (
    generate_ar,
    generate_dllm,
    set_tp_group,
)

eval_logger = logging.getLogger(__name__)

T = TypeVar("T", bound="MegatronDLLM")


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes")


class _DPAccelerator:
    """Accelerator shim that gathers only across DP-rank-0 processes.

    For TP>1, lm-eval's gather/barrier must only coordinate across the
    ``dp_world`` TP-rank-0 processes.  TP-follower ranks use a separate
    ``_TPFollowerAccelerator`` that participates in a barrier but provides
    dummy data for the gather.
    """

    def __init__(self, dp_group, dp_world_size):
        self.num_processes = dp_world_size
        self._dp_group = dp_group

    def wait_for_everyone(self):
        torch.distributed.barrier(group=self._dp_group)

    def gather(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, device="cuda")
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        gathered = [torch.zeros_like(tensor) for _ in range(self.num_processes)]
        torch.distributed.all_gather(gathered, tensor, group=self._dp_group)
        return torch.cat(gathered, dim=0)




@register_model("megatron_dllm")
class MegatronDLLM(LM):
    """lm-eval wrapper around a Megatron GPTModel loaded from a distcp checkpoint."""

    _instance = None  # singleton so TP re-construction is a no-op

    def __init__(
        self,
        megatron_load_path: str = "",
        hf_model_id: str = "Qwen/Qwen3-8B-Base",
        tokenizer: str = None,
        mask_token_id: int = 100,
        eval_mode: str = "dllm",
        batch_size: int = 1,
        max_new_tokens: int = 256,
        max_sequence_length: int = 4096,
        diffusion_steps: int = 256,
        temperature: float = 0.0,
        block_length: int = 32,
        shift_logits: bool = True,
        neg_entropy: bool = True,
        denoising_threshold: float = None,
        remasking: str = "low_confidence",
        tp: int = 1,
        pp: int = 1,
        add_bos_token: bool = False,
        nfe_log_path: str = None,
        latency_log_path: str = None,
        load_hf_weights: bool = True,
        cascade_schedule: str = None,
        **kwargs,
    ):
        # With TP>1, __init__ is called once explicitly (for ALL ranks) and
        # then again by lm-eval on TP-rank-0.  Skip the second call.
        if MegatronDLLM._instance is not None:
            self.__dict__.update(MegatronDLLM._instance.__dict__)
            return
        super().__init__()

        self.mask_token_id = int(mask_token_id)
        self.eval_mode = eval_mode
        self.batch_size_per_gpu = int(batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.max_sequence_length = int(max_sequence_length)
        self.diffusion_steps = int(diffusion_steps)
        self.temperature = float(temperature)
        self.block_length = int(block_length)
        self.shift_logits = _parse_bool(shift_logits)
        self.neg_entropy = _parse_bool(neg_entropy)
        self.remasking = remasking
        self.add_bos_token = _parse_bool(add_bos_token)
        self.nfe_log_path = nfe_log_path
        self.latency_log_path = latency_log_path
        self._latency_records = []

        if denoising_threshold in ("None", "none", None):
            self.denoising_threshold = None
        else:
            self.denoising_threshold = float(denoising_threshold)

        tp = int(tp)
        pp = int(pp)

        # Load tokenizer
        tok_path = tokenizer if tokenizer else hf_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Distributed setup ---
        if tp <= 1:
            # Pure DP: use accelerate.Accelerator (same approach as reference
            # eval.py).  Works with ``accelerate launch`` or single-process.
            accel_timeout = InitProcessGroupKwargs(timeout=timedelta(hours=2))
            self.accelerator = Accelerator(kwargs_handlers=[accel_timeout])
            self._device = self.accelerator.device
            global_rank = self.accelerator.process_index
            global_world = self.accelerator.num_processes
            self._rank = global_rank
            self._world_size = global_world
        else:
            # TP > 1: use torchrun (``torchrun --nproc_per_node N``).
            # ALL ranks run the full lm-eval loop so that TP partners
            # call model.forward() in sync.  gather/barrier use sub-groups.
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="nccl", init_method="env://",
                    timeout=timedelta(hours=2),
                )
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)

            global_rank = torch.distributed.get_rank()
            global_world = torch.distributed.get_world_size()
            dp_world = global_world // tp
            dp_rank = global_rank // tp

            # Create a DP-gather group for EACH TP-local position so that
            # all ranks (leaders and followers) can do a proper all_gather
            # across the dp_world.  TP partners process the same instances so
            # the gathered results are identical.
            my_dp_group = None
            for tp_offset in range(tp):
                ranks = [i * tp + tp_offset for i in range(dp_world)]
                g = torch.distributed.new_group(ranks)
                if global_rank % tp == tp_offset:
                    my_dp_group = g
            self.accelerator = _DPAccelerator(my_dp_group, dp_world)

            self._device = torch.device("cuda", local_rank)
            self._rank = dp_rank
            self._world_size = dp_world

        tp_local = global_rank % tp
        dp_rank = self._rank
        dp_world = self._world_size

        # --- Load Megatron model via AutoBridge ---
        load_hf = _parse_bool(load_hf_weights)
        if load_hf:
            eval_logger.info(f"Loading HF weights directly from {hf_model_id} via AutoBridge...")
        else:
            eval_logger.info(f"Loading Megatron model from {megatron_load_path} via AutoBridge...")

        bridge = AutoBridge.from_hf_pretrained(
            hf_model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
        )

        model_provider = bridge.to_megatron_provider(load_weights=load_hf)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.params_dtype = torch.bfloat16
        model_provider.seq_length = int(max_sequence_length)
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)

        if load_hf:
            megatron_models = model_provider.provide_distributed_model(wrap_with_ddp=False)
        else:
            from megatron.bridge.training.model_load_save import build_and_load_model
            megatron_models = build_and_load_model(
                checkpoint_path=megatron_load_path,
                model_cfg=model_provider,
                skip_temp_dist_context=True,
                dist_ckpt_strictness="ignore_all",
            )
        if isinstance(megatron_models, list):
            self.model = megatron_models[0].cuda().eval()
        else:
            self.model = megatron_models.cuda().eval()
        self._device = torch.device("cuda")

        # --- Load cascade schedule models ---
        # cascade_schedule format: flat |-separated list read in groups of 3:
        #   ckpt1|n_steps1|hf_id1|ckpt2|n_steps2|hf_id2|...
        # e.g. "path/to/14b|8|mistralai/Ministral-3-14B-Base-2512|path/to/8b|8|path/to/8b_hf|path/to/3b|16|mistralai/3b"
        self.model_schedule = None
        if cascade_schedule not in (None, "None", "none", ""):
            from megatron.bridge.training.model_load_save import build_and_load_model
            schedule_entries = []
            parts = str(cascade_schedule).split("|")
            if len(parts) % 3 != 0:
                raise ValueError(
                    f"cascade_schedule must have entries in groups of 3 (ckpt|n_steps|hf_id), "
                    f"got {len(parts)} pipe-separated tokens"
                )
            for i in range(0, len(parts), 3):
                ckpt_path = parts[i]
                n_steps = int(parts[i + 1])
                sched_hf_id = parts[i + 2]
                eval_logger.info(f"Loading cascade model from {ckpt_path} using HF config {sched_hf_id} ({n_steps} steps)")
                eval_logger.info(f"  cascade match check: sched_hf_id={repr(sched_hf_id)} vs hf_model_id={repr(hf_model_id)}")
                if os.path.normpath(sched_hf_id) == os.path.normpath(hf_model_id):
                    # Reuse primary model provider to avoid re-loading AutoBridge
                    # (needed when local HF path lacks _name_or_path in config.json)
                    sched_provider = model_provider
                else:
                    sched_bridge = AutoBridge.from_hf_pretrained(
                        sched_hf_id, trust_remote_code=True, torch_dtype=torch.bfloat16
                    )
                    sched_provider = sched_bridge.to_megatron_provider(load_weights=False)
                    sched_provider.tensor_model_parallel_size = tp
                    sched_provider.pipeline_model_parallel_size = pp
                    sched_provider.pipeline_dtype = torch.bfloat16
                    sched_provider.params_dtype = torch.bfloat16
                    sched_provider.seq_length = int(max_sequence_length)
                    sched_provider.finalize()
                sched_megatron_models = build_and_load_model(
                    checkpoint_path=ckpt_path,
                    model_cfg=sched_provider,
                    skip_temp_dist_context=True,
                    dist_ckpt_strictness="ignore_all",
                )
                if isinstance(sched_megatron_models, list):
                    sched_model = sched_megatron_models[0].cuda().eval()
                else:
                    sched_model = sched_megatron_models.cuda().eval()
                schedule_entries.append((sched_model, n_steps))
                eval_logger.info(f"Loaded cascade model from {ckpt_path} ({n_steps} steps)")
            self.model_schedule = schedule_entries

        self._tp = tp
        self._tp_local = tp_local

        # Set up TP broadcast so _model_forward syncs tokens across TP peers.
        if tp > 1:
            from megatron.core import parallel_state as mpu
            tp_group = mpu.get_tensor_model_parallel_group()
            tp_src = (global_rank // tp) * tp
            set_tp_group(tp_group, src_global_rank=tp_src)

            # Monkey-patch torch.distributed.gather_object so lm-eval's
            # direct calls use our DP sub-group instead of the default
            # (world-size) process group.  Applied AFTER model loading to
            # avoid interfering with dist_checkpointing.
            _orig_gather_object = torch.distributed.gather_object
            _my_dp_group = self.accelerator._dp_group

            def _patched_gather_object(
                obj, object_gather_list=None, dst=None, group=None, group_dst=None,
            ):
                if group is None:
                    group = _my_dp_group
                    # lm-eval passes dst=0 (global rank 0) which may not be in
                    # this DP sub-group.  Use group_dst=0 instead.
                    if dst is not None and group_dst is None:
                        group_dst = dst
                        dst = None
                _orig_gather_object(
                    obj, object_gather_list, dst=dst, group=group, group_dst=group_dst,
                )

            torch.distributed.gather_object = _patched_gather_object

        MegatronDLLM._instance = self

        eval_logger.info(
            f"Megatron model loaded on global_rank={global_rank}, "
            f"dp_rank={dp_rank}/{dp_world}, tp_local={tp_local}/{tp}. "
            f"eval_mode={eval_mode}, "
            f"mask_id={self.mask_token_id}, block_length={self.block_length}, "
            f"shift_logits={self.shift_logits}, neg_entropy={self.neg_entropy}, "
            f"denoising_threshold={self.denoising_threshold}"
        )

    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        additional_config = {} if additional_config is None else additional_config
        from lm_eval.utils import simple_parse_args_string
        args = simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.max_sequence_length

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        if self.add_bos_token and self.tokenizer.bos_token:
            prompts = [self.tokenizer.bos_token + p for p in prompts]

        prompt_ids = self.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).input_ids

        if prompt_ids.shape[1] > self.max_sequence_length - self.max_new_tokens:
            eval_logger.warning(
                f"Prompt length {prompt_ids.shape[1]} exceeds limit, truncating from left."
            )
            prompt_ids = prompt_ids[:, -(self.max_sequence_length - self.max_new_tokens) :]

        prompt_ids = prompt_ids.to(self.device)

        torch.cuda.synchronize()
        _t_gen_start = time.perf_counter()
        _timing = {"prefill_ms": 0.0, "denoise_ms": 0.0, "kv_update_ms": 0.0}
        if self.eval_mode == "ar":
            out = generate_ar(
                model=self.model,
                prompt=prompt_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            nfe = self.max_new_tokens
        else:
            out, nfe, _timing = generate_dllm(
                model=self.model,
                prompt=prompt_ids,
                gen_length=self.max_new_tokens,
                block_length=self.block_length,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                remasking=self.remasking,
                mask_id=self.mask_token_id,
                threshold=self.denoising_threshold,
                shift_logits=self.shift_logits,
                neg_entropy=self.neg_entropy,
                model_schedule=self.model_schedule,
            )
        torch.cuda.synchronize()
        _latency_ms = (time.perf_counter() - _t_gen_start) * 1000.0

        generated_tokens = out[:, prompt_ids.shape[1] :]
        tokenized_out = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

        if not isinstance(tokenized_out, str):
            tokenized_out = str(tokenized_out) if tokenized_out is not None else ""

        if self.nfe_log_path:
            self._log_nfe(nfe)
        self._latency_records.append({
            "latency_ms": _latency_ms,
            "batch_size": prompt_ids.shape[0],
            "tokens_generated": self.max_new_tokens,
            "ms_per_token": _latency_ms / (prompt_ids.shape[0] * self.max_new_tokens),
            "nfe": float(nfe),
            "prefill_ms": _timing["prefill_ms"],
            "denoise_ms": _timing["denoise_ms"],
            "kv_update_ms": _timing["kv_update_ms"],
        })
        if self.latency_log_path:
            self._log_latency(_latency_ms, prompt_ids.shape[0], self.max_new_tokens)

        return [tokenized_out]

    def _log_nfe(self, nfe):
        try:
            os.makedirs(os.path.dirname(self.nfe_log_path), exist_ok=True)
            if os.path.exists(self.nfe_log_path):
                try:
                    with open(self.nfe_log_path, "r") as f:
                        log_data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    log_data = []
            else:
                log_data = []
            log_data.append(nfe)
            with open(self.nfe_log_path, "w") as f:
                json.dump(log_data, f)
        except Exception as e:
            eval_logger.warning(f"Failed to log NFE: {e}")

    def _log_latency(self, latency_ms: float, batch_size: int, tokens_generated: int):
        try:
            os.makedirs(os.path.dirname(self.latency_log_path), exist_ok=True)
            if os.path.exists(self.latency_log_path):
                try:
                    with open(self.latency_log_path, "r") as f:
                        log_data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    log_data = []
            else:
                log_data = []
            log_data.append({
                "latency_ms": latency_ms,
                "batch_size": batch_size,
                "tokens_generated": tokens_generated,
                "ms_per_token": latency_ms / (batch_size * tokens_generated),
            })
            with open(self.latency_log_path, "w") as f:
                json.dump(log_data, f)
        except Exception as e:
            eval_logger.warning(f"Failed to log latency: {e}")

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(list(contexts))

            for i, r in enumerate(responses):
                for s in gen_args[0]["until"]:
                    r = r.split(s)[0]
                responses[i] = r

            res.extend(responses)
            pbar.update(len(contexts))

        if self._latency_records:
            lats_t = torch.tensor(
                [r["latency_ms"] for r in self._latency_records], device=self._device
            )
            mpt_t = torch.tensor(
                [r["ms_per_token"] for r in self._latency_records], device=self._device
            )
            nfe_t = torch.tensor(
                [r["nfe"] for r in self._latency_records], device=self._device
            )
            prefill_t = torch.tensor(
                [r["prefill_ms"] for r in self._latency_records], device=self._device
            )
            denoise_t = torch.tensor(
                [r["denoise_ms"] for r in self._latency_records], device=self._device
            )
            kv_update_t = torch.tensor(
                [r["kv_update_ms"] for r in self._latency_records], device=self._device
            )
            lats_t = self.accelerator.gather(lats_t)
            mpt_t = self.accelerator.gather(mpt_t)
            nfe_t = self.accelerator.gather(nfe_t)
            prefill_t = self.accelerator.gather(prefill_t)
            denoise_t = self.accelerator.gather(denoise_t)
            kv_update_t = self.accelerator.gather(kv_update_t)
            if self._rank == 0:
                lats = lats_t.tolist()
                mpt = mpt_t.tolist()
                nfes = nfe_t.tolist()
                prefills = prefill_t.tolist()
                denoises = denoise_t.tolist()
                kv_updates = kv_update_t.tolist()
                n = len(lats)
                def _s(vals):
                    m = sum(vals) / n
                    return m, (sum((x - m) ** 2 for x in vals) / n) ** 0.5, min(vals), max(vals)
                ml, sl, mnl, mxl = _s(lats)
                mm, sm, mnm, mxm = _s(mpt)
                mn, sn, mnn, mxn = _s(nfes)
                mp, sp, mnp, mxp = _s(prefills)
                md, sd, mnd, mxd = _s(denoises)
                mk, sk, mnk, mxk = _s(kv_updates)
                print(
                    f"\n{'=' * 60}\n"
                    f"  Latency summary ({n} samples, all DP ranks)\n"
                    f"  latency/sample (ms)      : mean={ml:.1f}  std={sl:.1f}  min={mnl:.1f}  max={mxl:.1f}\n"
                    f"  ms/token (incl. prefill) : mean={mm:.3f}  std={sm:.3f}  min={mnm:.3f}  max={mxm:.3f}\n"
                    f"  NFE                      : mean={mn:.1f}  std={sn:.1f}  min={mnn:.1f}  max={mxn:.1f}\n"
                    f"  --- phase breakdown (mean ms/sample) ---\n"
                    f"  prefill                  : {mp:.1f} ms  ({100*mp/ml:.1f}%)\n"
                    f"  denoise (NFE steps)      : {md:.1f} ms  ({100*md/ml:.1f}%)\n"
                    f"  kv_cache update          : {mk:.1f} ms  ({100*mk/ml:.1f}%)\n"
                    f"  other (overhead)         : {ml-mp-md-mk:.1f} ms  ({100*(ml-mp-md-mk)/ml:.1f}%)\n"
                    f"{'=' * 60}",
                    flush=True,
                )
        return res

    # ------------------------------------------------------------------
    # Log-likelihood (required by lm-eval interface)
    # ------------------------------------------------------------------

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        eval_logger.warning(
            "loglikelihood is not fully supported for diffusion models. "
            "Returning dummy values."
        )
        return [(0.0, False)] * len(requests)

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        eval_logger.warning(
            "loglikelihood_rolling is not supported for diffusion models. "
            "Returning dummy values."
        )
        return [0.0] * len(requests)


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate

    # lm-eval's cli_evaluate creates the model, which sets self._tp_local.
    # For TP>1, follower ranks enter tp_follower_loop inside __init__ wouldn't
    # work because we need the model first.  Instead we parse tp from the CLI
    # args to decide if this rank is a follower AFTER model construction.
    # The approach: cli_evaluate drives everything including model construction.
    # We hook via a custom __main__ wrapper.

    # Parse tp from --model_args to detect follower before lm-eval begins eval.
    _tp_val = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--model_args" and i + 1 < len(sys.argv):
            for kv in sys.argv[i + 1].split(","):
                if kv.startswith("tp="):
                    _tp_val = int(kv.split("=", 1)[1])
            break

    _global_rank = int(os.environ.get("RANK", "0"))
    _tp_local_rank = _global_rank % _tp_val

    if _tp_val > 1:
        # With TP>1, ALL ranks must construct the model together (collective
        # ops like new_group / initialize_model_parallel require it).
        # ALL ranks then run cli_evaluate — TP partners share the same dp_rank
        # so they process the same instances, keeping model.forward() in sync.
        _model_args_str = ""
        for i, arg in enumerate(sys.argv):
            if arg == "--model_args" and i + 1 < len(sys.argv):
                _model_args_str = sys.argv[i + 1]
                break

        kwargs = {}
        for kv in _model_args_str.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                kwargs[k] = v

        # All ranks construct the model simultaneously (singleton prevents
        # lm-eval from constructing a second time).
        MegatronDLLM(**kwargs)
        cli_evaluate()
    else:
        cli_evaluate()
