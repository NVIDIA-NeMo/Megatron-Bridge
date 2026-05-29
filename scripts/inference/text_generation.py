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

"""Bridge-backed offline text generation using the MCore high-level inference API."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path


G_REPO_ROOT = Path(__file__).resolve().parents[2]
G_SRC_ROOT = G_REPO_ROOT / "src"
G_MCORE_ROOT = G_REPO_ROOT / "3rdparty" / "Megatron-LM"
for _path in (G_SRC_ROOT, G_MCORE_ROOT):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.append(str(_path))

import torch
import torch.distributed as dist
from megatron.core.inference.apis import MegatronLLM, SamplingParams
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines.static_engine import StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.text_generation_controllers.text_generation_controller import TextGenerationController
from megatron.core.transformer.enums import AttnBackend
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.utils.checkpoint_utils import get_hf_model_id_from_checkpoint
from megatron.bridge.utils.common_utils import disable_mtp_for_inference, get_local_rank_preinit, print_rank_0


logger = logging.getLogger(__name__)


class HuggingFaceTextTokenizer:
    """Adapter exposing the tokenizer methods expected by MCore text generation."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def eod(self) -> int | None:
        """End-of-document token id used for early termination."""
        return self._tokenizer.eos_token_id

    @property
    def bos(self) -> int | None:
        """Beginning-of-sequence token id."""
        return self._tokenizer.bos_token_id

    @property
    def vocab_size(self) -> int:
        """Tokenizer vocabulary size."""
        return len(self._tokenizer)

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token ids."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token ids back to text."""
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add Bridge offline text generation arguments."""
    model_group = parser.add_argument_group("Model loading")
    model_group.add_argument(
        "--hf_model_path",
        "--hf-model-path",
        dest="hf_model_path",
        default=None,
        help=(
            "Hugging Face model id/path used for config and tokenizer. Required unless checkpoint metadata records it."
        ),
    )
    model_group.add_argument(
        "--megatron_model_path",
        "--megatron-model-path",
        dest="megatron_model_path",
        default=None,
        help="Optional Megatron Bridge checkpoint path. If omitted, load and convert HF weights in-process.",
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=None,
        help="Allow custom Hugging Face model/tokenizer code for trusted repositories.",
    )
    model_group.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Model parameter dtype for in-process HF conversion and provider setup.",
    )

    parallel_group = parser.add_argument_group("Parallelism")
    parallel_group.add_argument("--tp", type=int, default=1, help="Tensor model parallel size.")
    parallel_group.add_argument("--pp", type=int, default=1, help="Pipeline model parallel size.")
    parallel_group.add_argument("--ep", type=int, default=1, help="Expert model parallel size.")
    parallel_group.add_argument("--etp", type=int, default=1, help="Expert tensor parallel size.")
    parallel_group.add_argument("--sequence-parallel", action="store_true", help="Enable sequence parallelism.")
    parallel_group.add_argument("--seed", type=int, default=0, help="Model-parallel RNG seed.")
    parallel_group.add_argument(
        "--cache-mla-latents",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache MLA latents for dynamic inference. Defaults on for MLA models.",
    )

    prompt_group = parser.add_argument_group("Prompts")
    prompt_group.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt text. May be provided multiple times. Defaults to a short prompt if no prompt file is set.",
    )
    prompt_group.add_argument(
        "--prompt_file",
        "--prompt-file",
        dest="prompt_file",
        default=None,
        help="Line-oriented prompt file. JSONL lines use the `text` or `prompt` field; other lines are raw prompts.",
    )
    prompt_group.add_argument(
        "--prompt-file-num-truncate",
        type=int,
        default=None,
        help="Read at most this many prompts from --prompt_file.",
    )

    sampling_group = parser.add_argument_group("Sampling")
    sampling_group.add_argument("--max_new_tokens", type=int, default=30, help="Maximum generated tokens per prompt.")
    sampling_group.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    sampling_group.add_argument("--top_p", type=float, default=0.0, help="Top-p sampling.")
    sampling_group.add_argument("--top_k", type=int, default=1, help="Top-k sampling.")
    sampling_group.add_argument("--return-log-probs", action="store_true", help="Return token log probabilities.")
    sampling_group.add_argument("--skip-prompt-log-probs", action="store_true", help="Skip prompt log probabilities.")
    sampling_group.add_argument("--top-n-logprobs", type=int, default=0, help="Return top-n logprobs.")
    sampling_group.add_argument("--termination-id", type=int, default=None, help="Override tokenizer EOD id.")
    sampling_group.add_argument(
        "--stop-words",
        nargs="+",
        default=None,
        help="Stop words that terminate generation when produced.",
    )

    inference_group = parser.add_argument_group("Inference")
    inference_group.add_argument(
        "--use-legacy-generation",
        action="store_true",
        help="Use MCore legacy static-batching generation instead of the dynamic MegatronLLM engine.",
    )
    inference_group.add_argument(
        "--attention-backend",
        choices=("auto", "flash", "fused", "unfused", "local"),
        default=None,
        help="Override the provider attention backend before constructing the Megatron model.",
    )
    inference_group.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Prompt plus generation length limit.",
    )
    inference_group.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximum active requests. Defaults to the number of prompts.",
    )
    inference_group.add_argument("--max_tokens", type=int, default=None, help="Maximum active tokens.")
    inference_group.add_argument(
        "--block_size_tokens",
        type=int,
        default=256,
        help="KV-cache block size in tokens.",
    )
    inference_group.add_argument(
        "--kv_cache_buffer_size_gb",
        type=float,
        default=20.0,
        help="GPU buffer size reserved for KV cache.",
    )
    inference_group.add_argument("--enable-chunked-prefill", action="store_true", help="Enable chunked prefill.")
    inference_group.add_argument(
        "--inference-moe-token-dispatcher-type",
        choices=("nccl", "nvls"),
        default=None,
        help="Override the MCore MoE token dispatcher used during inference.",
    )

    coordinator_group = parser.add_argument_group("Coordinator")
    coordinator_group.add_argument(
        "--use-coordinator",
        action="store_true",
        help="Use global-rank-0 coordinator mode.",
    )
    coordinator_group.add_argument("--coordinator-host", default=None, help="Coordinator ZMQ host.")
    coordinator_group.add_argument("--coordinator-port", type=int, default=None, help="Coordinator ZMQ port.")

    distributed_group = parser.add_argument_group("Distributed")
    distributed_group.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=60,
        help="Process-group timeout in minutes for slow multi-node model setup.",
    )
    return parser


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.use_legacy_generation and args.use_coordinator:
        raise ValueError("--use-coordinator is only supported by dynamic generation.")
    if args.ep > 1 and not args.use_coordinator and not args.use_legacy_generation:
        raise ValueError("--use-coordinator is required when --ep is greater than 1.")
    if (args.coordinator_host is not None or args.coordinator_port is not None) and not args.use_coordinator:
        raise ValueError("--coordinator-host/--coordinator-port require --use-coordinator.")
    if args.top_n_logprobs > 0 and not args.return_log_probs:
        raise ValueError("--top-n-logprobs requires --return-log-probs.")
    if args.distributed_timeout_minutes <= 0:
        raise ValueError("--distributed-timeout-minutes must be positive.")


def _maybe_initialize_distributed(timeout_minutes: int) -> None:
    if not dist.is_available() or dist.is_initialized():
        return

    os.environ["RANK"] = os.environ.get("RANK", "0")
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    torch.cuda.set_device(get_local_rank_preinit())
    dist.init_process_group("nccl", timeout=timedelta(minutes=timeout_minutes))


def _resolve_hf_model_path(args: argparse.Namespace) -> str:
    if args.hf_model_path:
        return args.hf_model_path
    if args.megatron_model_path:
        hf_model_path = get_hf_model_id_from_checkpoint(args.megatron_model_path)
        if hf_model_path:
            return hf_model_path
    raise ValueError("--hf_model_path is required when checkpoint metadata does not include model.hf_model_id")


def _get_prompt_from_json_line(line: str) -> str | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    for key in ("text", "prompt", "input"):
        prompt = value.get(key)
        if isinstance(prompt, str):
            return prompt
    return None


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt)
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        with prompt_path.open("r", encoding="utf-8") as prompt_file:
            for line in prompt_file:
                raw_prompt = line.rstrip("\n")
                if not raw_prompt:
                    continue
                prompts.append(_get_prompt_from_json_line(raw_prompt) or raw_prompt)
                if args.prompt_file_num_truncate is not None and len(prompts) >= args.prompt_file_num_truncate:
                    break
    if not prompts:
        prompts.append("Megatron Bridge inference is")
    return prompts


def _build_sampling_params(args: argparse.Namespace, tokenizer: HuggingFaceTextTokenizer) -> SamplingParams:
    termination_id = args.termination_id if args.termination_id is not None else tokenizer.eod
    return SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        skip_prompt_log_probs=args.skip_prompt_log_probs,
        num_tokens_to_generate=args.max_new_tokens,
        termination_id=termination_id,
        top_n_logprobs=args.top_n_logprobs,
        stop_words=args.stop_words,
    )


def _apply_provider_parallelism(provider: object, args: argparse.Namespace, dtype: torch.dtype) -> None:
    setattr(provider, "tensor_model_parallel_size", args.tp)
    setattr(provider, "pipeline_model_parallel_size", args.pp)
    setattr(provider, "expert_model_parallel_size", args.ep)
    setattr(provider, "expert_tensor_parallel_size", args.etp)
    setattr(provider, "sequence_parallel", args.sequence_parallel)
    setattr(provider, "params_dtype", dtype)
    setattr(provider, "pipeline_dtype", dtype)
    setattr(provider, "bf16", dtype == torch.bfloat16)
    setattr(provider, "fp16", dtype == torch.float16)
    if args.attention_backend is not None:
        setattr(provider, "attention_backend", AttnBackend[args.attention_backend])
    is_mla_model = bool(getattr(provider, "multi_latent_attention", False))
    use_mla_latent_cache = args.cache_mla_latents
    if use_mla_latent_cache is None:
        use_mla_latent_cache = is_mla_model
    if args.cache_mla_latents is not None or is_mla_model or hasattr(provider, "cache_mla_latents"):
        setattr(provider, "cache_mla_latents", use_mla_latent_cache)
    if args.inference_moe_token_dispatcher_type is not None:
        if not hasattr(provider, "inference_moe_token_dispatcher_type"):
            raise ValueError(
                "--inference-moe-token-dispatcher-type was set, but the selected provider "
                "does not expose inference_moe_token_dispatcher_type."
            )
        setattr(provider, "inference_moe_token_dispatcher_type", args.inference_moe_token_dispatcher_type)


def _prepare_model_list(model_list: list[torch.nn.Module]) -> torch.nn.Module:
    if len(model_list) != 1:
        raise ValueError("MegatronLLM supports one local model stage; virtual pipeline parallelism is not supported.")
    model = model_list[0].cuda()
    model.eval()
    disable_mtp_for_inference(model)
    if hasattr(model, "config"):
        model.config.grad_scale_func = None
    return model


def _load_model(args: argparse.Namespace, hf_model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    trust_remote_code = is_safe_repo(hf_path=hf_model_path, trust_remote_code=args.trust_remote_code)

    if args.megatron_model_path:
        config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=trust_remote_code)
        bridge = AutoBridge.from_hf_config(config)
        provider = bridge.to_megatron_provider(load_weights=False)
        _apply_provider_parallelism(provider, args, dtype)
        provider.finalize()
        provider.initialize_model_parallel(seed=args.seed)
        mp_overrides = {
            "tensor_model_parallel_size": args.tp,
            "pipeline_model_parallel_size": args.pp,
            "expert_model_parallel_size": args.ep,
            "expert_tensor_parallel_size": args.etp,
            "sequence_parallel": args.sequence_parallel,
            "params_dtype": dtype,
            "pipeline_dtype": dtype,
            "bf16": dtype == torch.bfloat16,
            "fp16": dtype == torch.float16,
        }
        if hasattr(provider, "cache_mla_latents"):
            mp_overrides["cache_mla_latents"] = bool(getattr(provider, "cache_mla_latents"))
        if args.inference_moe_token_dispatcher_type is not None:
            mp_overrides["inference_moe_token_dispatcher_type"] = args.inference_moe_token_dispatcher_type
        model_list = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides=mp_overrides,
            wrap_with_ddp=False,
        )
    else:
        bridge = AutoBridge.from_hf_pretrained(
            hf_model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        provider = bridge.to_megatron_provider(load_weights=True)
        _apply_provider_parallelism(provider, args, dtype)
        provider.finalize()
        provider.initialize_model_parallel(seed=args.seed)
        model_list = provider.provide_distributed_model(wrap_with_ddp=False)

    return _prepare_model_list(model_list)


def _build_tokenizer(hf_model_path: str, trust_remote_code: bool | None) -> HuggingFaceTextTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_path,
        trust_remote_code=is_safe_repo(hf_path=hf_model_path, trust_remote_code=trust_remote_code),
    )
    return HuggingFaceTextTokenizer(tokenizer)


def _validate_sequence_length(
    args: argparse.Namespace,
    tokenizer: HuggingFaceTextTokenizer,
    prompts: list[str],
) -> None:
    longest_prompt = max(len(tokenizer.tokenize(prompt)) for prompt in prompts)
    required_sequence_length = longest_prompt + args.max_new_tokens
    if required_sequence_length > args.max_seq_length:
        raise ValueError(
            f"Longest prompt plus generation needs {required_sequence_length} tokens, "
            f"but --max_seq_length is {args.max_seq_length}."
        )


def _build_inference_config(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: HuggingFaceTextTokenizer,
    prompts: list[str],
) -> InferenceConfig:
    _validate_sequence_length(args, tokenizer, prompts)
    if getattr(getattr(model, "config", None), "cache_mla_latents", False) and args.block_size_tokens != 64:
        print_rank_0(
            f"Using block size 64 instead of {args.block_size_tokens} because MCore dynamic inference "
            "requires 64-token blocks when caching MLA latents."
        )
        args.block_size_tokens = 64

    max_requests = args.max_batch_size or len(prompts)
    if max_requests % args.tp != 0:
        rounded_max_requests = ((max_requests + args.tp - 1) // args.tp) * args.tp
        if args.max_batch_size is not None:
            raise ValueError(
                f"--max_batch_size must be divisible by --tp ({args.tp}); got --max_batch_size {args.max_batch_size}."
            )
        print_rank_0(
            f"Rounding max batch size from {max_requests} to {rounded_max_requests} "
            f"so it is divisible by tensor parallel size {args.tp}."
        )
        max_requests = rounded_max_requests

    return InferenceConfig(
        block_size_tokens=args.block_size_tokens,
        buffer_size_gb=args.kv_cache_buffer_size_gb,
        max_requests=max_requests,
        max_tokens=args.max_tokens,
        max_sequence_length=args.max_seq_length,
        mamba_inference_state_config=MambaInferenceStateConfig.from_model(model),
        pg_collection=getattr(model, "pg_collection", None),
        materialize_only_last_token_logits=not args.return_log_probs,
        enable_chunked_prefill=args.enable_chunked_prefill,
    )


def _print_results(prompts: list[str], outputs: list[object]) -> None:
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    for idx, output in enumerate(outputs):
        prompt = prompts[idx] if idx < len(prompts) else ""
        generated_text = getattr(output, "generated_text", "")
        print_rank_0(f"[{idx}] Prompt: {prompt}")
        print_rank_0(f"[{idx}] Generated: {generated_text}")
    print_rank_0("=======================================")


def _generate_with_dynamic_engine(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: HuggingFaceTextTokenizer,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> None:
    inference_config = _build_inference_config(args, model, tokenizer, prompts)
    with MegatronLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=inference_config,
        use_coordinator=args.use_coordinator,
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
    ) as llm:
        if llm.is_primary_rank:
            outputs = llm.generate(prompts, sampling_params)
            _print_results(prompts, outputs)


def _generate_with_legacy_static_engine(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: HuggingFaceTextTokenizer,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> None:
    _validate_sequence_length(args, tokenizer, prompts)
    max_batch_size = args.max_batch_size or len(prompts)
    inference_context = StaticInferenceContext(
        max_batch_size=max_batch_size,
        max_sequence_length=args.max_seq_length,
    )
    inference_wrapped_model = GPTInferenceWrapper(model, inference_context=inference_context)
    controller = TextGenerationController(inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer)
    engine = StaticInferenceEngine(
        text_generation_controller=controller,
        max_batch_size=max_batch_size,
        random_seed=args.seed,
        legacy=True,
    )
    outputs = engine.generate(prompts=prompts, sampling_params=sampling_params)
    _print_results(prompts, outputs)


def main() -> None:
    """Run Bridge-backed synchronous offline text generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()

    logging.basicConfig(level=logging.INFO)
    _validate_args(args)
    _maybe_initialize_distributed(args.distributed_timeout_minutes)
    dtype = _dtype_from_name(args.dtype)
    hf_model_path = _resolve_hf_model_path(args)
    prompts = _load_prompts(args)

    print_rank_0(f"Loading model config/tokenizer from: {hf_model_path}")
    tokenizer = _build_tokenizer(hf_model_path, args.trust_remote_code)
    model = _load_model(args, hf_model_path, dtype)
    sampling_params = _build_sampling_params(args, tokenizer)

    if args.use_legacy_generation:
        _generate_with_legacy_static_engine(args, model, tokenizer, prompts, sampling_params)
    else:
        _generate_with_dynamic_engine(args, model, tokenizer, prompts, sampling_params)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
