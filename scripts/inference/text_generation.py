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
import logging
import sys
from pathlib import Path


G_REPO_ROOT = Path(__file__).resolve().parents[2]
G_SRC_ROOT = G_REPO_ROOT / "src"
G_MCORE_ROOT = G_REPO_ROOT / "3rdparty" / "Megatron-LM"
for _path in (G_SRC_ROOT, G_MCORE_ROOT):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.append(str(_path))

import torch.distributed as dist
from megatron.core.inference.apis import MegatronLLM, SamplingParams
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines.static_engine import StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.text_generation_controllers.text_generation_controller import TextGenerationController

from megatron.bridge.inference.text_generation import (
    HFTokenizerAdapter,
    add_distributed_args,
    add_engine_args,
    add_model_loading_args,
    add_parallelism_args,
    add_prompt_args,
    add_sampling_args,
    build_inference_config,
    build_sampling_params,
    build_tokenizer,
    load_bridge_model,
    load_prompts,
    resolve_hf_model_path,
    validate_sequence_length,
)
from megatron.bridge.utils.activation_map import str_to_dtype
from megatron.bridge.utils.common_utils import maybe_initialize_distributed, print_rank_0


logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add Bridge offline text generation arguments."""
    add_model_loading_args(parser)
    add_parallelism_args(parser)
    add_prompt_args(parser)
    add_sampling_args(parser)
    add_engine_args(parser)
    add_distributed_args(parser)

    inference_group = parser.add_argument_group("Generation mode")
    inference_group.add_argument(
        "--use-legacy-generation",
        action="store_true",
        help="Use MCore legacy static-batching generation instead of the dynamic MegatronLLM engine.",
    )

    coordinator_group = parser.add_argument_group("Coordinator")
    coordinator_group.add_argument(
        "--use-coordinator",
        action="store_true",
        help="Use global-rank-0 coordinator mode.",
    )
    coordinator_group.add_argument("--coordinator-host", default=None, help="Coordinator ZMQ host.")
    coordinator_group.add_argument("--coordinator-port", type=int, default=None, help="Coordinator ZMQ port.")
    return parser


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


<<<<<<< HEAD
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


def _build_megatron_checkpoint_overrides(
    provider: object, args: argparse.Namespace, dtype: torch.dtype
) -> dict[str, object]:
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
    if args.attention_backend is not None:
        mp_overrides["attention_backend"] = AttnBackend[args.attention_backend]
    if hasattr(provider, "cache_mla_latents"):
        mp_overrides["cache_mla_latents"] = bool(getattr(provider, "cache_mla_latents"))
    if args.inference_moe_token_dispatcher_type is not None:
        mp_overrides["inference_moe_token_dispatcher_type"] = args.inference_moe_token_dispatcher_type
    return mp_overrides


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
        mp_overrides = _build_megatron_checkpoint_overrides(provider, args, dtype)
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


=======
>>>>>>> upstream/main
def _print_results(prompts: list[str], outputs: list[object]) -> None:
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    for idx, output in enumerate(outputs):
        prompt = prompts[idx] if idx < len(prompts) else ""
        generated_text = getattr(output, "generated_text", "")
        print_rank_0(f"[{idx}] Prompt: {prompt}")
        print_rank_0(f"[{idx}] Generated: {generated_text}")
    print_rank_0("=======================================")


def _longest_prompt_tokens(tokenizer: HFTokenizerAdapter, prompts: list[str]) -> int:
    return max(len(tokenizer.tokenize(prompt)) for prompt in prompts)


def _generate_with_dynamic_engine(
    args: argparse.Namespace,
    model: object,
    tokenizer: HFTokenizerAdapter,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> None:
    validate_sequence_length(
        longest_prompt_tokens=_longest_prompt_tokens(tokenizer, prompts),
        num_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
    )
    inference_config = build_inference_config(
        model=model,
        max_sequence_length=args.max_seq_length,
        max_batch_size=args.max_batch_size,
        num_prompts=len(prompts),
        tp=args.tp,
        block_size_tokens=args.block_size_tokens,
        kv_cache_buffer_size_gb=args.kv_cache_buffer_size_gb,
        max_tokens=args.max_tokens,
        return_log_probs=args.return_log_probs,
        enable_chunked_prefill=args.enable_chunked_prefill,
    )
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
    model: object,
    tokenizer: HFTokenizerAdapter,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> None:
    validate_sequence_length(
        longest_prompt_tokens=_longest_prompt_tokens(tokenizer, prompts),
        num_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
    )
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
    maybe_initialize_distributed(args.distributed_timeout_minutes)
    dtype = str_to_dtype(args.dtype)
    hf_model_path = resolve_hf_model_path(args.hf_model_path, args.megatron_model_path)
    prompts = load_prompts(
        args.prompt, args.prompt_file, args.prompt_file_num_truncate, ["Megatron Bridge inference is"]
    )

    print_rank_0(f"Loading model config/tokenizer from: {hf_model_path}")
    tokenizer = build_tokenizer(hf_model_path, args.trust_remote_code)
    model = load_bridge_model(
        hf_model_path=hf_model_path,
        megatron_model_path=args.megatron_model_path,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
        sequence_parallel=args.sequence_parallel,
        dtype=dtype,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        attention_backend=args.attention_backend,
        cache_mla_latents=args.cache_mla_latents,
        inference_moe_token_dispatcher_type=args.inference_moe_token_dispatcher_type,
    )
    sampling_params = build_sampling_params(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        skip_prompt_log_probs=args.skip_prompt_log_probs,
        num_tokens_to_generate=args.max_new_tokens,
        termination_id=args.termination_id if args.termination_id is not None else tokenizer.eod,
        top_n_logprobs=args.top_n_logprobs,
        stop_words=args.stop_words,
    )

    if args.use_legacy_generation:
        _generate_with_legacy_static_engine(args, model, tokenizer, prompts, sampling_params)
    else:
        _generate_with_dynamic_engine(args, model, tokenizer, prompts, sampling_params)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
