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

"""Shared building blocks for Bridge-backed MCore text generation.

This module holds the model-loading, tokenizer, distributed-init, prompt, sampling, and
``InferenceConfig`` helpers shared by the standalone scripts under ``scripts/inference/``
(``text_generation.py`` and ``async_text_generation.py``). It depends only on
``megatron.core`` and ``megatron.bridge`` -- never on the Megatron-LM reference layer
(``megatron.inference`` / ``megatron.training``) -- so the scripts stay on the Bridge/Core
layer and are insulated from refactors of Megatron-LM's inference entry points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from megatron.core.inference.apis import SamplingParams
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import get_attr_wrapped_model
from transformers import AutoConfig, AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.inference._tokenizer import HFTokenizerAdapter
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.utils.checkpoint_utils import get_hf_model_id_from_checkpoint
from megatron.bridge.utils.common_utils import disable_mtp_for_inference, print_rank_0


__all__ = [
    "HFTokenizerAdapter",
    "resolve_hf_model_path",
    "build_tokenizer",
    "load_prompts",
    "load_bridge_model",
    "build_inference_config",
    "build_sampling_params",
    "validate_sequence_length",
    "add_model_loading_args",
    "add_parallelism_args",
    "add_prompt_args",
    "add_sampling_args",
    "add_engine_args",
    "add_distributed_args",
]


# Dtype string -> torch.dtype: use the shared resolver (``str_to_dtype``) instead of a local map.
# Distributed bring-up: use ``megatron.bridge.utils.common_utils.maybe_initialize_distributed``.
# Tokenizer adapter: use ``megatron.bridge.inference._tokenizer.HFTokenizerAdapter``.


def resolve_hf_model_path(hf_model_path: str | None, megatron_model_path: str | None) -> str:
    """Resolve the HF model id used for config/tokenizer, falling back to checkpoint metadata."""
    if hf_model_path:
        return hf_model_path
    if megatron_model_path:
        resolved = get_hf_model_id_from_checkpoint(megatron_model_path)
        if resolved:
            return resolved
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


def load_prompts(
    prompts: list[str] | None,
    prompt_file: str | None,
    prompt_file_num_truncate: int | None,
    default_prompts: list[str],
) -> list[str]:
    """Collect prompts from explicit args and/or a line-oriented (optionally JSONL) file."""
    collected = list(prompts or [])
    if prompt_file:
        file_prompts_collected = 0
        with Path(prompt_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                raw_prompt = line.rstrip("\n")
                if not raw_prompt:
                    continue
                if prompt_file_num_truncate is not None and file_prompts_collected >= prompt_file_num_truncate:
                    break
                collected.append(_get_prompt_from_json_line(raw_prompt) or raw_prompt)
                file_prompts_collected += 1
    if not collected:
        collected = list(default_prompts)
    return collected


def build_tokenizer(hf_model_path: str, trust_remote_code: bool | None) -> HFTokenizerAdapter:
    """Build the HF-backed tokenizer adapter for MCore text generation."""
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_path,
        trust_remote_code=is_safe_repo(hf_path=hf_model_path, trust_remote_code=trust_remote_code),
    )
    return HFTokenizerAdapter(tokenizer)


def _apply_provider_parallelism(
    provider: object,
    *,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    sequence_parallel: bool,
    dtype: torch.dtype,
    attention_backend: str | None,
    cache_mla_latents: bool | None,
    inference_moe_token_dispatcher_type: str | None,
) -> None:
    setattr(provider, "tensor_model_parallel_size", tp)
    setattr(provider, "pipeline_model_parallel_size", pp)
    setattr(provider, "expert_model_parallel_size", ep)
    setattr(provider, "expert_tensor_parallel_size", etp)
    setattr(provider, "sequence_parallel", sequence_parallel)
    setattr(provider, "params_dtype", dtype)
    setattr(provider, "pipeline_dtype", dtype)
    setattr(provider, "bf16", dtype == torch.bfloat16)
    setattr(provider, "fp16", dtype == torch.float16)
    if attention_backend is not None:
        setattr(provider, "attention_backend", AttnBackend[attention_backend])
    is_mla_model = bool(getattr(provider, "multi_latent_attention", False))
    use_mla_latent_cache = cache_mla_latents
    if use_mla_latent_cache is None:
        use_mla_latent_cache = is_mla_model
    if cache_mla_latents is not None or is_mla_model or hasattr(provider, "cache_mla_latents"):
        setattr(provider, "cache_mla_latents", use_mla_latent_cache)
    if inference_moe_token_dispatcher_type is not None:
        if not hasattr(provider, "inference_moe_token_dispatcher_type"):
            raise ValueError(
                "--inference-moe-token-dispatcher-type was set, but the selected provider "
                "does not expose inference_moe_token_dispatcher_type."
            )
        setattr(provider, "inference_moe_token_dispatcher_type", inference_moe_token_dispatcher_type)


def _megatron_checkpoint_overrides(
    provider: object,
    *,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    sequence_parallel: bool,
    dtype: torch.dtype,
    attention_backend: str | None,
    inference_moe_token_dispatcher_type: str | None,
) -> dict[str, object]:
    overrides: dict[str, object] = {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "expert_tensor_parallel_size": etp,
        "sequence_parallel": sequence_parallel,
        "params_dtype": dtype,
        "pipeline_dtype": dtype,
        "bf16": dtype == torch.bfloat16,
        "fp16": dtype == torch.float16,
    }
    if attention_backend is not None:
        overrides["attention_backend"] = AttnBackend[attention_backend]
    if hasattr(provider, "cache_mla_latents"):
        overrides["cache_mla_latents"] = bool(getattr(provider, "cache_mla_latents"))
    if inference_moe_token_dispatcher_type is not None:
        overrides["inference_moe_token_dispatcher_type"] = inference_moe_token_dispatcher_type
    return overrides


def _prepare_model_list(model_list: list[torch.nn.Module]) -> torch.nn.Module:
    if len(model_list) != 1:
        raise ValueError(
            "MCore high-level inference supports one local model stage; virtual pipeline parallelism is not supported."
        )
    model = model_list[0].cuda()
    model.eval()
    disable_mtp_for_inference(model)
    if hasattr(model, "config"):
        model.config.grad_scale_func = None
    return model


def load_bridge_model(
    *,
    hf_model_path: str,
    megatron_model_path: str | None,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    sequence_parallel: bool,
    dtype: torch.dtype,
    seed: int,
    trust_remote_code: bool | None,
    attention_backend: str | None = None,
    cache_mla_latents: bool | None = None,
    inference_moe_token_dispatcher_type: str | None = None,
) -> torch.nn.Module:
    """Build (and optionally load weights for) a single-stage Megatron model via AutoBridge.

    When ``megatron_model_path`` is provided, the Megatron Bridge checkpoint is loaded with
    the given parallelism overrides; otherwise HF weights are converted in-process.
    """
    safe_trust_remote_code = is_safe_repo(hf_path=hf_model_path, trust_remote_code=trust_remote_code)

    parallelism = dict(
        tp=tp,
        pp=pp,
        ep=ep,
        etp=etp,
        sequence_parallel=sequence_parallel,
        dtype=dtype,
        attention_backend=attention_backend,
        inference_moe_token_dispatcher_type=inference_moe_token_dispatcher_type,
    )

    if megatron_model_path:
        config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=safe_trust_remote_code)
        bridge = AutoBridge.from_hf_config(config)
        provider = bridge.to_megatron_provider(load_weights=False)
        _apply_provider_parallelism(provider, cache_mla_latents=cache_mla_latents, **parallelism)
        provider.finalize()
        provider.initialize_model_parallel(seed=seed)
        mp_overrides = _megatron_checkpoint_overrides(provider, **parallelism)
        model_list = bridge.load_megatron_model(
            megatron_model_path,
            mp_overrides=mp_overrides,
            wrap_with_ddp=False,
        )
    else:
        bridge = AutoBridge.from_hf_pretrained(
            hf_model_path,
            torch_dtype=dtype,
            trust_remote_code=safe_trust_remote_code,
        )
        provider = bridge.to_megatron_provider(load_weights=True)
        _apply_provider_parallelism(provider, cache_mla_latents=cache_mla_latents, **parallelism)
        provider.finalize()
        provider.initialize_model_parallel(seed=seed)
        model_list = provider.provide_distributed_model(wrap_with_ddp=False)

    return _prepare_model_list(model_list)


def validate_sequence_length(
    *,
    longest_prompt_tokens: int,
    num_new_tokens: int,
    max_seq_length: int,
) -> None:
    """Raise if the longest prompt plus generation exceeds the configured sequence length."""
    required = longest_prompt_tokens + num_new_tokens
    if required > max_seq_length:
        raise ValueError(
            f"Longest prompt plus generation needs {required} tokens, but --max_seq_length is {max_seq_length}."
        )


def _effective_max_sequence_length(model: torch.nn.Module, max_sequence_length: int) -> int:
    """Clamp the requested sequence length to the model's table for learned-absolute pos-emb.

    For ``learned_absolute`` position embeddings the context's ``max_sequence_length`` must not
    exceed the model's, otherwise the position ids index past the embedding table. RoPE/other
    embeddings are unaffected (pass-through).
    """
    try:
        position_embedding_type = get_attr_wrapped_model(model, "position_embedding_type")
        model_max_seq_len = get_attr_wrapped_model(model, "max_sequence_length")
    except Exception:  # noqa: BLE001 - model may not expose these attrs; keep pass-through.
        return max_sequence_length
    if position_embedding_type == "learned_absolute" and model_max_seq_len:
        return min(model_max_seq_len, max_sequence_length)
    return max_sequence_length


def build_inference_config(
    *,
    model: torch.nn.Module,
    max_sequence_length: int,
    max_batch_size: int | None,
    num_prompts: int | None,
    tp: int,
    block_size_tokens: int,
    kv_cache_buffer_size_gb: float,
    max_tokens: int | None,
    return_log_probs: bool,
    enable_chunked_prefill: bool,
) -> InferenceConfig:
    """Construct the runtime ``megatron.core.inference.config.InferenceConfig``.

    Centralizes the KV-cache / batching translation so the offline scripts and the server share
    one source of truth. Pure: never mutates caller state.

    ``max_requests`` resolves to ``max_batch_size`` if set, else ``num_prompts``. When both are
    ``None`` (e.g. a server that should auto-size to the KV-cache memory buffer), ``max_requests``
    is left as ``None`` for the engine to size.
    """
    effective_block_size = block_size_tokens
    if getattr(getattr(model, "config", None), "cache_mla_latents", False) and block_size_tokens != 64:
        print_rank_0(
            f"Using block size 64 instead of {block_size_tokens} because MCore dynamic inference "
            "requires 64-token blocks when caching MLA latents."
        )
        effective_block_size = 64

    max_requests = max_batch_size or num_prompts
    if max_requests is not None and max_requests % tp != 0:
        rounded = ((max_requests + tp - 1) // tp) * tp
        if max_batch_size is not None:
            raise ValueError(
                f"--max_batch_size must be divisible by --tp ({tp}); got --max_batch_size {max_batch_size}."
            )
        print_rank_0(
            f"Rounding max batch size from {max_requests} to {rounded} "
            f"so it is divisible by tensor parallel size {tp}."
        )
        max_requests = rounded

    return InferenceConfig(
        block_size_tokens=effective_block_size,
        buffer_size_gb=kv_cache_buffer_size_gb,
        max_requests=max_requests,
        max_tokens=max_tokens,
        max_sequence_length=_effective_max_sequence_length(model, max_sequence_length),
        mamba_inference_state_config=MambaInferenceStateConfig.from_model(model),
        pg_collection=getattr(model, "pg_collection", None),
        materialize_only_last_token_logits=not return_log_probs,
        enable_chunked_prefill=enable_chunked_prefill,
    )


def build_sampling_params(
    *,
    temperature: float,
    top_k: int | None,
    top_p: float,
    return_log_probs: bool,
    skip_prompt_log_probs: bool,
    num_tokens_to_generate: int,
    termination_id: int | None,
    top_n_logprobs: int,
    stop_words: list[str] | None,
) -> SamplingParams:
    """Build MCore ``SamplingParams`` from CLI-derived values."""
    if top_k is None:
        top_k = 0 if top_p > 0.0 else 1
    if top_k > 0 and top_p > 0.0:
        raise ValueError("--top_k and --top_p cannot both be positive.")
    return SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_log_probs=return_log_probs,
        skip_prompt_log_probs=skip_prompt_log_probs,
        num_tokens_to_generate=num_tokens_to_generate,
        termination_id=termination_id,
        top_n_logprobs=top_n_logprobs,
        stop_words=stop_words,
    )


# ---------------------------------------------------------------------------
# Argument groups shared by the standalone scripts.
# ---------------------------------------------------------------------------


def add_model_loading_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add model-loading args (HF/Megatron source, trust-remote-code, dtype)."""
    group = parser.add_argument_group("Model loading")
    group.add_argument(
        "--hf_model_path",
        "--hf-model-path",
        dest="hf_model_path",
        default=None,
        help="Hugging Face model id/path for config and tokenizer. Required unless checkpoint metadata records it.",
    )
    group.add_argument(
        "--megatron_model_path",
        "--megatron-model-path",
        dest="megatron_model_path",
        default=None,
        help="Optional Megatron Bridge checkpoint path. If omitted, load and convert HF weights in-process.",
    )
    group.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=None,
        help="Allow custom Hugging Face model/tokenizer code for trusted repositories.",
    )
    group.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Model parameter dtype for in-process HF conversion and provider setup.",
    )
    return parser


def add_parallelism_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add parallelism + RNG-seed + MLA-cache args."""
    group = parser.add_argument_group("Parallelism")
    group.add_argument("--tp", type=int, default=1, help="Tensor model parallel size.")
    group.add_argument("--pp", type=int, default=1, help="Pipeline model parallel size.")
    group.add_argument("--ep", type=int, default=1, help="Expert model parallel size.")
    group.add_argument("--etp", type=int, default=1, help="Expert tensor parallel size.")
    group.add_argument("--sequence-parallel", action="store_true", help="Enable sequence parallelism.")
    group.add_argument("--seed", type=int, default=0, help="Model-parallel RNG seed.")
    group.add_argument(
        "--cache-mla-latents",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache MLA latents for dynamic inference. Defaults on for MLA models.",
    )
    return parser


def add_prompt_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add prompt-source args."""
    group = parser.add_argument_group("Prompts")
    group.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt text. May be provided multiple times. Defaults to a short prompt if no prompt file is set.",
    )
    group.add_argument(
        "--prompt_file",
        "--prompt-file",
        dest="prompt_file",
        default=None,
        help="Line-oriented prompt file. JSONL lines use the `text`/`prompt`/`input` field; other "
        "lines are raw prompts.",
    )
    group.add_argument(
        "--prompt-file-num-truncate",
        type=int,
        default=None,
        help="Read at most this many prompts from --prompt_file.",
    )
    return parser


def add_sampling_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add sampling args. ``--max_new_tokens`` is the canonical generation-length flag."""
    group = parser.add_argument_group("Sampling")
    group.add_argument("--max_new_tokens", type=int, default=30, help="Maximum generated tokens per prompt.")
    group.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    group.add_argument("--top_p", type=float, default=0.0, help="Top-p sampling.")
    group.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling. Defaults to 1 unless --top_p is set.",
    )
    group.add_argument("--return-log-probs", action="store_true", help="Return token log probabilities.")
    group.add_argument("--skip-prompt-log-probs", action="store_true", help="Skip prompt log probabilities.")
    group.add_argument("--top-n-logprobs", type=int, default=0, help="Return top-n logprobs.")
    group.add_argument("--termination-id", type=int, default=None, help="Override tokenizer EOD id.")
    group.add_argument(
        "--stop-words",
        nargs="+",
        default=None,
        help="Stop words that terminate generation when produced.",
    )
    return parser


def add_engine_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add KV-cache / batching / attention-backend engine args."""
    group = parser.add_argument_group("Inference engine")
    group.add_argument(
        "--attention-backend",
        choices=("auto", "flash", "fused", "unfused", "local"),
        default=None,
        help="Override the provider attention backend before constructing the Megatron model.",
    )
    group.add_argument("--max_seq_length", type=int, default=4096, help="Prompt plus generation length limit.")
    group.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximum active requests. Defaults to the number of prompts.",
    )
    group.add_argument("--max_tokens", type=int, default=None, help="Maximum active tokens.")
    group.add_argument("--block_size_tokens", type=int, default=256, help="KV-cache block size in tokens.")
    group.add_argument(
        "--kv_cache_buffer_size_gb",
        type=float,
        default=20.0,
        help="GPU buffer size reserved for KV cache.",
    )
    group.add_argument("--enable-chunked-prefill", action="store_true", help="Enable chunked prefill.")
    group.add_argument(
        "--inference-moe-token-dispatcher-type",
        choices=("nccl", "nvls"),
        default=None,
        help="Override the MCore MoE token dispatcher used during inference.",
    )
    return parser


def add_distributed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add distributed-init timeout arg."""
    group = parser.add_argument_group("Distributed")
    group.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=60,
        help="Process-group timeout in minutes for slow multi-node model setup.",
    )
    return parser
