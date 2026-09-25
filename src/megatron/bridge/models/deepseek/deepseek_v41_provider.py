# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Model provider for Megatron-Core's native DeepSeek-V4.1 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


def _load_deepseek_v41_model_class():
    """Load the native model lazily so older MCore pins can still import Bridge."""
    try:
        from megatron.core.models.deepseek_v41.model import DeepSeekV41Model
    except ImportError as exc:  # pragma: no cover - exercised with unsupported MCore pins
        raise RuntimeError(
            "DeepSeek-V4.1 requires the native Megatron-Core implementation from "
            "Megatron-LM-dev commit 0d4c86d17 (origin/pull-request/7503) or a descendant."
        ) from exc
    return DeepSeekV41Model


@dataclass
class DeepSeekV41ModelProvider(MLAModelProvider):
    """Build ``DeepSeekV41Model`` while carrying its component configurations.

    The native model is a ``HybridModel`` subclass, but it owns extra Engram,
    vision, and DSpark modules and therefore cannot be constructed by the generic
    HybridModel provider.
    """

    engram_config: Any | None = None
    vision_config: Any | None = None
    dspark_config: Any | None = None
    # V4.1's text/image correction biases are loaded from the checkpoint and
    # must remain fixed during actor updates.
    moe_router_bias_update_rate: float = 0.0

    @property
    def hybrid_pattern(self) -> str:
        """Return the attention/expert physical-layer pattern."""
        return "VE" * (int(self.num_layers) // 2)

    def finalize(self) -> None:
        """Validate the topology supported by the native V4.1 wrapper."""
        if (self.tensor_model_parallel_size or 1) != 1 or (self.context_parallel_size or 1) != 1:
            raise NotImplementedError("Megatron-Core DeepSeekV41Model currently requires TP=CP=1")
        if (self.pipeline_model_parallel_size or 1) != 1:
            raise NotImplementedError(
                "Megatron-Core DeepSeekV41Model currently requires pipeline_model_parallel_size=1"
            )
        if self.virtual_pipeline_model_parallel_size not in (None, 1):
            raise NotImplementedError("DeepSeekV41Model does not support virtual pipeline parallelism")
        if self.recompute_granularity is not None:
            raise NotImplementedError("DeepSeekV41Model does not support stack activation recomputation")

        # The released ``mtp.*`` namespace is DSpark, not legacy autoregressive MTP.
        self.mtp_num_layers = 0
        self.share_embeddings_and_output_weights = False
        super().finalize()

    def _tokenizer_for_engram(self):
        if self.engram_config is None:
            return None
        cached = getattr(self, "_deepseek_v41_tokenizer", None)
        if cached is not None:
            return cached
        if not self.hf_model_id:
            raise ValueError("DeepSeek-V4.1 Engram construction requires hf_model_id or an explicit token map")

        from transformers import AutoTokenizer

        kwargs = {"trust_remote_code": True}
        if self.hf_model_revision:
            kwargs["revision"] = self.hf_model_revision
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id, **kwargs)
        self._deepseek_v41_tokenizer = tokenizer
        return tokenizer

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Instantiate the native V4.1 model."""
        if (self.pipeline_model_parallel_size or 1) != 1:
            raise NotImplementedError(
                "Megatron-Core DeepSeekV41Model currently requires pipeline_model_parallel_size=1"
            )
        if vp_stage not in (None, 0):
            raise NotImplementedError("DeepSeekV41Model does not support virtual pipeline stages")
        if self.vocab_size is None:
            raise ValueError("vocab_size must be configured before building DeepSeekV41Model")
        if self._pg_collection is None:
            raise RuntimeError("Process groups must be initialized before building DeepSeekV41Model")

        vocab_size = int(self.vocab_size)
        if self.should_pad_vocab:
            vocab_size = calculate_padded_vocab_size(
                vocab_size,
                self.make_vocab_size_divisible_by,
                self.tensor_model_parallel_size,
            )

        pre_process = pre_process if pre_process is not None else is_pp_first_stage(self._pg_collection.pp)
        post_process = post_process if post_process is not None else is_pp_last_stage(self._pg_collection.pp)
        token_map = getattr(self, "engram_token_map", None)
        tokenizer = None if token_map is not None else self._tokenizer_for_engram()

        # The native implementation is intentionally guarded as experimental in
        # Megatron-Core.  Its own smoke harness enables the same switch before
        # constructing the model; Bridge owns that setup for downstream callers.
        from megatron.core import config as core_config

        core_config.ENABLE_EXPERIMENTAL = True
        model_class = _load_deepseek_v41_model_class()

        pg_collection = self._pg_collection
        self._pg_collection = None
        try:
            return model_class(
                config=self,
                vocab_size=vocab_size,
                max_sequence_length=self.seq_length,
                pg_collection=pg_collection,
                token_map=token_map,
                tokenizer=tokenizer,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                logit_dtype=self.logit_dtype,
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=False,
                scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
            )
        finally:
            self._pg_collection = pg_collection


__all__ = ["DeepSeekV41ModelProvider"]
