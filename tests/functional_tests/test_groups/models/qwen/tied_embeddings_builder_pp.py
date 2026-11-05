# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Distributed functional check for tied embeddings on the builder path with pipeline parallelism.

Builds the model through ``get_model_config()`` / ``get_model()``, as builder-backed conversion
and training initialization do, then checks that the last pipeline stage's output layer holds the
input embeddings and that the export reproduces the Hugging Face weights without a separate
``lm_head.weight``.
"""

import argparse

import torch
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
from megatron.core.utils import unwrap_model

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import _get_pp_group
from megatron.bridge.models.decorators import torchrun_main


@torchrun_main
def main(hf_model_id: str, pp: int) -> None:
    bridge = AutoBridge.from_hf_pretrained(hf_model_id, torch_dtype=torch.bfloat16)
    if not bridge._model_bridge.USE_MODEL_CONFIG_FOR_CONVERSION:
        raise AssertionError(f"{type(bridge._model_bridge).__name__} does not convert through the builder path")

    model_config = bridge.get_model_config()
    if not model_config.share_embeddings_and_output_weights:
        raise AssertionError("This check requires a checkpoint with tied embeddings")
    model_config.tensor_model_parallel_size = 1
    model_config.pipeline_model_parallel_size = pp
    model_config.pipeline_dtype = torch.bfloat16
    model_config.params_dtype = torch.bfloat16
    megatron_model = bridge.get_model(model_config, wrap_with_ddp=False, mixed_precision_wrapper=None)

    errors = []
    hf_state = bridge.hf_pretrained.state
    hf_embeddings = hf_state["model.embed_tokens.weight"]
    if is_pp_last_stage(_get_pp_group(megatron_model)):
        output_weight = unwrap_model(megatron_model)[-1].output_layer.weight
        output_weight = output_weight[: hf_embeddings.shape[0]].detach().cpu()
        if not torch.equal(output_weight, hf_embeddings.to(output_weight.dtype)):
            errors.append("last-stage output_layer.weight does not match the Hugging Face input embeddings")

    exported = {}
    for name, weight in bridge.export_hf_weights(megatron_model, cpu=True, show_progress=False):
        exported[name] = weight
    if torch.distributed.get_rank() == 0:
        expected_names = set(hf_state.keys())
        if set(exported) != expected_names:
            errors.append(
                f"exported names differ: missing={sorted(expected_names - set(exported))}, "
                f"extra={sorted(set(exported) - expected_names)}"
            )
        errors.extend(
            f"exported {name} differs from the Hugging Face weight"
            for name, weight in exported.items()
            if name in expected_names and not torch.equal(weight, hf_state[name].to(weight.dtype))
        )

    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, errors)
    all_errors = [error for rank_errors in gathered for error in rank_errors]
    if all_errors:
        raise AssertionError("Tied-embedding builder check failed: " + "; ".join(all_errors))
    if torch.distributed.get_rank() == 0:
        print(f"Tied-embedding builder verification passed: PP={pp}, exported={len(exported)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-id", required=True)
    parser.add_argument("--pp", type=int, default=2)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(args.hf_model_id, args.pp)
