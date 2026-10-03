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

"""Run deterministic BAGEL image-understanding inference from an MCore checkpoint."""

from __future__ import annotations

import argparse

import torch
from PIL import Image

from megatron.bridge import AutoBridge
from megatron.bridge.models.bagel.data.batch import _block_mask
from megatron.bridge.models.bagel.dependencies import (
    configure_official_bagel_repo,
    import_official_bagel_module,
)
from megatron.bridge.utils.common_utils import maybe_initialize_distributed, print_rank_0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bagel-repo", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--hf-revision")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def _load_model(args: argparse.Namespace) -> torch.nn.Module:
    hf_kwargs = {"revision": args.hf_revision} if args.hf_revision else {}
    bridge = AutoBridge.from_hf_pretrained(args.hf_model, **hf_kwargs)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.pipeline_dtype = torch.bfloat16
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    model = bridge.load_megatron_model(
        args.checkpoint,
        mp_overrides={
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "pipeline_dtype": torch.bfloat16,
            "params_dtype": torch.bfloat16,
            "bf16": True,
            "fp16": False,
        },
        wrap_with_ddp=False,
    )[0]
    model = model.module if hasattr(model, "module") else model
    return model.cuda().eval()


def _prepare_image(image_path: str, data_utils, transforms) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(image_path) as source:
        image = data_utils.pil_img2rgb(source)
    # Official understanding inference applies this resize before the ViT transform.
    image = transforms.ImageTransform(1024, 512, 16).resize_transform(image)
    image_tensor = transforms.ImageTransform(980, 224, 14)(image)
    positions = data_utils.get_flattened_position_ids_extrapolate(image_tensor.shape[1], image_tensor.shape[2], 14, 70)
    return data_utils.patchify(image_tensor, 14), positions


def _next_token(
    model: torch.nn.Module,
    *,
    vision_embeddings: torch.Tensor,
    vit_token_count: int,
    prompt_ids: list[int],
    generated: list[int],
    special_tokens: dict[str, int],
    data_utils,
) -> int:
    from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams

    device = torch.device("cuda")
    text_ids = [
        special_tokens["start_of_image"],
        special_tokens["end_of_image"],
        *prompt_ids,
        *generated,
    ]
    image_length = vit_token_count + 2
    sequence_length = image_length + len(prompt_ids) + len(generated)
    text_indexes = torch.tensor([0, vit_token_count + 1, *range(image_length, sequence_length)], device=device)
    vit_indexes = torch.arange(1, vit_token_count + 1, device=device)
    und_indexes = torch.cat((text_indexes, vit_indexes))
    empty = torch.empty(0, dtype=torch.long, device=device)
    packed = MoTPackedSeqParams(
        packed_text_indexes=text_indexes,
        packed_vit_token_indexes=vit_indexes,
        packed_vae_token_indexes=empty,
        packed_und_token_indexes=und_indexes,
        packed_gen_token_indexes=empty,
        local_und_token_indexes=und_indexes,
        local_gen_token_indexes=empty,
        padded_und_seqlen=sequence_length,
        padded_gen_seqlen=0,
    )
    full_positions = torch.cat(
        (
            torch.zeros(image_length, dtype=torch.long, device=device),
            torch.arange(1, 1 + len(prompt_ids) + len(generated), device=device),
        )
    )
    decoder_input = model.align_embeddings_by_token_positions(
        input_ids=torch.tensor(text_ids, device=device).unsqueeze(0),
        vision_embeddings=vision_embeddings,
        visual_latents=None,
        sequence_length=sequence_length,
        packed_seq_params=packed,
    )["decoder_input"]
    labels = torch.zeros(sequence_length, dtype=torch.long, device=device)
    loss_mask = torch.zeros(sequence_length, device=device)
    loss_mask[len(text_ids) - 1] = 1
    attention_mask = _block_mask(
        [
            data_utils.prepare_attention_mask_per_sample(
                [image_length, sequence_length - image_length], ["full", "causal"]
            )
        ],
        model.language_model.num_heads,
    )
    output = model.language_model(
        decoder_input=decoder_input,
        attention_mask=attention_mask,
        labels=labels,
        loss_mask=loss_mask,
        packed_position_ids=full_positions[und_indexes],
        sample_lens=[sequence_length],
        packed_seq_params=packed,
    )
    logits, _ = model.language_model.output_layer(output["last_hidden_state"][len(text_ids) - 1].unsqueeze(0))
    return int(logits.argmax(dim=-1).item())


def main() -> None:
    """Load BAGEL and generate one deterministic image-understanding response."""
    args = parse_args()
    configure_official_bagel_repo(args.bagel_repo)
    data_utils = import_official_bagel_module("data.data_utils")
    transforms = import_official_bagel_module("data.transforms")
    qwen2 = import_official_bagel_module("modeling.qwen2")
    maybe_initialize_distributed()
    model = _load_model(args)

    hf_kwargs = {"revision": args.hf_revision} if args.hf_revision else {}
    tokenizer = qwen2.Qwen2Tokenizer.from_pretrained(args.hf_model, **hf_kwargs)
    tokenizer, special_tokens, _ = data_utils.add_special_tokens(tokenizer)
    vit_tokens, vit_positions = _prepare_image(args.image, data_utils, transforms)
    device = torch.device("cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        vision_embeddings = model.modality_submodules["images"](
            {
                "vision_encoder": {
                    "packed_vit_tokens": vit_tokens.to(device),
                    "packed_vit_position_ids": vit_positions.to(device),
                    "vit_token_seqlens": torch.tensor([len(vit_tokens)], device=device),
                }
            }
        )
        prompt_ids = [
            special_tokens["bos_token_id"],
            *tokenizer.encode(args.prompt),
            special_tokens["eos_token_id"],
        ]
        generated = [special_tokens["bos_token_id"]]
        for _ in range(args.max_new_tokens):
            token = _next_token(
                model,
                vision_embeddings=vision_embeddings,
                vit_token_count=len(vit_tokens),
                prompt_ids=prompt_ids,
                generated=generated,
                special_tokens=special_tokens,
                data_utils=data_utils,
            )
            if token == special_tokens["eos_token_id"]:
                break
            generated.append(token)

    print_rank_0(f"Prompt: {args.prompt}")
    print_rank_0(f"Completion: {tokenizer.decode(generated[1:])}")


if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
