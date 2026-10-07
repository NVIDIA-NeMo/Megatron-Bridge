# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Record Megatron expert choices and replay them in HF (Nemotron-H only).

See router-replay.md. This is an intervention diagnostic, never a replacement
for natural forward parity or exact weight verification.
"""

import argparse
import inspect
import json
import logging
import os
from pathlib import Path

import torch
from examples.conversion.compare_hf_and_megatron.router_replay_utils import (
    BridgeRouteRecorder,
    HFRouteReplay,
    fixture_identity,
    route_set_changes,
    tensor_hash,
    validate_reference,
)


LOG = logging.getLogger(__name__)


def select_logits(output: torch.Tensor, *, sequence_length: int, positions: list[int], vocab: int) -> torch.Tensor:
    """Copy selected full-vocabulary rows, rejecting ambiguous sequence layouts."""
    if output.ndim != 3 or tuple(output.shape[:2]) != (1, sequence_length) or output.shape[-1] < vocab:
        raise ValueError(f"Expected [1, {sequence_length}, >= {vocab}] logits, got {output.shape}")
    if not positions or len(set(positions)) != len(positions) or any(p < 0 or p >= sequence_length for p in positions):
        raise ValueError("Positions must be unique, nonnegative, and within the full sequence")
    result = output[0, positions, :vocab].detach().cpu().clone()
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite logits")
    return result


def compare_logits(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    """Report raw/centered cosine and probability differences without a pass gate."""
    if reference.shape != actual.shape or reference.ndim != 2 or not reference.numel():
        raise ValueError("Unaligned or empty logit rows")
    a, b = reference.float(), actual.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Non-finite comparison")
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    centered = torch.nn.functional.cosine_similarity(
        a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True), dim=-1
    )
    pa, pb = a.double().softmax(-1), b.double().softmax(-1)
    tv = (pa - pb).abs().sum(-1) / 2
    return {
        "rows": len(a),
        "raw_cosine_min": cosine.min().item(),
        "raw_cosine_mean": cosine.mean().item(),
        "raw_cosine_per_row": cosine.tolist(),
        "centered_cosine_min": centered.min().item(),
        "top1_matches": int((a.argmax(-1) == b.argmax(-1)).sum()),
        "mean_absolute_difference": (a - b).abs().mean().item(),
        "max_absolute_difference": (a - b).abs().max().item(),
        "probability_tv_mean": tv.mean().item(),
        "probability_tv_max": tv.max().item(),
    }


def _require_equal(first: torch.Tensor, second: torch.Tensor, name: str) -> None:
    if first.dtype != second.dtype or first.shape != second.shape or not torch.equal(first, second):
        raise RuntimeError(f"{name} changed logits; this diagnostic is invalid")


def _config(args: argparse.Namespace):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.hf_model_path, revision=args.hf_revision, trust_remote_code=args.trust_remote_code
    )
    text = getattr(config, "llm_config", getattr(config, "text_config", config))
    if text.model_type != "nemotron_h":
        raise ValueError("Router replay currently supports only Nemotron-H sigmoid MoE routing")
    return config, text


def _inputs(args: argparse.Namespace, config) -> dict:
    from examples.conversion.compare_hf_and_megatron import compare

    from megatron.bridge.models.nemotron_omni.inference_inputs import (
        is_nemotron_omni,
        load_nemotron_omni_video,
        prepare_nemotron_omni_inputs,
    )

    has_media = bool(args.image_path or args.video_path)
    if has_media and not is_nemotron_omni(config):
        raise ValueError("Media replay requires a Nemotron-H multimodal checkpoint")
    if args.disable_hf_video_pruning and not args.video_path:
        raise ValueError("--disable-hf-video-pruning requires video input")
    if (
        args.video_path
        and float(getattr(config, "video_pruning_rate", 0.0)) != 0
        and not args.disable_hf_video_pruning
    ):
        raise ValueError("Bridge video is unpruned; explicitly opt into --disable-hf-video-pruning")
    tokenizer, processor = compare._setup_tokenizer_and_processor(args, has_media)
    sample = {"kind": "text", "pixels": None, "imgs_sizes": None, "num_frames": None, "pixel_values_videos": None}
    if has_media:
        if processor is None:
            raise ValueError("A native processor is required for media")
        frames = metadata = images = None
        if args.video_path:
            frames, metadata = load_nemotron_omni_video(args.video_path, fps=args.video_fps)
            sample["kind"] = "video"
        else:
            images = [compare.load_image(args.image_path).convert("RGB")]
            sample["kind"] = "image"
        prepared = prepare_nemotron_omni_inputs(
            processor, prompt=args.prompt, images=images, video_frames=frames, video_metadata=metadata
        )
        sample.update(
            input_ids=prepared.hf["input_ids"],
            pixels=prepared.bridge["pixel_values"],
            imgs_sizes=prepared.bridge["imgs_sizes"],
            num_frames=prepared.bridge["num_frames"],
            pixel_values_videos=prepared.hf.get("pixel_values_videos"),
            hf_pixel_values=prepared.hf.get("pixel_values"),
        )
    else:
        # A literal prompt allows explicit chat templates and teacher-forced continuations.
        sample["input_ids"] = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    sample["prompt_length"] = sample["input_ids"].shape[1]
    return sample


def _positions(value: str, length: int) -> list[int]:
    if value == "all":
        return list(range(length))
    if value == "last":
        return [length - 1]
    return [int(p) for p in value.split(",")]


def validate_hf_media_interface(model: torch.nn.Module, *, kind: str) -> None:
    """Reject older composite APIs rather than silently omitting required media inputs."""
    if kind == "text":
        return
    key = "pixel_values_videos" if kind == "video" else "pixel_values"
    parameters = inspect.signature(model.forward).parameters
    if not all(hasattr(model, name) for name in ("vision_model", "vision_projector")) or key not in parameters:
        raise ValueError(
            "Media replay requires the native vision_model/vision_projector and pixel_values[_videos] API; legacy Omni composites are unsupported"
        )
    if "image_flags" in parameters and parameters["image_flags"].default is inspect.Parameter.empty:
        raise ValueError("Legacy Omni image_flags preprocessing is not supported by this diagnostic")


def load_recording(path: str) -> dict:
    """Load tensors without arbitrary pickle execution and validate their binding."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact["schema"] != 1 or artifact["controls"] != {"megatron_observer_bitwise_equal": True}:
        raise ValueError("Not a validated Megatron recording")
    routes = artifact["routes"]
    if routes["source"] != "megatron_natural":
        raise ValueError("Reference routes must come from Megatron recording, not HF replay")
    validate_reference(
        routes,
        case="input",
        sample=artifact["sample"],
        layers=set(routes["layers"]),
        experts=routes["experts"],
        topk=routes["topk"],
    )
    logits = artifact["logits"]
    if logits.shape != (len(artifact["positions"]), artifact["vocab_size"]) or not torch.isfinite(logits).all():
        raise ValueError("Invalid reference logit inventory")
    return artifact


def record(args: argparse.Namespace) -> None:
    """Observe an uncached, singleton-batch TP1/PP1/CP1 forward before EP dispatch."""
    import torch.distributed as dist
    from examples.conversion.compare_hf_and_megatron import compare
    from megatron.core.inference.utils import InferenceMode

    from megatron.bridge.training.nemotron_omni_step import _build_vision_packed_seq_params

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    if dist.get_world_size() != args.ep:
        raise ValueError("Launch exactly --ep ranks; TP, PP, CP, ETP, and data replicas are fixed to one")
    config, text_config = _config(args)
    sample = _inputs(args, config)
    identity = fixture_identity(sample)
    identities = [None] * dist.get_world_size()
    dist.all_gather_object(identities, identity)
    if any(item != identity for item in identities):
        raise ValueError("EP ranks received different token/media inputs")
    models, _ = compare._load_megatron_model(args)
    if len(models) != 1:
        raise ValueError("Exactly one model component is required")
    model = models[0]
    inner = model.module if hasattr(model, "module") else model
    expected = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "expert_model_parallel_size": args.ep,
        "sequence_parallel": False,
    }
    if any(getattr(inner.config, key) != value for key, value in expected.items()):
        raise ValueError("Loaded model has unsupported parallelism or sequence sharding")
    ids = sample["input_ids"].cuda()
    positions = _positions(args.positions, ids.shape[1])
    kwargs = {
        "input_ids": ids,
        "position_ids": torch.arange(ids.shape[1], device=ids.device).unsqueeze(0),
        "attention_mask": None,
        "runtime_gather_output": True,
    }
    if sample["pixels"] is not None:
        sizes = sample["imgs_sizes"].cuda()
        kwargs.update(
            images=sample["pixels"].cuda().bfloat16(),
            imgs_sizes=sizes,
            num_frames=sample["num_frames"].cuda(),
            vision_packed_seq_params=_build_vision_packed_seq_params(sizes),
        )

    def forward() -> torch.Tensor:
        with torch.inference_mode(), InferenceMode.active():
            output = model(**kwargs)
        if isinstance(output, tuple):
            output = output[0]
        return select_logits(output, sequence_length=ids.shape[1], positions=positions, vocab=text_config.vocab_size)

    baseline = forward()
    recorder = BridgeRouteRecorder(inner)
    try:
        recorder.begin("input", sample)
        observed = forward()
        routes = recorder.finish()
    finally:
        recorder.close()
    _require_equal(baseline, observed, "Megatron observer")
    # Every EP rank sees the same sequence before dispatch. Validate this instead
    # of silently assuming rank zero is representative.
    route_hash = {i: tensor_hash(item["indices"]) for i, item in routes["layers"].items()}
    hashes = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, route_hash)
    if any(item != route_hash for item in hashes):
        raise ValueError("EP ranks selected different experts for the same input")
    if dist.get_rank() == 0:
        artifact = {
            "schema": 1,
            "sample": sample,
            "routes": routes,
            "logits": observed,
            "positions": positions,
            "vocab_size": text_config.vocab_size,
            "hf_model_path": args.hf_model_path,
            "hf_revision": args.hf_revision,
            "megatron_model_path": args.megatron_model_path,
            "parallelism": expected,
            "disable_hf_video_pruning": args.disable_hf_video_pruning,
            "controls": {"megatron_observer_bitwise_equal": True},
        }
        with Path(args.output).open("xb") as stream:
            torch.save(artifact, stream)
        LOG.info("Saved routes and reference logits to %s", args.output)
    dist.barrier()


def replay(args: argparse.Namespace) -> None:
    """Compare natural HF, HF self-replay, forced Megatron choices, and restoration."""
    from examples.conversion.compare_hf_and_megatron.compare import _get_hf_forward_model
    from transformers import AutoModelForCausalLM

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("HF replay runs in one process; device_map shards it across visible GPUs")
    artifact = load_recording(args.routes)
    if args.hf_model_path != artifact["hf_model_path"] or args.hf_revision != artifact["hf_revision"]:
        raise ValueError("Replay must use the same HF path/revision as recording (use the exported HF path in both)")
    config, text_config = _config(args)
    if text_config.vocab_size != artifact["vocab_size"]:
        raise ValueError("HF vocabulary differs from the recording")
    if artifact["disable_hf_video_pruning"]:
        config.video_pruning_rate = 0.0
    # An explicit JSON map is useful for large VL models: native forward_video
    # can bypass Accelerate's __call__ hooks, so keep vision + projector together.
    device_map = args.hf_device_map
    if Path(device_map).is_file():
        device_map = json.loads(Path(device_map).read_text())
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path,
        config=config,
        revision=args.hf_revision,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        output_loading_info=True,
    )
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"HF checkpoint did not load exactly: {loading_info}")
    model.eval()
    sample = artifact["sample"]
    validate_hf_media_interface(model, kind=sample["kind"])
    pixels = sample.get("hf_pixel_values")
    videos = sample.get("pixel_values_videos")
    if videos is not None:
        devices = {p.device for module in (model.vision_model, model.vision_projector) for p in module.parameters()}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError(
                "Video requires vision_model and vision_projector on one CUDA device; supply a JSON device map"
            )
    forward_model = _get_hf_forward_model(model, pixels, videos)
    device = next(forward_model.parameters()).device
    ids = sample["input_ids"].to(device)
    kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool), "use_cache": False}
    if pixels is not None:
        vision_device = next(model.vision_model.parameters()).device
        kwargs["pixel_values"] = (
            [p.to(device=vision_device, dtype=torch.bfloat16) for p in pixels]
            if isinstance(pixels, (list, tuple))
            else pixels.to(device=vision_device, dtype=torch.bfloat16)
        )
    if videos is not None:
        # The maintained processor may return a list of equal-size video frames.
        videos = torch.stack(list(videos)) if isinstance(videos, (list, tuple)) else videos
        kwargs["pixel_values_videos"] = videos.to(
            device=next(model.vision_model.parameters()).device, dtype=torch.bfloat16
        )

    def forward() -> torch.Tensor:
        with torch.inference_mode():
            output = forward_model(**kwargs).logits
        return select_logits(
            output, sequence_length=ids.shape[1], positions=artifact["positions"], vocab=artifact["vocab_size"]
        )

    baseline = forward()
    with HFRouteReplay(model) as router:
        router.begin("input", sample, reference=None)
        natural = forward()
        natural_routes = router.finish()
        _require_equal(baseline, natural, "HF observer")
        router.begin("input", sample, reference=natural_routes)
        self_replay = forward()
        router.finish()
        _require_equal(natural, self_replay, "HF self-replay")
        router.begin("input", sample, reference=artifact["routes"])
        forced = forward()
        applied = router.finish()
    _require_equal(baseline, forward(), "HF restoration")
    counts = {}
    for layer, item in artifact["routes"]["layers"].items():
        counts[layer] = {
            "natural_different_token_rows": int(
                route_set_changes(item["indices"], natural_routes["layers"][layer]["indices"]).sum()
            ),
            "replay_overridden_token_rows": len(applied["layers"][layer]["overridden_token_rows"]),
        }
    report = {
        "status": "diagnostic_complete_controls_passed",
        "natural_parity_replaced": False,
        "controls": {
            **artifact["controls"],
            "hf_observer_bitwise_equal": True,
            "hf_self_replay_bitwise_equal": True,
            "hf_restoration_bitwise_equal": True,
        },
        "fixture": fixture_identity(sample),
        "positions": artifact["positions"],
        "hf_model_path": args.hf_model_path,
        "hf_revision": args.hf_revision,
        "parallelism": artifact["parallelism"],
        "route_counts_by_layer": counts,
        "natural": compare_logits(artifact["logits"], natural),
        "replayed": compare_logits(artifact["logits"], forced),
    }
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    LOG.info(
        "Natural minimum cosine %.8f; replayed %.8f (diagnostic only)",
        report["natural"]["raw_cosine_min"],
        report["replayed"]["raw_cosine_min"],
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the two-phase standalone diagnostic CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("record", "replay"):
        sub = subparsers.add_parser(phase)
        sub.add_argument("--hf-model-path", required=True)
        sub.add_argument("--hf-revision", help="Pin an immutable Hub commit; use the same value in both phases")
        sub.add_argument("--trust-remote-code", action="store_true")
        sub.add_argument("--output", required=True, help="New file; existing evidence is never overwritten")
        if phase == "record":
            sub.add_argument("--megatron-model-path", required=True)
            sub.add_argument("--prompt", required=True)
            media = sub.add_mutually_exclusive_group()
            media.add_argument("--image-path")
            media.add_argument("--video-path")
            sub.add_argument("--video-fps", type=float, default=2.0)
            sub.add_argument("--disable-hf-video-pruning", action="store_true")
            sub.add_argument("--ep", type=int, required=True)
            sub.add_argument("--positions", default="last", help="last, all, or comma-separated zero-based logit rows")
            sub.set_defaults(tp=1, pp=1, etp=1, enable_debug_hooks=False)
        else:
            sub.add_argument("--routes", required=True, help="Artifact produced by record")
            sub.add_argument(
                "--hf-device-map", default="auto", help="HF/Accelerate device map (auto, cuda:0, or a JSON file)"
            )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    try:
        (record if args.phase == "record" else replay)(args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
