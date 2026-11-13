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
"""Nemotron-H selected-expert replay: substitute IDs, never Megatron routing weights.

Reference rows are full, unpacked B=1 sequences before EP dispatch. This is
an intervention diagnostic, not evidence of natural HF/Megatron parity.
"""

import hashlib
from types import MethodType

import torch


def tensor_hash(value: torch.Tensor) -> dict:
    """Identify a tensor by shape, dtype, and exact contiguous bytes."""
    data = value.detach().cpu().contiguous()
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "sha256": hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def fixture_identity(sample: dict) -> dict:
    """Bind route rows to identical token order, media, labels, and supervision."""
    result = {"kind": sample["kind"], "prompt_length": sample["prompt_length"]}
    for key in (
        "input_ids",
        "labels",
        "shifted_labels",
        "loss_mask",
        "pixels",
        "imgs_sizes",
        "num_frames",
        "pixel_values_videos",
        "hf_pixel_values",
    ):
        value = sample.get(key)
        result[key] = (
            None
            if value is None
            else ([tensor_hash(item) for item in value] if isinstance(value, (list, tuple)) else tensor_hash(value))
        )
    if sample["input_ids"].ndim != 2 or sample["input_ids"].shape[0] != 1:
        raise AssertionError("Replay requires unpacked singleton-batch sequences")
    if sample["input_ids"].dtype != torch.int64 or sample["input_ids"].shape[1] == 0:
        raise AssertionError("Replay requires nonempty int64 token IDs")
    result["sequence_length"] = sample["input_ids"].shape[1]
    return result


def validate_indices(indices: torch.Tensor, *, rows: int, experts: int, topk: int) -> None:
    """Reject partial, duplicated, out-of-range, or misaligned route inventories."""
    if indices.dtype != torch.int64 or tuple(indices.shape) != (rows, topk):
        raise AssertionError("Route index dtype/shape does not match this full sequence")
    if not 0 < topk <= experts or bool((indices < 0).any()) or bool((indices >= experts).any()):
        raise AssertionError("Out-of-range expert ID")
    ordered = indices.sort(-1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise AssertionError("Duplicate expert in a token route")


def route_set_changes(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Ignore expert enumeration order when comparing selection sets."""
    if first.shape != second.shape:
        raise AssertionError("Unaligned route sets")
    return (first.sort(-1).values != second.sort(-1).values).any(-1)


def replay_hf_selection(
    logits: torch.Tensor,
    *,
    native_indices: torch.Tensor,
    forced_indices: torch.Tensor,
    norm_topk_prob: bool,
    scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select reference experts and calculate their weights using current HF logits.

    Preserve native HF enumeration for retained experts, then append new experts
    in ascending-ID order. A self-replay therefore does not perturb summation
    order. Megatron exposes a routing set, not an ordered top-k list.
    """
    if logits.dtype != torch.float32 or logits.ndim != 2 or not torch.isfinite(logits).all():
        raise AssertionError("Expected finite FP32 full-sequence HF gate logits")
    rows, experts = logits.shape
    topk = native_indices.shape[-1]
    validate_indices(native_indices, rows=rows, experts=experts, topk=topk)
    forced = forced_indices.to(device=logits.device)
    validate_indices(forced, rows=rows, experts=experts, topk=topk)
    priority = torch.arange(experts, device=logits.device).expand(rows, -1).clone() + topk
    priority.scatter_(1, native_indices, torch.arange(topk, device=logits.device).expand(rows, -1))
    order = priority.gather(1, forced).argsort(-1)
    indices = forced.gather(1, order)
    weights = logits.sigmoid().gather(1, indices)
    # These operations match the native NemotronHMoE implementation, including
    # its epsilon, normalization order, and post-normalization scaling.
    if norm_topk_prob:
        weights /= weights.sum(-1, keepdim=True) + 1e-20
    weights = weights * scaling_factor
    return indices, weights


def validate_reference(reference: dict, *, case: str, sample: dict, layers: set[int], experts: int, topk: int) -> None:
    """Fail closed on source/fixture/layer coverage before enabling intervention."""
    if reference["schema"] != 1 or reference["case"] != case:
        raise AssertionError("Wrong route artifact schema/case")
    if reference["fixture"] != fixture_identity(sample):
        raise AssertionError("Reference routes belong to different tokens/media/labels")
    if reference["experts"] != experts or reference["topk"] != topk:
        raise AssertionError("Reference router dimensions differ")
    if set(reference["layers"]) != layers:
        raise AssertionError("Reference must cover exactly every language MoE layer")
    for item in reference["layers"].values():
        validate_indices(item["indices"], rows=reference["fixture"]["sequence_length"], experts=experts, topk=topk)


class BridgeRouteRecorder:
    """Observe full-sequence router outputs before dispatch without changing values."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.modules = {
            index: layer.mlp.router
            for index, layer in enumerate(getattr(model, "language_model", model).decoder.layers)
            if hasattr(getattr(layer, "mlp", None), "router")
        }
        if not self.modules:
            raise AssertionError("No native language MoE routers")
        dimensions = {(router.config.num_moe_experts, router.topk) for router in self.modules.values()}
        if len(dimensions) != 1:
            raise AssertionError("Mixed router dimensions are out of scope")
        self.experts, self.topk = dimensions.pop()
        self.active = False
        self.record = {}
        self.handles = []
        for index, router in self.modules.items():

            def post(module, inputs, output, layer=index):
                if not self.active:
                    return
                if layer in self.record["layers"]:
                    raise AssertionError("A router ran more than once in an uncached forward")
                probabilities, selected = output
                rows = self.record["fixture"]["sequence_length"]
                if selected.dtype != torch.bool or tuple(selected.shape) != (rows, self.experts):
                    raise AssertionError("Bridge router is not full-sequence/global-expert aligned")
                if probabilities.shape != selected.shape or not torch.isfinite(probabilities).all():
                    raise AssertionError("Invalid native routing probabilities")
                if not torch.all(selected.sum(-1) == self.topk):
                    raise AssertionError("Dropping/padding changed token expert counts")
                indices = selected.nonzero(as_tuple=True)[1].reshape(rows, self.topk)
                validate_indices(indices, rows=rows, experts=self.experts, topk=self.topk)
                self.record["layers"][layer] = {
                    "indices": indices.cpu().clone(),
                    "weights_for_audit_only": probabilities.gather(1, indices).detach().cpu().clone(),
                }

            self.handles.append(router.register_forward_hook(post))

    def begin(self, case: str, sample: dict) -> None:
        """Start one complete fixture; never reuse choices across sequences."""
        if self.active:
            raise AssertionError("Previous recording not completed")
        self.record = {
            "schema": 1,
            "source": "megatron_natural",
            "case": case,
            "fixture": fixture_identity(sample),
            "experts": self.experts,
            "topk": self.topk,
            "layers": {},
        }
        self.active = True

    def finish(self) -> dict:
        """Require exact layer coverage and return the immutable reference inventory."""
        self.active = False
        if set(self.record["layers"]) != set(self.modules):
            raise AssertionError("Missing language MoE router captures")
        return self.record

    def close(self) -> None:
        """Remove every observer installed by this diagnostic."""
        for handle in self.handles:
            handle.remove()


class HFRouteReplay:
    """Temporarily wrap only native expert selection; restore original methods fully."""

    def __init__(self, model: torch.nn.Module) -> None:
        language = getattr(model, "language_model", model)
        # Remote-code and Transformers-native Nemotron-H use different names
        # for the same language backbone. Neither includes the separate MTP stack.
        backbone = getattr(language, "backbone", None)
        if backbone is None:
            backbone = getattr(language, "model", None)
        if backbone is None:
            raise ValueError("Expected a Nemotron-H backbone or model containing layers")
        self.modules = {index: layer.mixer for index, layer in enumerate(backbone.layers) if layer.block_type == "moe"}
        if not self.modules:
            raise AssertionError("No native HF language MoE modules")
        dimensions = {(m.n_routed_experts, m.top_k) for m in self.modules.values()}
        if len(dimensions) != 1:
            raise AssertionError("Mixed router dimensions are out of scope")
        self.experts, self.topk = dimensions.pop()
        self.active, self.originals = False, []
        self.reference = None
        self.record = {}
        for module in self.modules.values():
            if not callable(getattr(module, "route_tokens_to_experts", None)):
                raise ValueError("Expected Nemotron-H route_tokens_to_experts")
            if not hasattr(module, "norm_topk_prob") or not hasattr(module, "routed_scaling_factor"):
                raise ValueError("Unsupported HF router weighting semantics")
        self.bias_hashes = {i: tensor_hash(m.gate.e_score_correction_bias) for i, m in self.modules.items()}

    def __enter__(self) -> "HFRouteReplay":
        for index, module in self.modules.items():
            original = module.route_tokens_to_experts
            existed = "route_tokens_to_experts" in module.__dict__
            previous = module.__dict__.get("route_tokens_to_experts")
            self.originals.append((module, existed, previous))

            def route(current, logits, layer=index, native=original):
                ids, weights = native(logits)
                if not self.active:
                    return ids, weights
                if layer in self.record["layers"]:
                    raise AssertionError("Repeated HF router in an uncached forward")
                validate_indices(
                    ids, rows=self.record["fixture"]["sequence_length"], experts=self.experts, topk=self.topk
                )
                changed = torch.zeros(len(ids), device=ids.device, dtype=torch.bool)
                if self.reference is not None:
                    expected = self.reference["layers"][layer]["indices"].to(ids.device)
                    changed = route_set_changes(ids, expected)
                    ids, weights = replay_hf_selection(
                        logits,
                        native_indices=ids,
                        forced_indices=expected,
                        norm_topk_prob=current.norm_topk_prob,
                        scaling_factor=current.routed_scaling_factor,
                    )
                    if bool(route_set_changes(ids, expected).any()):
                        raise AssertionError("Replay did not use the exact recorded expert set")
                self.record["layers"][layer] = {
                    "indices": ids.detach().cpu().clone(),
                    "weights_for_audit_only": weights.detach().cpu().clone(),
                    "overridden_token_rows": changed.nonzero().flatten().cpu().tolist(),
                }
                return ids, weights

            module.route_tokens_to_experts = MethodType(route, module)
        return self

    def begin(self, case: str, sample: dict, *, reference: dict | None) -> None:
        """Record natural HF choices or replay a validated reference choice inventory."""
        if self.active:
            raise AssertionError("Previous HF recording not completed")
        if reference is not None:
            validate_reference(
                reference, case=case, sample=sample, layers=set(self.modules), experts=self.experts, topk=self.topk
            )
        self.reference = reference
        self.record = {
            "schema": 1,
            "source": "hf_natural" if reference is None else "hf_ID_only_replay",
            "reference_source": None if reference is None else reference["source"],
            "case": case,
            "fixture": fixture_identity(sample),
            "experts": self.experts,
            "topk": self.topk,
            "layers": {},
        }
        self.active = True

    def finish(self) -> dict:
        """Require all routes, and prove that the checkpoint's router biases were untouched."""
        self.active = False
        if set(self.record["layers"]) != set(self.modules):
            raise AssertionError("Missing HF language MoE routes")
        if self.bias_hashes != {i: tensor_hash(m.gate.e_score_correction_bias) for i, m in self.modules.items()}:
            raise AssertionError("Replay changed expert-bias buffers")
        return self.record

    def __exit__(self, *exc: object) -> None:
        self.active = False
        for module, existed, previous in reversed(self.originals):
            if existed:
                module.route_tokens_to_experts = previous
            else:
                delattr(module, "route_tokens_to_experts")
        self.originals.clear()
