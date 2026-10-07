# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU controls for full-sequence Nemotron-H expert-choice replay."""

from types import SimpleNamespace

import pytest
import torch
from examples.conversion.compare_hf_and_megatron.router_replay_utils import (
    BridgeRouteRecorder,
    HFRouteReplay,
    fixture_identity,
    replay_hf_selection,
    route_set_changes,
    validate_indices,
    validate_reference,
)


pytestmark = pytest.mark.unit


def sample():
    ids = torch.tensor([[1, 2, 3]])
    return {
        "input_ids": ids,
        "labels": ids,
        "shifted_labels": ids,
        "loss_mask": ids.bool(),
        "pixels": None,
        "imgs_sizes": None,
        "num_frames": None,
        "kind": "text",
        "prompt_length": 2,
    }


def native(logits, bias):
    values = logits.sigmoid()
    indices = (values + bias).topk(2, dim=-1, sorted=False).indices
    weights = values.gather(1, indices)
    weights /= weights.sum(-1, keepdim=True) + 1e-20
    return indices, weights * 5.0


def test_self_replay_exact_with_noncanonical_order_and_fp32_weights():
    torch.manual_seed(12)
    logits = torch.randn(7, 11)
    ids, weights = native(logits, torch.randn(11))
    replay_ids, replay_weights = replay_hf_selection(
        logits, native_indices=ids, forced_indices=ids.sort(-1).values, norm_topk_prob=True, scaling_factor=5.0
    )
    assert torch.equal(ids, replay_ids)
    assert torch.equal(weights, replay_weights)


def test_forced_experts_use_hf_weights_not_reference_weights_or_bias():
    logits = torch.tensor([[1.0, 2.0, -1.0, 0.0]])
    ids, _ = native(logits, torch.zeros(4))
    forced = torch.tensor([[2, 3]])
    actual, weights = replay_hf_selection(
        logits, native_indices=ids, forced_indices=forced, norm_topk_prob=True, scaling_factor=5.0
    )
    assert not route_set_changes(actual, forced).any()
    expected = logits.sigmoid().gather(1, actual)
    expected = expected / (expected.sum(-1, keepdim=True) + 1e-20) * 5.0
    assert torch.equal(weights, expected)


@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([[1, 1]]),
        torch.tensor([[0, 4]]),
        torch.tensor([[-1, 1]]),
        torch.tensor([[1]]),
        torch.tensor([[1.0, 2.0]]),
    ],
)
def test_invalid_route_inventory_fails_closed(indices):
    with pytest.raises(AssertionError):
        validate_indices(indices, rows=1, experts=4, topk=2)


class ToyMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k, self.n_routed_experts = 2, 4
        self.norm_topk_prob, self.routed_scaling_factor = True, 5.0
        self.gate = SimpleNamespace(e_score_correction_bias=torch.zeros(4))

    def route_tokens_to_experts(self, logits):
        return native(logits, self.gate.e_score_correction_bias)


def toy_model(module):
    return SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(block_type="moe", mixer=module)]))
    )


def test_context_records_replays_and_restores_even_after_error():
    module = ToyMoE()
    logits = torch.tensor([[1.0, 0.0, 2.0, -1.0]]).repeat(3, 1)
    expected = module.route_tokens_to_experts(logits)
    with pytest.raises(RuntimeError, match="injected"):
        with HFRouteReplay(toy_model(module)) as replay:
            replay.begin("text", sample(), reference=None)
            actual = module.route_tokens_to_experts(logits)
            record = replay.finish()
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
            replay.begin("text", sample(), reference=record)
            again = module.route_tokens_to_experts(logits)
            replay.finish()
            assert all(torch.equal(a, b) for a, b in zip(again, expected))
            raise RuntimeError("injected")
    assert "route_tokens_to_experts" not in module.__dict__
    assert all(torch.equal(a, b) for a, b in zip(module.route_tokens_to_experts(logits), expected))


def test_reference_rejects_wrong_token_order_and_layer_coverage():
    module = ToyMoE()
    with HFRouteReplay(toy_model(module)) as replay:
        replay.begin("text", sample(), reference=None)
        module.route_tokens_to_experts(torch.randn(3, 4))
        record = replay.finish()
    kwargs = dict(case="text", sample=sample(), layers={0}, experts=4, topk=2)
    validate_reference(record, **kwargs)
    with pytest.raises(AssertionError):
        validate_reference(record, **(kwargs | {"layers": {0, 1}}))
    changed = sample()
    changed["input_ids"] = changed["input_ids"].flip(-1)
    with pytest.raises(AssertionError):
        validate_reference(record, **(kwargs | {"sample": changed}))


def test_duplicate_or_missing_router_calls_rejected():
    module = ToyMoE()
    with HFRouteReplay(toy_model(module)) as replay:
        replay.begin("text", sample(), reference=None)
        with pytest.raises(AssertionError, match="Missing"):
            replay.finish()
        replay.begin("text", sample(), reference=None)
        module.route_tokens_to_experts(torch.randn(3, 4))
        with pytest.raises(AssertionError, match="Repeated"):
            module.route_tokens_to_experts(torch.randn(3, 4))


def test_multibatch_fixture_rejected():
    changed = sample()
    changed["input_ids"] = changed["input_ids"].repeat(2, 1)
    with pytest.raises(AssertionError):
        fixture_identity(changed)


class ToyBridgeRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config, self.topk = SimpleNamespace(num_moe_experts=4), 2

    def forward(self, logits):
        ids, weights = native(logits, torch.zeros(4))
        probs = torch.zeros_like(logits).scatter(1, ids, weights)
        return probs, torch.zeros_like(probs, dtype=torch.bool).scatter(1, ids, True)


def test_bridge_observer_records_global_expert_ids_without_changing_outputs():
    router = ToyBridgeRouter()
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            decoder=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(router=router))])
        )
    )
    logits = torch.randn(3, 4)
    baseline = router(logits)
    recorder = BridgeRouteRecorder(model)
    recorder.begin("text", sample())
    observed = router(logits)
    record = recorder.finish()
    recorder.close()
    assert all(torch.equal(a, b) for a, b in zip(baseline, observed))
    expected = baseline[1].nonzero(as_tuple=True)[1].reshape(3, 2)
    assert torch.equal(record["layers"][0]["indices"], expected)


def test_reference_audit_weights_cannot_influence_replay():
    module = ToyMoE()
    logits = torch.randn(3, 4)
    with HFRouteReplay(toy_model(module)) as replay:
        replay.begin("text", sample(), reference=None)
        expected = module.route_tokens_to_experts(logits)
        record = replay.finish()
        record["layers"][0]["weights_for_audit_only"].fill_(float("nan"))
        replay.begin("text", sample(), reference=record)
        actual = module.route_tokens_to_experts(logits)
        replay.finish()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_media_identity_includes_hf_pixels_and_video_lists():
    first = sample()
    first["hf_pixel_values"] = [torch.ones(3, 4, 8)]
    first["pixel_values_videos"] = [torch.ones(3, 4, 8), torch.zeros(3, 4, 8)]
    identity = fixture_identity(first)
    first["hf_pixel_values"][0][0, 0, 0] = 2
    assert fixture_identity(first) != identity


def test_text_only_backbones_use_same_recorders():
    module = ToyMoE()
    with HFRouteReplay(toy_model(module).language_model) as replay:
        replay.begin("text", sample(), reference=None)
        module.route_tokens_to_experts(torch.randn(3, 4))
        assert set(replay.finish()["layers"]) == {0}


@pytest.mark.parametrize("wrapped", [False, True])
def test_remote_code_backbone_topology(wrapped):
    module = ToyMoE()
    language = SimpleNamespace(backbone=SimpleNamespace(layers=[SimpleNamespace(block_type="moe", mixer=module)]))
    model = SimpleNamespace(language_model=language) if wrapped else language
    with HFRouteReplay(model) as replay:
        replay.begin("text", sample(), reference=None)
        module.route_tokens_to_experts(torch.randn(3, 4))
        assert set(replay.finish()["layers"]) == {0}


def test_router_bias_mutation_invalidates_control():
    module = ToyMoE()
    with HFRouteReplay(toy_model(module)) as replay:
        replay.begin("text", sample(), reference=None)
        module.route_tokens_to_experts(torch.randn(3, 4))
        module.gate.e_score_correction_bias.add_(1)
        with pytest.raises(AssertionError, match="bias"):
            replay.finish()


@pytest.mark.parametrize("bad", [torch.zeros(3, 4, dtype=torch.bfloat16), torch.full((3, 4), float("nan"))])
def test_replay_rejects_non_fp32_or_nonfinite_gate_logits(bad):
    with pytest.raises(AssertionError, match="FP32"):
        replay_hf_selection(
            bad,
            native_indices=torch.tensor([[0, 1]]).repeat(3, 1),
            forced_indices=torch.tensor([[0, 1]]).repeat(3, 1),
            norm_topk_prob=True,
            scaling_factor=1.0,
        )


def test_selection_and_report_metrics():
    from examples.conversion.compare_hf_and_megatron.router_replay import compare_logits, select_logits

    values = torch.arange(30, dtype=torch.float32).reshape(1, 3, 10)
    selected = select_logits(values, sequence_length=3, positions=[0, 2], vocab=8)
    assert selected.shape == (2, 8)
    report = compare_logits(selected, selected + 5)
    assert report["top1_matches"] == 2
    assert report["raw_cosine_min"] < 1
    assert report["centered_cosine_min"] == pytest.approx(1)
    assert report["probability_tv_max"] == pytest.approx(0)


@pytest.mark.parametrize("positions", [[], [3], [-1], [0, 0]])
def test_invalid_logit_positions(positions):
    from examples.conversion.compare_hf_and_megatron.router_replay import select_logits

    with pytest.raises(ValueError, match="Positions"):
        select_logits(torch.zeros(1, 3, 4), sequence_length=3, positions=positions, vocab=4)


def test_wrong_logit_layout_and_nonfinite_report_rejected():
    from examples.conversion.compare_hf_and_megatron.router_replay import compare_logits, select_logits

    with pytest.raises(ValueError, match="Expected"):
        select_logits(torch.zeros(3, 1, 4), sequence_length=3, positions=[0], vocab=4)
    with pytest.raises(ValueError, match="Non-finite"):
        compare_logits(torch.zeros(1, 4), torch.full((1, 4), float("nan")))


def test_cli_rejects_simultaneous_image_and_video():
    from examples.conversion.compare_hf_and_megatron.router_replay import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "record",
                "--hf-model-path",
                "hf",
                "--megatron-model-path",
                "mcore",
                "--ep",
                "1",
                "--prompt",
                "hello",
                "--output",
                "routes.pt",
                "--image-path",
                "image",
                "--video-path",
                "video",
            ]
        )


def test_media_interface_rejects_legacy_composites():
    from examples.conversion.compare_hf_and_megatron.router_replay import validate_hf_media_interface

    class Legacy:
        def forward(self, pixel_values, image_flags):
            pass

    with pytest.raises(ValueError, match="legacy"):
        validate_hf_media_interface(Legacy(), kind="image")
    validate_hf_media_interface(Legacy(), kind="text")


def test_replay_orchestration_with_tiny_native_router(tmp_path, monkeypatch):
    """Exercise artifact loading, all controls, replay, and report with CPU toy modules."""
    import json
    import sys

    from examples.conversion.compare_hf_and_megatron import router_replay as script

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.moe = ToyMoE()
            self.language_model = toy_model(self.moe).language_model

        def forward(self, input_ids, **kwargs):
            assert kwargs["use_cache"] is False
            logits = torch.tensor([[1.0, 0.0, 2.0, -1.0]]).repeat(input_ids.shape[1], 1)
            ids, weights = self.moe.route_tokens_to_experts(logits)
            return SimpleNamespace(logits=torch.zeros_like(logits).scatter(1, ids, weights).unsqueeze(0))

    model = Model()
    fixture = sample()
    with HFRouteReplay(model) as recorder:
        recorder.begin("input", fixture, reference=None)
        natural = model(fixture["input_ids"], use_cache=False).logits[0]
        routes = recorder.finish()
    # A different reference set ensures this test exercises actual intervention.
    routes["layers"][0]["indices"][:] = torch.tensor([1, 3])
    routes["source"] = "megatron_natural"
    artifact = {
        "schema": 1,
        "controls": {"megatron_observer_bitwise_equal": True},
        "hf_model_path": "toy",
        "hf_revision": None,
        "sample": fixture,
        "routes": routes,
        "logits": natural,
        "positions": [0, 1, 2],
        "vocab_size": 4,
        "parallelism": {},
        "disable_hf_video_pruning": False,
    }
    route_path, output = tmp_path / "routes.pt", tmp_path / "report.json"
    torch.save(artifact, route_path)
    config = SimpleNamespace(model_type="nemotron_h", vocab_size=4)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: config),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: (model, {})),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "examples.conversion.compare_hf_and_megatron.compare",
        SimpleNamespace(_get_hf_forward_model=lambda model, *args: model),
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = script.build_parser().parse_args(
        [
            "replay",
            "--hf-model-path",
            "toy",
            "--routes",
            str(route_path),
            "--output",
            str(output),
            "--hf-device-map",
            "cpu",
        ]
    )
    script.replay(args)
    result = json.loads(output.read_text())
    assert result["status"] == "diagnostic_complete_controls_passed"
    assert result["natural_parity_replaced"] is False
    assert all(result["controls"].values())
    assert result["route_counts_by_layer"]["0"]["replay_overridden_token_rows"] == 3
    assert result["natural"]["raw_cosine_min"] > result["replayed"]["raw_cosine_min"]
    assert "route_tokens_to_experts" not in model.moe.__dict__
    # Preserve old evidence; do not silently overwrite an existing report.
    with pytest.raises(FileExistsError):
        script.replay(args)


def test_weights_only_loader_rejects_pickle_side_effect(tmp_path):
    import pickle
    from pathlib import Path

    from examples.conversion.compare_hf_and_megatron.router_replay import load_recording

    marker = tmp_path / "must-not-exist"

    class Payload:
        def __reduce__(self):
            return Path.touch, (marker,)

    path = tmp_path / "malicious.pt"
    torch.save(Payload(), path)
    with pytest.raises(pickle.UnpicklingError):
        load_recording(str(path))
    assert not marker.exists()


def test_record_orchestration_with_tiny_router(tmp_path, monkeypatch):
    """Exercise the record CLI path with isolated CPU substitutes for distributed setup."""
    import contextlib
    import sys

    import examples.conversion.compare_hf_and_megatron as comparison_package
    from examples.conversion.compare_hf_and_megatron import router_replay as script

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = ToyBridgeRouter()
            self.config = SimpleNamespace(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_tensor_parallel_size=1,
                expert_model_parallel_size=1,
                sequence_parallel=False,
            )
            self.language_model = SimpleNamespace(
                decoder=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(router=self.router))])
            )

        def forward(self, input_ids, **kwargs):
            logits = torch.tensor([[1.0, 0.0, 2.0, -1.0]]).repeat(input_ids.shape[1], 1)
            return self.router(logits)[0].unsqueeze(0)

    model = Model()
    config = SimpleNamespace(vocab_size=4)
    monkeypatch.setattr(script, "_config", lambda args: (config, config))
    monkeypatch.setattr(script, "_inputs", lambda *args: sample())
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda *a: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    def gather(output, value):
        output[0] = value

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.inference.utils",
        SimpleNamespace(InferenceMode=SimpleNamespace(active=contextlib.nullcontext)),
    )
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.training.nemotron_omni_step",
        SimpleNamespace(_build_vision_packed_seq_params=lambda sizes: None),
    )
    comparison_stub = SimpleNamespace(_load_megatron_model=lambda args: ([model], None))
    monkeypatch.setitem(sys.modules, "examples.conversion.compare_hf_and_megatron.compare", comparison_stub)
    # Other CI tests may already have imported compare into the parent package.
    monkeypatch.setattr(comparison_package, "compare", comparison_stub, raising=False)
    path = tmp_path / "record.pt"
    args = script.build_parser().parse_args(
        [
            "record",
            "--hf-model-path",
            "toy",
            "--megatron-model-path",
            "mcore",
            "--prompt",
            "hello",
            "--ep",
            "1",
            "--positions",
            "all",
            "--output",
            str(path),
        ]
    )
    script.record(args)
    result = script.load_recording(str(path))
    assert result["logits"].shape == (3, 4)
    assert result["controls"]["megatron_observer_bitwise_equal"] is True
    assert len(result["routes"]["layers"]) == 1
    assert not model.router._forward_hooks
    with pytest.raises(FileExistsError):
        script.record(args)
    with pytest.raises(AssertionError, match="different"):
        result["sample"]["input_ids"] = result["sample"]["input_ids"].flip(-1)
        corrupted = tmp_path / "corrupted.pt"
        torch.save(result, corrupted)
        script.load_recording(str(corrupted))
