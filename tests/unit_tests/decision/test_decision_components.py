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

"""CPU unit tests for decision-model post-training utilities."""

import math

import pytest
import torch

from megatron.bridge.decision import clef_head as clef
from megatron.bridge.decision import clm_heads as clm
from megatron.bridge.decision import clm_prompting, losses, metrics
from megatron.bridge.decision import records as schema


QUESTIONS = {
    "verdict": {
        "type": "choice",
        "instructions": "What is the verdict?",
        "criteria": {"malicious": "Real attack.", "benign_authorized": "Authorized.", "false_positive": "Misfire."},
    },
    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["Info", "Low", "Medium", "High"]},
    "needs_human": {"type": "noul", "instructions": "Does a human need to review this?"},
}


def make_record(rid="r1", verdict="malicious", severity="2", needs_human="true"):
    return {
        "id": rid,
        "state": {"alert": {"title": "PsExec to 3 hosts", "events": ["a", "b"]}, "asset": {"criticality": "high"}},
        "questions": QUESTIONS,
        "answers": {"verdict": verdict, "severity": severity, "needs_human": needs_human},
        "rewards": {
            "verdict": {"malicious": 1.0, "benign_authorized": -1.0, "false_positive": -1.0},
            "severity": {"0": 0.25, "1": 0.5, "2": 1.0, "3": 0.75},
            "needs_human": {"true": 1.0, "false": -0.5},
        },
        "meta": {"evidence": "clear"},
    }


class CharTokenizer:
    """Deterministic offline tokenizer: one id per character (enough for span bookkeeping)."""

    pad_token_id = 0

    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        return self._Out([1 + (ord(c) % 250) for c in text])


# ---------------------------------------------------------------------------------------------------------------------
def test_validate_and_targets_follow_model_order():
    record = make_record()
    schema.validate_record(record)
    order = {"verdict": ["false_positive", "malicious", "benign_authorized"], "needs_human": ["false", "true"]}
    targets = schema.question_targets(record, order)
    assert [t.question_id for t in targets] == ["verdict", "needs_human"]
    assert targets[0].label == 1 and targets[0].rewards == (-1.0, 1.0, -1.0)
    assert targets[1].label == 1 and targets[1].rewards == (-0.5, 1.0)


def test_validate_rejects_unknown_answer():
    record = make_record(verdict="ransomware")
    with pytest.raises(ValueError):
        schema.validate_record(record)


def test_sft_loss_prefers_gold_and_is_finite():
    record = make_record()
    order = {q: schema.canonical_option_ids(QUESTIONS[q]) for q in QUESTIONS}
    t = losses.build_targets(schema.question_targets(record, order))
    logits = torch.zeros(t.option_mask.shape, requires_grad=True)
    loss, m = losses.DecisionLossConfig(mode="sft").build()(logits, t)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    # gradient descent raises the gold logit
    for i, label in enumerate(t.labels.tolist()):
        assert logits.grad[i, label] < 0
    # padded options get no gradient
    assert (logits.grad[~t.option_mask] == 0).all()


def test_expected_reward_gradient_is_exact_policy_gradient():
    t = losses.DecisionTargets(
        labels=torch.tensor([0]),
        option_mask=torch.tensor([[True, True, True]]),
        question_types=torch.tensor([1]),
        rewards=torch.tensor([[1.0, -1.0, 0.5]]),
        reward_mask=torch.tensor([True]),
    )
    logits = torch.tensor([[0.3, -0.2, 0.1]], requires_grad=True)
    cfg = losses.DecisionLossConfig(mode="rl", rl_objective="expected_reward", kl_coef=0.0, anchor_weight=0.0)
    loss, m = cfg.build()(logits, t)
    loss.backward()
    p = torch.softmax(logits.detach(), -1)[0]
    r = t.rewards[0]
    expected_grad = -(p * (r - (p * r).sum()))
    assert torch.allclose(logits.grad[0], expected_grad, atol=1e-6)
    assert math.isclose(m["expected_reward"].item(), (p * r).sum().item(), rel_tol=1e-5)


def test_grpo_matches_expected_reward_in_expectation():
    t = losses.DecisionTargets(
        labels=torch.tensor([0]),
        option_mask=torch.tensor([[True, True, True, False]]),
        question_types=torch.tensor([1]),
        rewards=torch.tensor([[1.0, -1.0, 0.5, 0.0]]),
        reward_mask=torch.tensor([True]),
    )
    logits = torch.tensor([[0.3, -0.2, 0.1, 0.0]], requires_grad=True)
    cfg = losses.DecisionLossConfig(
        mode="rl", rl_objective="grpo", num_samples=20000, normalize_advantages=False, kl_coef=0.0, anchor_weight=0.0
    )
    loss, _ = cfg.build()(logits, t, generator=torch.Generator().manual_seed(0))
    loss.backward()
    p = torch.softmax(logits.detach()[0, :3], -1)
    r = t.rewards[0, :3]
    exact = -(p * (r - (p * r).sum()))
    assert torch.allclose(logits.grad[0, :3], exact, atol=2e-2)
    assert logits.grad[0, 3] == 0


def test_kl_to_identical_reference_is_zero():
    t = losses.build_targets(
        schema.question_targets(make_record(), {q: schema.canonical_option_ids(QUESTIONS[q]) for q in QUESTIONS})
    )
    logits = torch.randn(t.option_mask.shape)
    ref = losses.masked_log_softmax(logits, t.option_mask)
    t.ref_logprobs = ref
    _, m = losses.DecisionLossConfig(mode="rl").build()(logits, t)
    assert abs(m["kl"].item()) < 1e-6


def test_metrics_summary_and_table():
    ev = metrics.DecisionEvaluator(slice_keys=["evidence"])
    ev.add("r1", "verdict", ["a", "b"], [0.9, 0.1], 0, [1.0, -1.0], {"evidence": "clear"})
    ev.add("r1", "severity", ["0", "1"], [0.4, 0.6], 0, [1.0, 0.5], {"evidence": "clear"})
    ev.add("r2", "verdict", ["a", "b"], [0.2, 0.8], 1, [-1.0, 1.0], {"evidence": "subtle"})
    s = ev.summary()
    assert s["questions"]["verdict"]["accuracy"] == 1.0
    assert s["overall"]["exact_match"] == 0.5
    assert math.isclose(s["overall"]["reward"], (1.0 + 0.5 + 1.0) / 3)
    table = metrics.format_comparison({"before": s, "after": s})
    assert "| verdict |" in table and "overall/exact_match" in table


# ---------------------------------------------------------------------------------------------------------------------
def test_clm_rendering_matches_reference_package():
    clm_schema = pytest.importorskip("clm.schema")
    record = make_record()
    mine = clm_prompting.clm_pairs(record)
    ref = clm_schema.build_pairs(record["state"], record["questions"])
    assert mine == {k: (v[0], list(v[1]), list(v[2])) for k, v in ref.items()}


def test_clm_heads_roundtrip_and_logits(tmp_path):
    cfg = {"width": 64, "depth": 3, "projection_dim": 16, "layernorm": True, "residual": True, "hidden_size": 32}
    heads = clm.CLMHeads(cfg)
    path = tmp_path / "heads.pt"
    clm.save_clm_heads(heads, path)
    loaded = clm.load_clm_heads(path)
    states, options = torch.randn(2, 32), torch.randn(5, 32)
    a = heads.question_logits(states, options, [3, 2])
    b = loaded.question_logits(states, options, [3, 2])
    assert [x.shape[0] for x in a] == [3, 2]
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-6)
    assert all(x.abs().max() <= heads.scale + 1e-4 for x in a)


def test_clm_heads_match_reference_package(tmp_path):
    clm_heads = pytest.importorskip("clm.heads")
    import numpy as np

    cfg = {"width": 64, "depth": 3, "projection_dim": 16, "layernorm": True, "residual": False, "hidden_size": 32}
    heads = clm.CLMHeads(cfg, logit_scale=2.0)
    path = tmp_path / "heads.pt"
    clm.save_clm_heads(heads, path)
    pair = clm_heads.HeadPair("t", str(path), device="cpu").ensure()
    states = np.random.randn(4, 32).astype("float32")
    zs_ref, za_ref = pair.project(states, states)
    with torch.no_grad():
        zs = heads.project_states(torch.from_numpy(states)).numpy()
    assert np.allclose(zs, zs_ref, atol=1e-5)
    assert math.isclose(pair.scale, heads.scale.item(), rel_tol=1e-6)


def test_embedding_table_and_clm_batch():
    records = [make_record("a"), make_record("b", verdict="false_positive")]
    texts = clm.collect_clm_texts(records)
    table = clm.EmbeddingTable(texts, torch.randn(len(texts), 32))
    states, options, counts, orders = clm.clm_batch(records, table)
    assert states.shape == (6, 32) and options.shape[0] == sum(counts) == 2 * (3 + 4 + 2)
    assert orders[0]["needs_human"] == ["false", "true"]


# ---------------------------------------------------------------------------------------------------------------------
def tiny_head():
    torch.manual_seed(0)
    return clef.JointSchemaHead(hidden_size=32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)


def test_clef_encode_head_and_targets():
    tok = CharTokenizer()
    records = [make_record("a"), make_record("b", verdict="benign_authorized", severity="0", needs_human="false")]
    encoded = [clef.encode_record(tok, r, max_length=4096) for r in records]
    batch = clef.collate_records(encoded, tok.pad_token_id, torch.device("cpu"))
    head = tiny_head()
    vocab = torch.randn(256, 32)
    hidden = torch.randn(batch["input_ids"].shape + (32,))
    out = head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"], vocab)
    assert [len(x) for x in out] == [3, 3]
    order = clef.clef_option_orders(encoded[0])
    assert order["verdict"] == ("benign_authorized", "false_positive", "malicious")  # Clef sorts choice keys
    assert order["needs_human"] == ("true", "false")
    targets = [schema.question_targets(r, clef.clef_option_orders(e)) for r, e in zip(records, encoded)]
    t = losses.build_targets([q for ts in targets for q in ts])
    logits, mask = losses.pad_logits([q for rec in out for q in rec])
    assert torch.equal(mask, t.option_mask)
    loss, _ = losses.DecisionLossConfig(mode="sft").build()(logits, t)
    loss.backward()
    assert head.prior_logit_scale.grad is not None


def test_clef_lexical_table_matches_dense_lookup():
    tok = CharTokenizer()
    encoded = [clef.encode_record(tok, make_record())]
    batch = clef.collate_records(encoded, tok.pad_token_id, torch.device("cpu"))
    head = tiny_head().eval()
    vocab = torch.randn(256, 32)
    ids = clef.option_token_ids(encoded)
    table = clef.LexicalTable(ids, vocab[ids])
    hidden = torch.randn(batch["input_ids"].shape + (32,))
    with torch.no_grad():
        a = head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"], vocab)
        b = head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"], table)
    for x, y in zip(a[0], b[0]):
        assert torch.allclose(x, y)


def test_clef_head_save_load_roundtrip(tmp_path):
    head = tiny_head()
    clef.save_joint_head(head, tmp_path, dtype=torch.float32)
    loaded = clef.load_joint_head(tmp_path)
    for (k, v), (k2, v2) in zip(head.state_dict().items(), loaded.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)


class _UnsafeCheckpointValue:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')",)


def test_clm_checkpoint_rejects_arbitrary_pickle_execution(tmp_path):
    import pickle

    marker = tmp_path / "executed"
    path = tmp_path / "unsafe.pt"
    torch.save({"payload": _UnsafeCheckpointValue(marker)}, path)
    with pytest.raises(pickle.UnpicklingError):
        clm.load_clm_heads(path)
    assert not marker.exists()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_clm_grpo_training_uses_device_matched_generator(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA regression requires a real GPU")
    from megatron.bridge.decision.clm_trainer import CLMTrainConfig, train_heads

    records = [make_record()]
    texts = clm.collect_clm_texts(records)
    table = clm.EmbeddingTable(texts, torch.randn(len(texts), 32))
    cfg = {"width": 16, "depth": 2, "projection_dim": 8, "hidden_size": 32}
    heads = clm.CLMHeads(cfg).to(device)
    trained = train_heads(
        heads,
        records,
        None,
        table,
        loss_config=losses.DecisionLossConfig(mode="rl", rl_objective="grpo"),
        train_config=CLMTrainConfig(epochs=1, batch_size=1),
        device=torch.device(device),
    )
    assert all(torch.isfinite(parameter).all() for parameter in trained.parameters())


def _two_option_targets(labels, *, rewards=None, weights=None):
    return losses.DecisionTargets(
        labels=torch.tensor(labels),
        option_mask=torch.ones(len(labels), 2, dtype=torch.bool),
        question_types=torch.ones(len(labels), dtype=torch.long),
        rewards=None if rewards is None else torch.tensor(rewards),
        weights=None if weights is None else torch.tensor(weights),
    )


def test_unanswered_unrewarded_rl_is_graph_connected_zero():
    logits = torch.tensor([[0.3, -0.2]], requires_grad=True)
    targets = _two_option_targets([-100])
    objective = losses.DecisionLossConfig(mode="rl").build()
    loss, metrics = objective(logits, targets)
    loss.backward()
    assert loss.item() == 0
    assert objective.normalization_weight(targets).item() == 0
    assert metrics["normalization_weight"].item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_reward_only_unlabelled_rl_retains_explicit_feedback():
    logits = torch.tensor([[0.3, -0.2]], requires_grad=True)
    targets = _two_option_targets([-100], rewards=[[1.0, -1.0]])
    objective = losses.DecisionLossConfig(mode="rl", kl_coef=0, anchor_weight=1).build()
    loss, _ = objective(logits, targets)
    expected = -(logits.softmax(-1) * targets.rewards).sum()
    torch.testing.assert_close(loss, expected)
    assert objective.normalization_weight(targets).item() == 1
    loss.backward()
    assert logits.grad[0, 0] < 0


def test_unrewarded_unlabelled_rows_do_not_dilute_mixed_rl_or_apply_kl():
    objective = losses.DecisionLossConfig(mode="rl", kl_coef=0.3).build()
    one = torch.tensor([[0.3, -0.2]], requires_grad=True)
    reference = _two_option_targets([1], weights=[2.0])
    expected, _ = objective(one, reference)
    expected.backward()
    mixed = torch.tensor([[0.3, -0.2], [-0.7, 0.6]], requires_grad=True)
    targets = _two_option_targets([1, -100], weights=[2.0, 19.0])
    targets.ref_logprobs = torch.tensor([one.detach()[0].log_softmax(-1).tolist(), [-0.2, -1.7]])
    actual, _ = objective(mixed, targets)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(mixed.grad[0], one.grad[0])
    assert torch.equal(mixed.grad[1], torch.zeros(2))
    assert objective.normalization_weight(targets).item() == 2


def test_empty_option_row_is_rejected_before_softmax():
    targets = _two_option_targets([0])
    targets.option_mask.zero_()
    with pytest.raises(ValueError, match="valid option"):
        losses.DecisionLossConfig().build()(torch.zeros(1, 2), targets)


def test_global_accumulation_weight_matches_single_process_question_mean():
    objective = losses.DecisionLossConfig(mode="sft", label_smoothing=0, brier_weight=0, ordinal_weight=0).build()
    # Two DP ranks, two microbatches each, with unequal question counts and a
    # feedback-free local microbatch that must still participate in backward.
    windows = [
        [_two_option_targets([0]), _two_option_targets([-100])],
        [_two_option_targets([1, 1, 1]), _two_option_targets([1])],
    ]
    global_weight = sum(objective.normalization_weight(t) for rank in windows for t in rank)
    per_rank = []
    for rank in windows:
        theta = torch.tensor([0.3, -0.2], requires_grad=True)
        for targets in rank:
            loss, _ = objective(theta.expand(len(targets.labels), -1), targets)
            (loss * 2 * objective.normalization_weight(targets) / global_weight).backward()
        per_rank.append(theta.grad)
    reference_theta = torch.tensor([0.3, -0.2], requires_grad=True)
    reference_targets = _two_option_targets([0, -100, 1, 1, 1, 1])
    reference_loss, _ = objective(reference_theta.expand(6, -1), reference_targets)
    reference_loss.backward()
    torch.testing.assert_close(torch.stack(per_rank).mean(0), reference_theta.grad)


def test_clm_training_rejects_batch_without_feedback():
    from megatron.bridge.decision.clm_trainer import CLMTrainConfig, train_heads

    record = make_record()
    record["answers"] = {}
    record["rewards"] = {}
    texts = clm.collect_clm_texts([record])
    table = clm.EmbeddingTable(texts, torch.randn(len(texts), 32))
    heads = clm.CLMHeads({"width": 16, "depth": 2, "projection_dim": 8, "hidden_size": 32})
    with pytest.raises(ValueError, match="eligible labels or reward"):
        train_heads(
            heads,
            [record],
            None,
            table,
            loss_config=losses.DecisionLossConfig(mode="rl"),
            train_config=CLMTrainConfig(epochs=1, batch_size=1),
            device=torch.device("cpu"),
        )


def test_clm_reward_only_validation_does_not_require_accuracy():
    from megatron.bridge.decision.clm_trainer import CLMTrainConfig, train_heads

    record = make_record()
    record["answers"] = {}
    texts = clm.collect_clm_texts([record])
    table = clm.EmbeddingTable(texts, torch.randn(len(texts), 32))
    heads = clm.CLMHeads({"width": 16, "depth": 2, "projection_dim": 8, "hidden_size": 32})
    result = train_heads(
        heads,
        [record],
        [record],
        table,
        loss_config=losses.DecisionLossConfig(mode="rl"),
        train_config=CLMTrainConfig(epochs=1, batch_size=1, eval_metric="reward"),
        device=torch.device("cpu"),
    )
    assert all(torch.isfinite(parameter).all() for parameter in result.parameters())


def test_grpo_scores_the_tempered_sampling_policy():
    targets = _two_option_targets([-100], rewards=[[1.0, -0.5]])
    logits = torch.tensor([[0.3, -0.2]], requires_grad=True)
    temperature = 0.6
    objective = losses.DecisionLossConfig(
        mode="rl",
        rl_objective="grpo",
        num_samples=100000,
        sample_temperature=temperature,
        normalize_advantages=False,
        kl_coef=0,
        anchor_weight=0,
    ).build()
    loss, _ = objective(logits, targets, generator=torch.Generator().manual_seed(13))
    loss.backward()
    probabilities = (logits.detach()[0] / temperature).softmax(-1)
    rewards = targets.rewards[0]
    expected_gradient = -probabilities * (rewards - (probabilities * rewards).sum()) / temperature
    torch.testing.assert_close(logits.grad[0], expected_gradient, atol=0.01, rtol=0.01)
