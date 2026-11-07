"""CPU checks for the strict-validator machinery, without publisher code."""

import pytest
import torch
from d1_bridge_validation import assert_complete_round_trip, copy_and_round_trip
from d1_publisher_fixtures import initialize_nonunit_rmsnorm


class IdentityMapping:
    """Fixture mapping: tests coverage checks, not native conversion correctness."""

    def __init__(self, name):
        self.hf_param = name

    def hf_to_megatron(self, source, owner):
        return source

    def megatron_to_hf(self, tensor, owner):
        return {self.hf_param: tensor}


class IdentityRegistry:
    def megatron_to_hf_lookup(self, name):
        return IdentityMapping(name)


def _fixture():
    module = torch.nn.Linear(3, 2, bias=False)
    module.register_buffer("statistics", torch.zeros(2))
    module.register_buffer("scratch", torch.zeros(2), persistent=False)
    original = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "statistics": torch.tensor([2.0, 4.0])}
    return module, original


def test_complete_state_includes_persistent_buffer_and_ignores_nonpersistent_buffer():
    native, original = _fixture()
    copy_and_round_trip(native, IdentityRegistry(), original, [("statistics", native.statistics)])
    assert set(native.state_dict()) == set(original)
    for name, tensor in native.state_dict().items():
        assert torch.equal(tensor, original[name])
    assert torch.equal(native.scratch, torch.zeros(2))


def test_parameter_only_conversion_cannot_pass_full_state_coverage():
    native, original = _fixture()
    with pytest.raises(AssertionError, match="statistics"):
        copy_and_round_trip(native, IdentityRegistry(), original, [])


@pytest.mark.parametrize("missing", ["consumed", "exported"])
def test_full_state_rejects_missing_checkpoint_keys(missing):
    original = {"weight": torch.ones(2), "buffer": torch.ones(1)}
    consumed, exported = set(original), dict(original)
    if missing == "consumed":
        consumed.remove("buffer")
    else:
        del exported["buffer"]
    with pytest.raises(AssertionError, match="buffer"):
        assert_complete_round_trip(original, consumed, exported)


def test_full_state_rejects_unexpected_checkpoint_keys():
    original = {"weight": torch.ones(2)}
    with pytest.raises(AssertionError, match="extra"):
        assert_complete_round_trip(original, set(original), {**original, "extra": torch.ones(1)})


@pytest.mark.parametrize("change", ["value", "dtype"])
def test_exact_round_trip_rejects_changed_values_or_dtype(change):
    original = {"weight": torch.ones(2)}
    value = original["weight"] + 1 if change == "value" else original["weight"].bfloat16()
    with pytest.raises(AssertionError, match="weight"):
        assert_complete_round_trip(original, set(original), {"weight": value})


def test_nonunit_norm_fixture_changes_all_affine_weights_and_enforces_coverage():
    class FixtureRMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(8))

    native = torch.nn.ModuleList([FixtureRMSNorm() for _ in range(7)])
    assert initialize_nonunit_rmsnorm(native) == 7
    for module in native:
        assert torch.equal(module.weight, torch.linspace(0.65, 1.35, 8))
    with pytest.raises(ValueError, match="coverage"):
        initialize_nonunit_rmsnorm(torch.nn.ModuleList([FixtureRMSNorm()]))


@pytest.mark.parametrize("tied", [True, False])
def test_full_state_checks_tied_embedding_alias_and_identity(tied):
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 3)
            self.lm_head = torch.nn.Linear(3, 4, bias=False)
            if tied:
                self.lm_head.weight = self.embedding.weight

        def get_input_embeddings(self):
            return self.embedding

    class Registry(IdentityRegistry):
        def megatron_to_hf_lookup(self, name):
            if name == "embedding.weight":
                name = "model.language_model.embed_tokens.weight"
            return IdentityMapping(name)

    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    original = {"model.language_model.embed_tokens.weight": weight, "lm_head.weight": weight}
    native = Fixture()
    if tied:
        copy_and_round_trip(native, Registry(), original, [])
        assert native.lm_head.weight is native.embedding.weight
        assert torch.equal(native.lm_head.weight, weight)
    else:
        with pytest.raises(AssertionError):
            copy_and_round_trip(native, Registry(), original, [])
