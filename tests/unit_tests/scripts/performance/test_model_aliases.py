"""Tests for public performance model aliases."""

from scripts.performance.utils.model_aliases import resolve_model_alias


def test_resolve_nemodiag_v0_alias() -> None:
    """Map the public diagnostic name to its backing performance recipe."""
    assert resolve_model_alias("nemodiag", "nemodiag_v0") == ("deepseek", "deepseek_v3")


def test_resolve_model_alias_preserves_other_recipes() -> None:
    """Leave model recipes without aliases unchanged."""
    assert resolve_model_alias("llama", "llama3_8b") == ("llama", "llama3_8b")
