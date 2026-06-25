import pytest

from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


@pytest.mark.parametrize(
    "hf_path",
    [
        "Qwen/attacker_processor",
        "nvidia/local-model",
        "google/custom-tokenizer",
        "meta-llama/custom-dataset",
        "./Qwen/attacker_processor",
        "/tmp/attacker_processor",
    ],
)
def test_is_safe_repo_defaults_to_no_remote_code(hf_path):
    """Test that omitted trust_remote_code never enables remote code."""
    assert is_safe_repo(hf_path=hf_path, trust_remote_code=None) is False


def test_is_safe_repo_explicit_trust_remote_code_wins():
    """Test that explicit trust_remote_code values are honored."""
    assert is_safe_repo(hf_path="attacker/repo", trust_remote_code=True) is True
    assert is_safe_repo(hf_path="Qwen/model", trust_remote_code=False) is False
