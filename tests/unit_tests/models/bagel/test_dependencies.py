from unittest.mock import patch

import pytest

from megatron.bridge.models.bagel.dependencies import (
    configure_official_bagel_repo,
    import_official_bagel_module,
)


def test_configure_official_bagel_repo_rejects_invalid_checkout(tmp_path):
    """Reject a path that does not contain the official source layout."""
    with pytest.raises(ImportError, match=r"uv sync --extra bagel"):
        configure_official_bagel_repo(str(tmp_path))


def test_import_official_bagel_module_adds_install_hint():
    """Replace a transitive import failure with an actionable BAGEL error."""
    with patch("megatron.bridge.models.bagel.dependencies.importlib.import_module", side_effect=ImportError):
        with pytest.raises(ImportError, match=r"uv sync --extra bagel"):
            import_official_bagel_module("modeling.bagel")
