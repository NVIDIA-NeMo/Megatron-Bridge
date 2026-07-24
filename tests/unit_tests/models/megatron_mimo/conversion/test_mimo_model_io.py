from pathlib import Path

import pytest

from megatron.bridge.models.megatron_mimo.conversion.mimo_model_io import load_megatron_mimo_model


@pytest.mark.unit
def test_parent_load_uses_tracker_selected_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider config should come from the same durable iteration as weights."""
    durable_iteration = tmp_path / "iter_0000010"
    durable_iteration.mkdir()
    (durable_iteration / "run_config.yaml").touch()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("10", encoding="utf-8")

    # Async save creates its iteration directory before finalization updates the
    # tracker and writes run_config.yaml. An interrupted save can leave it behind.
    (tmp_path / "iter_0000020").mkdir()

    selected_config_path: Path | None = None

    class StopAfterConfigSelection(Exception):
        pass

    def record_config_path(checkpoint_path: str) -> None:
        nonlocal selected_config_path
        selected_config_path = Path(checkpoint_path)
        raise StopAfterConfigSelection

    monkeypatch.setattr(
        "megatron.bridge.training.model_load_save.load_model_config",
        record_config_path,
    )

    with pytest.raises(StopAfterConfigSelection):
        load_megatron_mimo_model(tmp_path)

    assert selected_config_path == durable_iteration
