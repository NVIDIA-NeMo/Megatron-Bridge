import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[3]
SWITCH_SCRIPT = REPO_ROOT / "scripts" / "switch_mcore.sh"
pytestmark = pytest.mark.unit


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _commit(cwd: Path, message: str) -> str:
    _git(cwd, "add", "tracked.txt")
    _git(
        cwd,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        message,
    )
    return _git(cwd, "rev-parse", "HEAD").stdout.strip()


def _create_test_checkout(tmp_path: Path) -> tuple[Path, Path, str, str]:
    remote = tmp_path / "mcore.git"
    source = tmp_path / "mcore-source"
    bridge = tmp_path / "bridge"
    submodule = bridge / "3rdparty" / "Megatron-LM"

    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", str(source))
    (source / "tracked.txt").write_text("main\n")
    main_commit = _commit(source, "main commit")
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "origin", "HEAD:main")

    bridge.mkdir()
    (bridge / "scripts").mkdir()
    (bridge / "3rdparty").mkdir()
    subprocess.run(
        ["git", "clone", "--no-local", "--branch", "main", str(remote), str(submodule)],
        check=True,
        capture_output=True,
        text=True,
    )

    (source / "tracked.txt").write_text("dev\n")
    dev_commit = _commit(source, "dev commit")
    _git(source, "push", "origin", "HEAD:dev")

    (bridge / ".main.commit").write_text(f"{main_commit}\n")
    (bridge / ".dev.commit").write_text(f"{dev_commit}\n")
    (bridge / "scripts" / "switch_mcore.sh").write_bytes(SWITCH_SCRIPT.read_bytes())
    (bridge / "scripts" / "switch_mcore.sh").chmod(0o755)
    return bridge, submodule, main_commit, dev_commit


def test_switch_fetches_missing_pinned_commit(tmp_path: Path) -> None:
    bridge, submodule, main_commit, dev_commit = _create_test_checkout(tmp_path)
    assert _git(submodule, "cat-file", "-e", f"{dev_commit}^{{commit}}", check=False).returncode != 0

    result = subprocess.run(
        [str(bridge / "scripts" / "switch_mcore.sh"), "dev"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"Before: {main_commit[:12]}" in result.stdout
    assert f"Pinned dev commit {dev_commit[:12]} is not available locally. Fetching from origin..." in result.stdout
    assert _git(submodule, "rev-parse", "HEAD").stdout.strip() == dev_commit


def test_switch_does_not_fetch_commit_that_is_already_present(tmp_path: Path) -> None:
    bridge, submodule, _, dev_commit = _create_test_checkout(tmp_path)
    _git(submodule, "fetch", "origin", dev_commit)
    _git(submodule, "remote", "set-url", "origin", str(tmp_path / "unavailable.git"))

    result = subprocess.run(
        [str(bridge / "scripts" / "switch_mcore.sh"), "dev"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Fetching from origin" not in result.stdout
    assert _git(submodule, "rev-parse", "HEAD").stdout.strip() == dev_commit


def test_switch_reports_fetch_failure_before_checkout(tmp_path: Path) -> None:
    bridge, submodule, main_commit, dev_commit = _create_test_checkout(tmp_path)
    _git(submodule, "remote", "set-url", "origin", str(tmp_path / "unavailable.git"))

    result = subprocess.run(
        [str(bridge / "scripts" / "switch_mcore.sh"), "dev"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert f"Failed to fetch dev commit {dev_commit} from origin" in result.stderr
    assert "reference is not a tree" not in result.stderr
    assert _git(submodule, "rev-parse", "HEAD").stdout.strip() == main_commit
