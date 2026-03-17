# Remote Debug Workflow for Megatron-Hub

## Environment Details

- **SSH Host**: `yuya@cw-dfw-cs-001-login-01`
- **Tmux Session**: `aa`
- **Container Project Path**: `/opt/Megatron-Bridge`
- **Local Project Path**: `/Users/yuya/Library/CloudStorage/OneDrive-NVIDIACorporation/Documents/projects/Megatron-Hub/`
- **Remote Sync Path**: `/lustre/fsw/portfolios/coreai/users/yuya/Megatron-Hub`

## Syncing Code to Remote

### Rsync command to update code on remote server
```bash
rsync -avz --progress \
  --exclude='.git/' --exclude='.venv/' --exclude='docs/_build/' --exclude='3rdparty' \
  --exclude='.gitignore' \
  --exclude='.gitattributes' \
  "/Users/yuya/Library/CloudStorage/OneDrive-NVIDIACorporation/Documents/projects/Megatron-Hub/" \
  yuya@cw-dfw-cs-001-login-01:/lustre/fsw/portfolios/coreai/users/yuya/Megatron-Hub
```

## Running Commands on Remote

### Method 1: Send command to tmux session (interactive)
```bash
ssh yuya@cw-dfw-cs-001-login-01 "tmux send-keys -t aa '<COMMAND>' Enter"
```

### Method 2: Capture output after running command
```bash
# Send command
ssh yuya@cw-dfw-cs-001-login-01 "tmux send-keys -t aa '<COMMAND>' Enter"

# Wait and capture output (adjust sleep time based on expected duration)
ssh yuya@cw-dfw-cs-001-login-01 "sleep <SECONDS>; tmux capture-pane -t aa -p -S -<NUM_LINES>"
```

### Method 3: Combined send and capture
```bash
ssh yuya@cw-dfw-cs-001-login-01 "tmux send-keys -t aa '<COMMAND>' Enter; sleep <SECONDS>; tmux capture-pane -t aa -p -S -100"
```

## Running Tests

### Run specific test file
```bash
ssh yuya@cw-dfw-cs-001-login-01 "tmux send-keys -t aa 'cd /opt/Megatron-Bridge && uv run python -m pytest tests/unit_tests/path/to/test_file.py -v' Enter"
```

### Run all unit tests
```bash
ssh yuya@cw-dfw-cs-001-login-01 "tmux send-keys -t aa 'cd /opt/Megatron-Bridge && uv run python -m pytest tests/unit_tests/ -v' Enter"
```

### Run specific test class or method
```bash
# Specific class
ssh yuya@cw-dfw-cs-001-login-01 "tmux send-keys -t aa 'cd /opt/Megatron-Bridge && uv run python -m pytest tests/unit_tests/path/to/test_file.py::TestClassName -v' Enter"

# Specific method
ssh yuya@cw-dfw-cs-001-login-01 "tmux send-keys -t aa 'cd /opt/Megatron-Bridge && uv run python -m pytest tests/unit_tests/path/to/test_file.py::TestClassName::test_method_name -v' Enter"
```

### Run tests with pattern matching
```bash
ssh yuya@cw-dfw-cs-001-login-01 "tmux send-keys -t aa 'cd /opt/Megatron-Bridge && uv run python -m pytest tests/unit_tests/ -k \"pattern\" -v' Enter"
```

## Useful Tmux Commands

### List sessions
```bash
ssh yuya@cw-dfw-cs-001-login-01 "tmux list-sessions"
```

### View current tmux pane content
```bash
ssh yuya@cw-dfw-cs-001-login-01 "tmux capture-pane -t aa -p -S -100"  # Last 100 lines
```

### Check if command is still running
```bash
ssh yuya@cw-dfw-cs-001-login-01 "tmux capture-pane -t aa -p | tail -5"  # Check for prompt
```

## Notes

- The container uses `uv run` to execute Python commands (manages dependencies via `uv.lock`)
- **Important**: Use `uv run python -m pytest` (not `uv run pytest`) and `uv run python -m torch.distributed.run` (not `uv run torchrun`)
- Project uses pytest with configuration in `pyproject.toml`
- Tests are organized in `tests/unit_tests/` and `tests/functional_tests/`
- Test markers available: `unit`, `integration`, `system`, `acceptance`, `pleasefixme`

## Workflow Summary

1. **Edit code locally** in `/Users/yuya/.../Megatron-Hub/`
2. **Sync code** using rsync to `/lustre/fsw/portfolios/coreai/users/yuya/Megatron-Hub`
3. **Run tests** in the container at `/opt/Megatron-Bridge` via tmux session `aa`

**Note**: The container's `/opt/Megatron-Bridge` may be a separate mount or copy. Verify if rsync destination is mounted into the container, or if additional steps are needed to update the container's code.

## Quick Reference Commands

```bash
# 1. Sync code to remote
rsync -avz --progress \
  --exclude='.git/' --exclude='.venv/' --exclude='docs/_build/' --exclude='3rdparty' \
  --exclude='.gitignore' --exclude='.gitattributes' \
  "/Users/yuya/Library/CloudStorage/OneDrive-NVIDIACorporation/Documents/projects/Megatron-Hub/" \
  yuya@cw-dfw-cs-001-login-01:/lustre/fsw/portfolios/coreai/users/yuya/Megatron-Hub

# 2. Run specific test
ssh yuya@cw-dfw-cs-001-login-01 "tmux send-keys -t aa 'cd /opt/Megatron-Bridge && uv run python -m pytest tests/unit_tests/models/qwen_vl/modelling_qwen3_vl/test_utils.py -v' Enter"

# 3. Capture test output
ssh yuya@cw-dfw-cs-001-login-01 "sleep 30; tmux capture-pane -t aa -p -S -100"
```
