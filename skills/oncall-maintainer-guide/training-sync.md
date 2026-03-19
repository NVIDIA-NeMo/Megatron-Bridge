# Megatron-Training Sync

Weekly sync check (Wednesday) to downstream new Megatron-LM changes into
Megatron Bridge.

---

## What Gets Synced

Bridge mirrors a subset of Megatron-LM's `megatron/training/` module. When
upstream changes those files, Bridge may need corresponding updates.

### File Mapping

| MCore file (`megatron/training/`) | Bridge file (`src/megatron/bridge/training/`) |
|---|---|
| `arguments.py` | `config.py` |
| `training.py` | `train.py`, `pretrain.py`, `setup.py` |
| `checkpointing.py` | `checkpointing.py` |
| `initialize.py` | `initialize.py` |
| `global_vars.py` | `state.py` |

---

## Procedure

### Step 1: Identify new commits

Compare the pinned submodule commit against upstream `main` to find all commits
touching `megatron/training/`.

### Step 2: Classify each commit

#### SYNC REQUIRED

The change affects training behavior that Bridge replicates:

- New training arg that Bridge should expose in its config dataclass
- Bug fix in training loop logic that Bridge has copied/adapted
- Changed function signature that Bridge calls directly
- New required config field with no default

#### WATCH ON BUMP

The change is in a mapped file but Bridge may get it for free when the submodule
is bumped:

- New field added to an MCore dataclass that Bridge inherits from
- Validation logic in `arguments.py` that Bridge doesn't use
- Changes to code paths Bridge doesn't call

#### NO SYNC NEEDED

The change does not affect Bridge:

- Inference-only args
- Code in files Bridge doesn't mirror
- Cosmetic changes in code Bridge doesn't use
- RL-specific args

### Step 3: Verify Bridge coverage

For any commit classified as SYNC REQUIRED, check whether Bridge already has the
change. If so, reclassify as NO SYNC NEEDED.

---

## Bridge Conventions for Sync Changes

1. **Bridge uses dataclass configs**, not argparse. New MLM args become fields
   on the appropriate `@dataclass` in `config.py`.

2. **Bridge import paths** start with `megatron.bridge.training`, not
   `megatron.training`.

3. **Bridge wraps MCore base classes** — new fields in MCore base classes are
   inherited automatically.

4. **Bridge has its own validation** in `ConfigContainer.validate()`. MCore
   `validate_args()` changes may or may not need mirroring.

5. **Bridge does not use MCore's `get_model()`** from `training.py`. It has its
   own model setup in `setup.py`.

6. **PR convention** for sync changes:
   ```
   [sync][training] feat: mirror <description> from mcore
   ```

---

## Common Breakage Patterns

| MCore change | Bridge fix location |
|---|---|
| Renamed argument in `arguments.py` | `config.py` — rename field |
| New argument in `arguments.py` | `config.py` — add field with default |
| Changed function signature in `training.py` | `train.py` or `pretrain.py` |
| New required config field (no default) | Add default in Bridge's config dataclass |
| Removed/moved module | Update imports in Bridge |
| Changed validation logic | Check if Bridge's `validate()` needs the same check |
