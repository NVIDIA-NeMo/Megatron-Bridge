# Callbacks

Megatron Bridge provides a lightweight callback system for injecting custom logic into the training and evaluation loop without modifying framework code. This is ideal for propietary integrations or custom logging and metrics tracking.

## Quick Start

### Class-Based Callbacks

Subclass {py:class}`bridge.training.callbacks.Callback` and override event methods:

```python
import time

from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.recipes.qwen import qwen25_500m_pretrain_config

class MyCallback(Callback):
    def on_train_start(self, context):
        context.user_state['start_time'] = time.time()
        print(f"Training started at step {context.state.train_state.step}")

    def on_train_step_end(self, context):
        if context.loss_dict:
            print(f"Step {context.state.train_state.step}: loss={context.loss_dict}")

    def on_train_end(self, context):
        elapsed = time.time() - context.user_state['start_time']
        print(f"Training completed in {elapsed:.2f}s")

# Create a config that fits on a single GPU
config = qwen25_500m_pretrain_config()

# Pass callbacks to pretrain
pretrain(config, forward_step, callbacks=[MyCallback()])
```

### Functional Callbacks

Register functions directly with {py:class}`bridge.training.callbacks.CallbackManager`:

```python
from megatron.bridge.training.callbacks import CallbackManager
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.recipes.qwen import qwen25_500m_pretrain_config

def log_step(context):
    step = context.state.train_state.step
    if context.loss_dict:
        print(f"Step {step}: {context.loss_dict}")

callback_manager = CallbackManager()
callback_manager.register("on_train_step_end", log_step)

# Create a config that fits on a single GPU
config = qwen25_500m_pretrain_config()

pretrain(config, forward_step, callbacks=callback_manager)
```

### Mixing Both Patterns

Both registration patterns can be combined:

```python
from megatron.bridge.training.callbacks import CallbackManager
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.recipes.qwen import qwen25_500m_pretrain_config

manager = CallbackManager()
manager.add(MyCallback())
manager.add([TimingCallback(), MetricsCallback()])
manager.register("on_eval_end", lambda ctx: print("Evaluation complete!"))

# Create a config that fits on a single GPU
config = qwen25_500m_pretrain_config()

pretrain(config, forward_step, callbacks=manager)
```

## Available Events

### Training Events

| Event | When Fired | Available Context Fields |
|-------|------------|-------------------------|
| `on_train_start` | After `model.train()`, before training loop | `state`, `model`, `user_state`, `optimizer`, `scheduler` |
| `on_train_step_start` | Before each training step | `state`, `model`, `user_state`, `optimizer`, `scheduler` |
| `on_train_step_end` | After each training step | `state`, `model`, `user_state`, `optimizer`, `scheduler`, `loss_dict`, `grad_norm`, `skipped_iter` |
| `on_train_end` | After training loop completes | `state`, `model`, `user_state`, `optimizer`, `scheduler` |

### Validation Events

| Event | When Fired | Available Context Fields |
|-------|------------|-------------------------|
| `on_eval_start` | After `model.eval()`, before validation loop | `state`, `model`, `user_state` |
| `on_eval_step_start` | Before each validation step | `state`, `model`, `user_state` |
| `on_eval_step_end` | After each validation step | `state`, `model`, `user_state` |
| `on_eval_end` | After validation completes | `state`, `model`, `user_state`, `total_loss_dict` |

### Test Events

| Event | When Fired | Available Context Fields |
|-------|------------|-------------------------|
| `on_test_start` | After `model.eval()`, before test loop | `state`, `model`, `user_state` |
| `on_test_step_start` | Before each test step | `state`, `model`, `user_state` |
| `on_test_step_end` | After each test step | `state`, `model`, `user_state` |
| `on_test_end` | After test completes | `state`, `model`, `user_state`, `total_loss_dict` |

## CallbackContext

The {py:class}`bridge.training.callbacks.CallbackContext` provides access to framework state:

### Always Available

- **`state`**: {py:class}`bridge.training.state.GlobalState` - Contains config, train_state, timers, and loggers
- **`model`**: List of model chunks
- **`user_state`**: Mutable dict for storing data across callback invocations

### Training Events Only

- **`optimizer`**: The optimizer instance
- **`scheduler`**: Learning rate scheduler

### Event-Specific Fields

- **`loss_dict`** (`on_train_step_end`): Dictionary of reduced losses from the training step
- **`grad_norm`** (`on_train_step_end`): Gradient norm (if computed)
- **`skipped_iter`** (`on_train_step_end`): Whether the iteration was skipped
- **`total_loss_dict`** (`on_eval_end`, `on_test_end`): Aggregated evaluation/test losses

## User State

The `CallbackManager` owns a `user_state` dictionary that persists across all callback invocations during a training run. Use it to share data between callbacks or accumulate metrics:

```python
class StepCounterCallback(Callback):
    def on_train_start(self, context):
        context.user_state['callback_step_count'] = 0

    def on_train_step_end(self, context):
        context.user_state['callback_step_count'] += 1

    def on_train_end(self, context):
        print(f"Callback saw {context.user_state['callback_step_count']} steps")
```

## Distributed Training

Callbacks fire on **all ranks** without framework-level synchronization. If your callback should only run on specific ranks, add guards:

```python
import torch.distributed as dist

class RankZeroCallback(Callback):
    def on_train_step_end(self, context):
        if dist.get_rank() == 0:
            print(f"Step {context.state.train_state.step} complete")
```

## Exception Handling

Exceptions from callbacks propagate to the caller. The framework does not catch or handle callback exceptions. If your callback might fail, wrap it in a try-except:

```python
def safe_callback(context):
    try:
        # Your logic here
        external_service.log(context.loss_dict)
    except Exception as e:
        print(f"Callback failed: {e}")
        # Don't re-raise to avoid stopping training
```

## Execution Order

Callbacks fire in registration order:

1. Callbacks added via `add()` fire in the order they were added
2. Callbacks registered via `register()` fire in the order they were registered
3. If both methods are used, the order depends on when each was called

## Introspection

Query registered callbacks:

```python
manager = CallbackManager()
manager.register("on_train_start", my_fn)

# Check if any callbacks exist for an event
if manager.has_callbacks("on_train_start"):
    print("Callbacks registered for on_train_start")

# List all callbacks for an event
callbacks = manager.list_callbacks("on_train_start")
print(f"Found {len(callbacks)} callbacks")

# Get all valid event names
print(manager.events)  # frozenset of valid event names
```

## Design Principles

The callback system follows these principles:

1. **First-Party Isolation**: Framework code never uses callbacks for its own logic. Callbacks are strictly for third-party extensions.

2. **Zero Overhead**: When no callbacks are registered, there is zero performance overhead.

3. **Safety**: Callbacks receive framework state but modifying it is at the user's own risk. The framework makes no guarantees about the effects of modifications.

## Examples

### Proprietary Metrics

```python
class ProprietaryMetricsCallback(Callback):
    """Send metrics to internal monitoring system."""

    def __init__(self, endpoint: str):
        self.client = InternalMetricsClient(endpoint)

    def on_train_step_end(self, context):
        if context.loss_dict:
            self.client.send({
                "step": context.state.train_state.step,
                "loss": context.loss_dict.get("lm loss"),
                "grad_norm": context.grad_norm,
                "cluster_id": os.environ.get("CLUSTER_ID"),
            })
```

## API Reference

- {py:class}`bridge.training.callbacks.Callback`
- {py:class}`bridge.training.callbacks.CallbackContext`
- {py:class}`bridge.training.callbacks.CallbackManager`
- {py:func}`bridge.training.callbacks.normalize_callbacks`
- {py:func}`bridge.training.callbacks.should_fire`
