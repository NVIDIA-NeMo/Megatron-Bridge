# Exponential Moving Average (EMA)

Megatron Bridge provides Exponential Moving Average (EMA) support through a callback-based integration.

EMA maintains a shadow copy of the trainable model parameters and updates that shadow copy after each successful training step using a fixed decay factor. This is useful when you want a smoothed parameter trajectory during training without changing the main optimization path.

In Megatron Bridge, EMA is implemented as a training callback and uses checkpoint lifecycle callback hooks for persistence and restore.

## How EMA Works

The EMA callback:

1. Initializes a float32 shadow copy of trainable parameters at train start
2. Updates the shadow parameters after each non-skipped training step
3. Persists EMA state as a per-rank checkpoint sidecar
4. Restores EMA state on checkpoint resume when the sidecar is present

EMA state is stored as a per-rank sidecar under each checkpoint directory.

## Limitations

- EMA sidecar persistence currently supports persistent checkpoints only
- `async_save=True` is not yet supported for EMA sidecar persistence
- Local checkpoints are not yet supported
- The current implementation uses a fixed-decay EMA callback design

## Usage

### Enable EMA

```python
from megatron.bridge.training.ema import EMACallback
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.recipes.qwen import qwen25_500m_pretrain_config

config = qwen25_500m_pretrain_config()
config.checkpoint.async_save = False

ema_callback = EMACallback(
    decay=0.999,
    start_step=0,
    store_on_cpu=False,
    log_interval=10,
)

pretrain(config, forward_step, callbacks=[ema_callback])
```

### EMA Parameters

- `decay`: EMA decay factor.
- `start_step`: Step at which EMA updates begin.
- `store_on_cpu`: Whether to store EMA state on CPU.
- `log_interval`: Interval for EMA logging.

### Checkpointing

EMA state is saved as a per-rank sidecar under the checkpoint directory.

On resume:

- If the EMA sidecar exists, the callback restores it.
- If the EMA sidecar is missing, EMA is re-initialized on train start.

### Design Notes

EMA is implemented through the callback system rather than through framework-owned checkpoint logic. This keeps EMA isolated from the default training path when the feature is not enabled.

EMA persistence and restore are handled through checkpoint lifecycle callback hooks.