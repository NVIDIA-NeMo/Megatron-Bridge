# Adapting Megatron Bridge in Reinforcement Learning Frameworks

Megatron Bridge provides a clean, parallelism-aware path to use 🤗 Hugging Face models with Megatron-Core training and convert back again for inference. This guide shows how to adapt Megatron Bridge into a new RL framework to:

- Convert Hugging Face (HF) checkpoints → Megatron format for scalable training
- Train with Megatron-Core using TP/PP/CP/MoE parallelism, checkpointing, and efficient data paths
- Refit trained weights back to HF for deployment with inference engines (e.g., vLLM), including zero-copy/IPC flows

The examples mirror how NeMo-RL integrates Megatron Bridge:

- [nemo_rl/models/megatron/community_import.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/megatron/community_import.py)
- [nemo_rl/models/policy/megatron_policy_worker.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/policy/megatron_policy_worker.py)

- Local example script in this repo: [examples/rl/rlhf_with_bridge.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/rl/rlhf_with_bridge.py)


## Prerequisites

- A working PyTorch + NCCL GPU stack
- Megatron-LM (MCore) and Megatron-Bridge installed
- A distributed launcher (e.g., `torchrun`, `srun`) for multi-GPU setups
- Access to gated HF repos if needed (export `HF_TOKEN`)

```bash
export HF_TOKEN=<your_hf_token_if_needed>
```


## 1) One-time HF → Megatron checkpoint conversion

Use `AutoBridge` to import an HF model into Megatron format. This writes a Megatron checkpoint directory with a `run_config.yaml` you will reuse during training.

```python
from megatron.bridge import AutoBridge

# Import a model to Megatron checkpoint format (one call)
AutoBridge.import_ckpt(
    hf_model_id="meta-llama/Llama-3.2-1B",
    megatron_path="/path/to/megatron_ckpt/llama32_1b",
)
```

Or, with explicit provider and parallelism settings (similar to [nemo_rl/models/megatron/community_import.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/megatron/community_import.py)):

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=True)

# Configure distributed parallelism used during IMPORT
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 1
provider.expert_tensor_parallel_size = 1
provider.num_layers_in_first_pipeline_stage = 0
provider.num_layers_in_last_pipeline_stage = 0
provider.finalize()

# Create distributed model and save as Megatron checkpoint
megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)
bridge.save_megatron_model(megatron_model, "/path/to/megatron_ckpt")
```

You can also check and try out our multi-GPU conversion example script: [examples/conversion/hf_megatron_roundtrip_multi_gpu.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_megatron_roundtrip_multi_gpu.py)


Notes:
- The import-time parallelism is only for loading/conversion. The saved config is restored to canonical values to avoid validation issues at training time.
- If you are running inside a framework, make sure to clean up any existing distributed state before and after import by destroying or initializing process groups as needed. The `provide_distributed_model` method will initialize a new distributed environment if one is not already set up.


## 2) Build training configuration and initialize Megatron-Core

Translate your RL framework config into Megatron Bridge's `ConfigContainer` for model, optimizer, scheduler, DDP, tokenizer, and checkpoints.

```python
import torch
from megatron.bridge.training.config import (
    ConfigContainer,
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    DistributedDataParallelConfig,
    CheckpointConfig,
    TokenizerConfig,
)
from nemo_rl.models.policy import PolicyConfig  # or your own policy cfg type

# Example: map your RL config to Megatron config
def build_megatron_config(rl_cfg: PolicyConfig, pretrained_ckpt_dir: str) -> ConfigContainer:
    model_cfg = rl_cfg["megatron_cfg"].copy()
    # Precision
    dtype = rl_cfg["precision"]
    model_cfg["bf16"] = dtype == "bfloat16"
    model_cfg["fp16"] = dtype == "float16"

    checkpoint = CheckpointConfig(
        save_interval=100,
        save=rl_cfg["train_ckpt_dir"],
        load=rl_cfg["train_ckpt_dir"],
        pretrained_checkpoint=pretrained_ckpt_dir,
        async_save=False,
        fully_parallel_save=True,
        fully_parallel_load=True,
        load_rng=False,
    )

    ddp = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=rl_cfg["megatron_cfg"]["distributed_data_parallel_config"]["grad_reduce_in_fp32"],
        overlap_grad_reduce=rl_cfg["megatron_cfg"]["distributed_data_parallel_config"]["overlap_grad_reduce"],
        overlap_param_gather=rl_cfg["megatron_cfg"]["distributed_data_parallel_config"]["overlap_param_gather"],
        average_in_collective=rl_cfg["megatron_cfg"]["distributed_data_parallel_config"]["average_in_collective"],
        use_distributed_optimizer=rl_cfg["megatron_cfg"]["optimizer"]["use_distributed_optimizer"],
        data_parallel_sharding_strategy=rl_cfg["megatron_cfg"]["distributed_data_parallel_config"]["data_parallel_sharding_strategy"],
    )

    opt = OptimizerConfig(**rl_cfg["megatron_cfg"]["optimizer"])  # lr, wd, etc.
    sch = SchedulerConfig(**rl_cfg["megatron_cfg"]["scheduler"])  # warmup, decay, etc.

    train = TrainingConfig(
        micro_batch_size=rl_cfg["train_micro_batch_size"],
        global_batch_size=rl_cfg["train_global_batch_size"],
        train_iters=rl_cfg["megatron_cfg"]["train_iters"],
    )

    tokenizer = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=rl_cfg["model_name"],
    )

    return ConfigContainer(
        model=model_cfg,
        checkpoint=checkpoint,
        logger=None,
        train=train,
        optimizer=opt,
        ddp=ddp,
        scheduler=sch,
        dataset=None,
        tokenizer=tokenizer,
    )
```

Initialize Megatron-Core using a helper similar to `setup_megatron_model` from NeMo-RL:

```python
from megatron.bridge.training.initialize import initialize_megatron, set_jit_fusion_options
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.checkpointing import init_checkpointing_context, load_checkpoint
from megatron.bridge.training.state import GlobalState

# Minimal bootstrap
state = GlobalState()
state.cfg = megatron_cfg
initialize_megatron(cfg=megatron_cfg)
set_jit_fusion_options(megatron_cfg.model, megatron_cfg.train.micro_batch_size)

ckpt_ctx = init_checkpointing_context(megatron_cfg.checkpoint)
model_list = get_model(
    megatron_cfg.model,
    megatron_cfg.ddp,
    use_torch_fsdp2=megatron_cfg.dist.use_torch_fsdp2,
    overlap_param_gather_with_optimizer_step=megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
    data_parallel_random_init=megatron_cfg.rng.data_parallel_random_init,
)
optimizer, scheduler = setup_optimizer(
    optimizer_config=megatron_cfg.optimizer,
    scheduler_config=megatron_cfg.scheduler,
    model=model_list,
    use_gloo_process_groups=megatron_cfg.dist.use_gloo_process_groups,
)

# Optional: load pretrained checkpoint
load_checkpoint(
    state,
    model_list,
    optimizer,
    scheduler,
    checkpointing_context=ckpt_ctx,
    skip_load_to_model_and_opt=False,
)

model = model_list[0]
```

Key mappings to handle in your RL config:
- Parallelism: `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `context_parallel_size` (requires sequence packing), and MoE (`expert_*`).
- Precision: `bf16`/`fp16` plus `pipeline_dtype`.
- Activation checkpointing: recompute settings for memory savings.
- FP8 (advanced): be mindful of alignment/padding requirements if enabled.


## 3) Training loop integration (forward/backward/microbatches)

Megatron-Core exposes `get_forward_backward_func()` to run a microbatch loop. Plug in your RL loss function.

```python
from functools import partial
from megatron.core.pipeline_parallel import get_forward_backward_func

model.train()
forward_backward = get_forward_backward_func()

# Your loss function should return (loss_tensor, metrics_dict)
def rl_loss_fn(outputs, batch):
    # Compute logits → loss for your RL objective (e.g., PPO, DPO)
    loss = outputs.sum() * 0.0  # placeholder
    return loss, {"loss": loss.detach()}

# Forward step: prepare inputs; return outputs and a collector that yields loss

def forward_step_fn(data_iterator, model):
    batch = next(data_iterator).to("cuda")
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        packed_seq_params=batch.get("packed_seq_params"),  # if sequence packing
        # multimodal features can be passed as kwargs
    )
    return outputs, (lambda _out: rl_loss_fn(outputs, batch))

losses_reduced = forward_backward(
    forward_step_func=forward_step_fn,
    data_iterator=your_microbatch_iterator,
    model=model,
    num_microbatches=num_microbatches,
    seq_length=sequence_length,
    micro_batch_size=micro_batch_size,
    decoder_seq_length=sequence_length,
    forward_only=False,
    do_not_average_loss=True,
)

# Optimizer/scheduler steps
update_successful, grad_norm, _ = optimizer.step()
scheduler.step(increment=global_batch_size)
```

Sequence packing and context parallelism:
- If `context_parallel_size > 1`, enable sequence packing and build `packed_seq_params` and `cu_seqlens` per microbatch before calling the model.
- With FP8, ensure sequence padding respects hardware-friendly multiples (e.g., lcm(16, 2 × TP × CP)).


## 4) Token logprobs for RL objectives (advantages, DPO, etc.)

For evaluation of token logprobs, run forward-only and reduce TP-sharded logits to per-token logprobs.

```python
import torch
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_rank

@torch.no_grad()
def get_token_logprobs(model, batch):
    model.eval()
    input_ids = batch["input_ids"].to("cuda")
    outputs = model(input_ids=input_ids)

    # Reduce TP logits → local logprobs for targets
    tp_group = get_tensor_model_parallel_group()
    tp_rank = get_tensor_model_parallel_rank()

    # Use a reducer similar to NeMo-RL's `from_parallel_logits_to_logprobs`
    token_logprobs = your_reduce_parallel_logits_to_logprobs(
        outputs,
        target=input_ids,
        vocab_start_index=tp_rank * outputs.shape[-1],
        vocab_end_index=(tp_rank + 1) * outputs.shape[-1],
        tp_group=tp_group,
        inference_only=True,
    )

    # Prepend a zero to keep the same sequence length as the inputs
    token_logprobs = torch.cat([torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1)
    return token_logprobs.cpu()
```

If using sequence packing + context parallelism, switch to the packed variant that uses `packed_seq_params` and `cu_seqlens` for correct alignment.


## 5) Checkpointing (save/load)

Use Megatron-Bridge’s checkpoint helpers. Temporarily disable overlapping param-gather hooks if needed during save.

```python
from megatron.bridge.training.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    init_checkpointing_context,
)

ckpt_ctx = init_checkpointing_context(megatron_cfg.checkpoint)
save_checkpoint(
    state=state,
    model=[model],
    optimizer=optimizer,
    opt_param_scheduler=scheduler,
    num_floating_point_operations_so_far=state.train_state.floating_point_operations_so_far,
    checkpointing_context=ckpt_ctx,
)
```

Tips:
- Prefer fully-parallel save/load at scale (`fully_parallel_save=True`, `fully_parallel_load=True`).


## 6) Refit: Megatron → HF for inference (vLLM, Triton, etc.)

Two common pathways:

### A) Export full HF checkpoint (simplest)

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)
# Load Megatron model from your training checkpoint
megatron_model = bridge.load_megatron_model("/path/to/train_ckpt")

# Iterate over HF weights parameter-by-parameter
for name, weight in bridge.export_hf_weights(megatron_model, cpu=True, show_progress=False):
    # process_or_save(name, weight)
    pass
```

Point your inference engine (e.g., vLLM) to `"/path/to/hf_export"`.

### B) Zero-copy streaming via ZMQ (fast refit, colocated)

Stream tensors from the training side to your inference runtime without writing to disk. The transport is ZMQ peer-to-peer with async send/recv and ping‑pong buffers for overlap; Ray is used only for lightweight coordination. This replaces the earlier ad‑hoc per‑tensor IPC handle passing and aligns with the refactor in [NVIDIA-NeMo/RL#1267](https://github.com/NVIDIA-NeMo/RL/pull/1267).

**Concepts (how the plan and chunking work):**
- **Transport and overlap:** ZMQ P2P streaming with asynchronous send/recv and ping‑pong buffers enables overlap between gathering and applying weights.
- **Conversion tasks (planning):** `bridge.get_conversion_tasks([model])` returns an ordered list of per-parameter conversion tasks that encode how to transform sharded Megatron weights (TP/PP/MoE/CP) back to HF tensors. The worker stores this in `self.refit_conversion_tasks` and advances a cursor (`self.refit_conversion_tasks_current_index`) as chunks are streamed. See `nemo_rl/models/policy/megatron_policy_worker.py` methods `prepare_refit_info()`, `_calculate_refit_param_info()`, and `get_weights_ipc_handles()`.
- **Size estimation across PP ranks:** Parameters are only materialized on their owning PP rank. The worker computes per-parameter byte sizes and then broadcasts those sizes to all PP ranks so the entire pipeline can agree on chunk boundaries. See `broadcast_object_across_pp_ranks()` and `_calculate_refit_param_info()` in `megatron_policy_worker.py`.
- **Memory-aware chunking:** Use your free GPU memory budget (e.g., `NRL_REFIT_BUFFER_MEMORY_RATIO`) to decide how many parameters to include in the next chunk (the set of `keys`). The worker exposes `prepare_weights_for_ipc()` which returns `(param_info, total_available_bytes)` and resets the conversion cursor; then the controller repeatedly selects `keys` whose cumulative byte size ≤ budget and streams them to the consumer over ZMQ.
- **Device routing:** Handles are returned under a `device_uuid` key (NVML UUID of the CUDA device). The inference side should map handles on the same device (or coordinate via your communicator). For collective updates, the worker can also broadcast tensors directly (`broadcast_weights_for_collective`).
- **Parallelism nuances:** With TP/EP, exported HF tensors are reassembled from shards; with CP/sequence packing, shapes/dtypes are already consistent at export time. FP8 or mixed precision can affect size estimates; the worker accounts for dtype scaling when estimating bytes.

```python
import os
import torch
from collections import defaultdict
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)

# 1) Plan: inspect names/shapes/dtypes and estimate memory
refit_param_info_hf = {}
for name, tensor in bridge.export_hf_weights([model], show_progress=False):
    refit_param_info_hf[name] = (tuple(tensor.shape), tensor.dtype)

# 2) Budget for staging buffers (optionally)
from nemo_rl.utils.nvml import get_free_memory_bytes  # or your own NVML wrapper
free_bytes = get_free_memory_bytes(torch.cuda.current_device())
ratio = float(os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.2"))
allowed_bytes = int(free_bytes * ratio)

# 3) Stream chunks over ZMQ
from nemo_rl.utils.nvml import get_device_uuid

# Build conversion tasks once and advance an index as you stream
refit_conversion_tasks = bridge.get_conversion_tasks([model])
refit_tasks_current_index = 0

def stream_next_chunk(keys: list[str]):
    """Yield ZMQ multipart frames for this chunk.
    Frames typically include: (metadata_json_bytes, payload_bytes).
    """
    global refit_tasks_current_index
    conversion_tasks = refit_conversion_tasks[
        refit_tasks_current_index : refit_tasks_current_index + len(keys)
    ]
    refit_tasks_current_index += len(keys)

    device_uuid = get_device_uuid(torch.cuda.current_device())

    # Worker exposes a streaming generator that overlaps gather and send
    for frames in worker.stream_refit_chunks(
        conversion_tasks=conversion_tasks, device_uuid=device_uuid
    ):
        yield frames  # send via zmq_socket.send_multipart(frames)

# Example usage (producer)
for frames in stream_next_chunk(list(refit_param_info_hf.keys())):
    zmq_socket.send_multipart(frames)
```

**Chunking in practice (controller-side selection of keys):**

```python
# param_info like [(name, size_bytes), ...] from prepare_refit_info or prepare_weights_for_ipc
param_info, budget_bytes = worker.prepare_weights_for_ipc()

cursor = 0
while cursor < len(param_info):
    batch_keys = []
    used = 0
    # Greedy pack parameters into this chunk until we run out of budget
    while cursor < len(param_info):
        name, size_bytes = param_info[cursor]
        # size_bytes is already broadcast to all PP ranks; can be int
        if used + int(size_bytes) > budget_bytes and len(batch_keys) > 0:
            break
        batch_keys.append(name)
        used += int(size_bytes)
        cursor += 1

    # Stream this chunk and consume on the inference side
    for frames in worker.stream_refit_chunks(keys=batch_keys):
        zmq_socket.send_multipart(frames)
```

Environment knobs:
- `NRL_REFIT_BUFFER_MEMORY_RATIO` (default `0.2`) — fraction of free GPU memory to plan staging


## 7) Minimal adapter skeleton

Use this skeleton to embed Megatron Bridge into your RL codebase. Fill in the config mapping, microbatching, and loss logic.

```python
import torch
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.checkpointing import init_checkpointing_context, save_checkpoint
from megatron.core.pipeline_parallel import get_forward_backward_func

class MegatronBridgeAdapter:
    def __init__(self, rl_cfg, pretrained_ckpt_dir: str):
        self.rl_cfg = rl_cfg
        self.megatron_cfg = build_megatron_config(rl_cfg, pretrained_ckpt_dir)
        self.state = GlobalState(); self.state.cfg = self.megatron_cfg
        self.ckpt_ctx = init_checkpointing_context(self.megatron_cfg.checkpoint)
        self._init_model()

    def _init_model(self):
        from megatron.bridge.training.initialize import initialize_megatron, set_jit_fusion_options
        from megatron.bridge.models.model_provider import get_model
        from megatron.bridge.training.optim import setup_optimizer
        initialize_megatron(cfg=self.megatron_cfg)
        set_jit_fusion_options(self.megatron_cfg.model, self.megatron_cfg.train.micro_batch_size)
        self.model_list = get_model(self.megatron_cfg.model, self.megatron_cfg.ddp,
                                    use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                                    overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step)
        self.model = self.model_list[0]
        self.optimizer, self.scheduler = setup_optimizer(self.megatron_cfg.optimizer, self.megatron_cfg.scheduler, self.model_list,
                                                         use_gloo_process_groups=self.megatron_cfg.dist.use_gloo_process_groups)

    @torch.no_grad()
    def get_logprobs(self, batch):
        self.model.eval()
        # Implement reduction from parallel logits to token logprobs
        ...

    def train_step(self, mb_iter, num_microbatches, seq_len, mbs, loss_fn):
        self.model.train()
        fb = get_forward_backward_func()
        def fwd(data_it, model):
            batch = next(data_it).to("cuda")
            out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
            return out, (lambda _o: loss_fn(out, batch))
        fb(forward_step_func=fwd, data_iterator=mb_iter, model=self.model, num_microbatches=num_microbatches,
           seq_length=seq_len, micro_batch_size=mbs, decoder_seq_length=seq_len, forward_only=False, do_not_average_loss=True)
        ok, _, _ = self.optimizer.step(); self.scheduler.step(increment=self.rl_cfg["train_global_batch_size"])
        return ok

    def save_ckpt(self, path: str):
        save_checkpoint(self.state, [self.model], self.optimizer, self.scheduler,
                        num_floating_point_operations_so_far=self.state.train_state.floating_point_operations_so_far,
                        checkpointing_context=self.ckpt_ctx)

    def export_hf(self, out_dir: str):
        from megatron.bridge import AutoBridge
        bridge = AutoBridge.from_hf_pretrained(self.rl_cfg["model_name"], trust_remote_code=True)
        # Stream weights directly using AutoBridge.export_hf_weights; consume (save/IPC) as needed
        for name, tensor in bridge.export_hf_weights([self.model], show_progress=False):
            # process_or_save(name, tensor, out_dir)  # implement your consumer (e.g., safetensors or IPC)
            pass
        # Optionally, to persist safetensors shards without config/tokenizer:
        # bridge.save_hf_weights([self.model], out_dir, show_progress=False)
```


## 8) Best practices and pitfalls

- Parallelism
  - If `context_parallel_size > 1`, enable sequence packing. Use packed logprob reducers.
  - With FP8, pad to hardware-friendly multiples (e.g., lcm(16, 2 × TP × CP)).
- Offloading
  - TBA
- MoE router stability
  - Consider freezing router weights and disabling router load balancing to reduce training instability (see `freeze_moe_router`, `moe_router_bias_update_rate=0.0`).
- Hooks
  - Temporarily disable DDP forward pre-hooks when swapping weights or saving to avoid conflicts with overlapping param gather.
- Checkpointing
  - Use fully-parallel save/load at scale. Avoid async save unless validated in your environment.


## See also

- [Bridge with 🤗 Hugging Face](./bridge-guide.md) for HF↔Megatron conversion overview
- [nemo_rl/models/megatron/community_import.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/megatron/community_import.py) for import/export helpers
- [nemo_rl/models/policy/megatron_policy_worker.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/models/policy/megatron_policy_worker.py) for end-to-end RL integration (training, logprobs, generation, refit)
