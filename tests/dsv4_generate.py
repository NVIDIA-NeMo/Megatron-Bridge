"""
DSv4-Flash inference sanity check.
Mirrors test_dsv4_full_import.py for distributed setup, then runs greedy generation.
Usage: run via run_dsv4_inference.sh (srun, 8 ranks, TP=8 EP=1 ETP=8)
"""

import os

import torch
import torch.distributed as dist


MODEL_PATH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/nemo_home/models/deepseek/DeepSeek-V4-Flash"
PROMPT = "What is 1+1? Answer:"
MAX_NEW_TOKENS = 50


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def is_rank0():
    return rank() == 0


def log(msg):
    if is_rank0():
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Phase 1: Distributed init
# ---------------------------------------------------------------------------
log("\n=== Phase 1: Distributed init ===")
master_addr = os.environ.get("MASTER_ADDR", "localhost")
master_port = os.environ.get("MASTER_PORT", "29500")
world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank_id = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
tp_size = int(os.environ.get("TP_SIZE", "8"))
etp_size = int(os.environ.get("ETP_SIZE", "8"))

torch.cuda.set_device(local_rank)
dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://{master_addr}:{master_port}",
    world_size=world_size,
    rank=rank_id,
)
from megatron.core import parallel_state, tensor_parallel


parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=tp_size,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=etp_size,
)
tensor_parallel.model_parallel_cuda_manual_seed(42)
log(f"  TP={tp_size} ETP={etp_size} world_size={world_size}")

# ---------------------------------------------------------------------------
# Phase 2: Provider
# ---------------------------------------------------------------------------
log("\n=== Phase 2: Provider ===")
from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


bridge = AutoBridge.from_hf_pretrained(MODEL_PATH)
hf_pretrained = PreTrainedCausalLM(MODEL_PATH)
provider = bridge._model_bridge.provider_bridge(hf_pretrained)
provider.tensor_model_parallel_size = tp_size
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 1
provider.expert_tensor_parallel_size = etp_size
provider.finalize()
log(f"  model_type={bridge.hf_pretrained.config.model_type}  layers={provider.num_layers}")

# ---------------------------------------------------------------------------
# Phase 3: Model init
# ---------------------------------------------------------------------------
log("\n=== Phase 3: Model init ===")
model = provider.provide_distributed_model(wrap_with_ddp=False)
log(f"  {type(model[0]).__name__}  params={sum(p.numel() for p in model[0].parameters()):,}")

# ---------------------------------------------------------------------------
# Phase 4: Load weights
# ---------------------------------------------------------------------------
log("\n=== Phase 4: Load HF weights (~2.5h) ===")
bridge.load_hf_weights(model)
log("  load_hf_weights() done")

# ---------------------------------------------------------------------------
# Phase 5: Greedy generation
# ---------------------------------------------------------------------------
log("\n=== Phase 5: Greedy generation ===")
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

from megatron.bridge.utils.common_utils import disable_mtp_for_inference, get_last_rank


for m in model:
    m.eval()
    disable_mtp_for_inference(m)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer.encode(PROMPT, return_tensors="pt").cuda()
generated_ids = input_ids.clone()
log(f"  prompt: {PROMPT!r}  tokens={input_ids.shape[1]}")


class _Iter:
    def __init__(self, tokens, pos):
        self.b = {"tokens": tokens, "position_ids": pos}
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return self.b


def _fwd(data_iterator, model, **_):
    b = next(data_iterator)
    return model(input_ids=b["tokens"], position_ids=b["position_ids"], attention_mask=None), lambda x, **k: x


fwd_bwd = get_forward_backward_func()

for step in range(MAX_NEW_TOKENS):
    pos_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device="cuda").unsqueeze(0)
    with torch.no_grad():
        out = fwd_bwd(
            forward_step_func=_fwd,
            data_iterator=_Iter(input_ids, pos_ids),
            model=model,
            num_microbatches=1,
            forward_only=True,
            seq_length=input_ids.shape[1],
            micro_batch_size=1,
            collect_non_loss_data=True,
        )

    if isinstance(out, list) and out:
        out = out[0]

    if parallel_state.is_pipeline_last_stage():
        ws = parallel_state.get_tensor_model_parallel_world_size()
        gathered = [torch.zeros_like(out) for _ in range(ws)]
        dist.all_gather(gathered, out, group=parallel_state.get_tensor_model_parallel_group())
        out = torch.cat(gathered, dim=2)
        next_tok = torch.argmax(out[:, -1], dim=-1, keepdim=True)
        if step < 5:
            logits = out[0, -1]
            top5v, top5i = torch.topk(logits, 5)
            log(f"  step {step}: top5={[(tokenizer.decode([i]), round(v.item(), 2)) for i, v in zip(top5i, top5v)]}")
    else:
        next_tok = torch.ones((1, 1), device="cuda", dtype=torch.long)

    dist.broadcast(next_tok, get_last_rank())
    generated_ids = torch.cat([generated_ids, next_tok], dim=-1)
    input_ids = generated_ids
    if next_tok.item() == tokenizer.eos_token_id:
        break

log("\n=== RESULT ===")
log(f"Prompt   : {PROMPT}")
log(f"Generated: {tokenizer.decode(generated_ids[0].tolist())}")
log("==============")

dist.destroy_process_group()
