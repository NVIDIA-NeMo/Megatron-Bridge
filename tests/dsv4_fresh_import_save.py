"""Fresh import DSv4 with dev+4518 MCore + fused mHC + HC head weights.
Saves checkpoint + runs forward with last hidden state capture.
"""

import os
import time

import torch
import torch.distributed as dist


WKDIR = "/lustre/fsw/portfolios/coreai/users/weijiac"
MODEL_PATH = f"{WKDIR}/models/deepseek-ai/DeepSeek-V4-Flash"
CKPT_DIR = f"{WKDIR}/dsv4_flash_megatron_ckpt_05062026"
SAVE_DIR = f"{WKDIR}/dsv4_cosine_analysis"
INPUT_IDS = [0, 128803, 3085, 344, 223, 19, 13, 19, 33, 128804, 128822]

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "4"))


def log(msg):
    if rank == 0:
        print(msg, flush=True)


torch.cuda.set_device(local_rank)
dist.init_process_group(
    "nccl",
    init_method="tcp://" + os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"],
    world_size=world_size,
    rank=rank,
)
from megatron.core import parallel_state, tensor_parallel


parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=4,
)
tensor_parallel.model_parallel_cuda_manual_seed(42)

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.utils.common_utils import disable_mtp_for_inference


log("=== Fresh import with dev+4518 MCore + fused mHC ===")
t0 = time.time()
bridge = AutoBridge.from_hf_pretrained(MODEL_PATH)
hf = PreTrainedCausalLM(MODEL_PATH)
prov = bridge._model_bridge.provider_bridge(hf)
prov.tensor_model_parallel_size = 1
prov.pipeline_model_parallel_size = 1
prov.expert_model_parallel_size = 1
prov.expert_tensor_parallel_size = 4
prov.finalize()
log(f"  mscale={prov.mscale}, mscale_all_dim={prov.mscale_all_dim}")
log(f"  use_fused_mhc={prov.use_fused_mhc}")
log(f"  apply_rope_fusion={prov.apply_rope_fusion}")

model = prov.provide_distributed_model(wrap_with_ddp=False)
log(f"  Model created in {time.time() - t0:.1f}s")

log("Loading HF weights (fresh import with MXFP4 dequant)...")
t0 = time.time()
bridge.load_hf_weights(model)
dist.barrier()
log(f"  Weights loaded in {time.time() - t0:.1f}s")

# Check HC head weights are loaded (not random)
mg = model[0]
if hasattr(mg, "module"):
    mg = mg.module
decoder = mg.decoder
if hasattr(decoder, "hc_head_fn"):
    hc_fn = decoder.hc_head_fn.data
    log(f"  hc_head_fn: shape={list(hc_fn.shape)}, norm={hc_fn.float().norm():.4f}, mean={hc_fn.float().mean():.6f}")
    log(f"  hc_head_base: norm={decoder.hc_head_base.data.float().norm():.4f}")
    log(f"  hc_head_scale: {decoder.hc_head_scale.data.tolist()}")
else:
    log("  WARNING: decoder has no hc_head_fn!")

for m in model:
    m.eval()
    disable_mtp_for_inference(m)

# Save checkpoint
log(f"\nSaving checkpoint to {CKPT_DIR}...")
os.makedirs(CKPT_DIR, exist_ok=True)
t0 = time.time()
torch.save(model[0].state_dict(), f"{CKPT_DIR}/rank_{rank}.pt")
dist.barrier()
log(f"  Saved in {time.time() - t0:.1f}s")

# Hook final_layernorm for last hidden state
captured = {}
if decoder.final_layernorm is not None:

    def ln_hook(mod, inp, out):
        if rank == 0:
            t = out if isinstance(out, torch.Tensor) else out[0]
            captured["last_hidden_state"] = t.detach().float().cpu()
            if isinstance(inp, tuple) and len(inp) > 0:
                captured["post_contraction"] = inp[0].detach().float().cpu()

    decoder.final_layernorm.register_forward_hook(ln_hook)

# Also hook all layer outputs
layer_cos = {}
for i, layer in enumerate(decoder.layers):

    def make_hook(idx):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor) and rank == 0:
                layer_cos[idx] = t.detach().float().cpu()

        return fn

    layer.register_forward_hook(make_hook(i))

# Forward
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func


ids = torch.tensor([INPUT_IDS], dtype=torch.long, device="cuda")
pos = torch.arange(len(INPUT_IDS), device="cuda").unsqueeze(0)


class It:
    def __init__(self):
        self._d = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._d:
            raise StopIteration
        self._d = True
        return {"tokens": ids, "position_ids": pos}


def fwd(di, mdl, **_):
    b = next(di)
    return mdl(input_ids=b["tokens"], position_ids=b["position_ids"], attention_mask=None), lambda x, **k: x


log("\nForward pass...")
with torch.no_grad():
    out = get_forward_backward_func()(
        forward_step_func=fwd,
        data_iterator=It(),
        model=model,
        num_microbatches=1,
        forward_only=True,
        seq_length=len(INPUT_IDS),
        micro_batch_size=1,
        collect_non_loss_data=True,
    )

if rank == 0:
    if isinstance(out, list) and out:
        captured["logits"] = out[0].detach().float().cpu()

    # Save Megatron states
    captured["layer_outputs"] = layer_cos
    torch.save(captured, f"{SAVE_DIR}/megatron_last_hidden_05062026.pt")
    log(f"\nSaved {len(captured)} items to megatron_last_hidden_05062026.pt")

    if "last_hidden_state" in captured:
        log(f"  last_hidden_state: {list(captured['last_hidden_state'].shape)}")
    if "logits" in captured:
        logits = captured["logits"].squeeze()
        if logits.dim() == 2:
            last = logits[-1]
        else:
            last = logits
        top5 = last.topk(5)
        log(f"  Logit top5: {top5.indices.tolist()}")

    # Quick compare with existing official
    import torch.nn.functional as F

    off_path = f"{SAVE_DIR}/official_last_hidden.pt"
    if os.path.exists(off_path):
        off = torch.load(off_path, map_location="cpu", weights_only=False)
        for key in ["post_contraction", "last_hidden_state"]:
            if key in off and key in captured:
                o = off[key].flatten().float()
                m = captured[key].flatten().float()
                v = min(o.shape[0], m.shape[0])
                cos = F.cosine_similarity(o[:v].unsqueeze(0), m[:v].unsqueeze(0)).item()
                log(f"  {key}: cos={cos:.6f}")
        if "logits" in off and "logits" in captured:
            o_l = off["logits"].squeeze().float()
            m_l = captured["logits"].squeeze().float()
            if m_l.dim() == 2:
                m_l = m_l[-1]
            if o_l.dim() == 2:
                o_l = o_l[-1]
            v = min(o_l.shape[0], m_l.shape[0])
            cos = F.cosine_similarity(o_l[:v].unsqueeze(0), m_l[:v].unsqueeze(0)).item()
            log(f"  Last-token logit cos: {cos:.6f}")

log("\nDone")
dist.destroy_process_group()
