"""Compare last hidden state (post HC contraction + layernorm, pre lm_head) between official and MCore."""

import json
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F


MODE = os.environ.get("MODE", "official")
WKDIR = "/lustre/fsw/portfolios/coreai/users/weijiac"
SAVE_DIR = f"{WKDIR}/dsv4_cosine_analysis"
MODEL_PATH = f"{WKDIR}/models/deepseek-ai/DeepSeek-V4-Flash"
INPUT_IDS = [0, 128803, 3085, 344, 223, 19, 13, 19, 33, 128804, 128822]

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "4"))


def log(msg):
    if rank == 0:
        print(msg, flush=True)


os.makedirs(SAVE_DIR, exist_ok=True)
torch.cuda.set_device(local_rank)

if MODE == "official":
    log("=== Official: capture last hidden state ===")
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.set_default_dtype(torch.bfloat16)
    sys.path.insert(0, f"{WKDIR}/nemo_workspace/dsv4_official_inference")
    with open(f"{WKDIR}/config_bf16.json") as f:
        args_dict = json.load(f)
    from model import ModelArgs, Transformer

    args = ModelArgs(**args_dict)
    args.max_batch_size = 1
    args.max_seq_len = 64
    with torch.device("cuda"):
        model = Transformer(args)
    from safetensors.torch import load_model

    load_model(model, f"{WKDIR}/dsv4_flash_converted_bf16_mp4/model{rank}-mp{world_size}.safetensors", strict=False)
    torch.set_default_device("cuda")

    # Hook ParallelHead to capture post-contraction + post-norm hidden state
    captured = {}
    _orig_head_forward = model.head.forward.__func__

    def _capture_head_forward(self, x, hc_fn, hc_scale, hc_base, norm):
        x_contracted = self.hc_head(x, hc_fn, hc_scale, hc_base)  # [b,s,d]
        x_normed = norm(x_contracted)  # [b,s,d]
        if rank == 0:
            captured["post_contraction"] = x_contracted.detach().float().cpu()
            captured["last_hidden_state"] = x_normed.detach().float().cpu()
            log(f"  post_contraction: {list(x_contracted.shape)}")
            log(f"  last_hidden_state: {list(x_normed.shape)}")
        logits = self.get_logits(x_normed)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits

    model.head.forward = lambda *a, **kw: _capture_head_forward(model.head, *a, **kw)

    tokens = torch.tensor([INPUT_IDS], dtype=torch.long, device="cuda")
    log("Forward...")
    with torch.no_grad():
        logits = model(tokens, start_pos=0)
    if rank == 0:
        captured["logits"] = logits.detach().float().cpu()
        torch.save(captured, f"{SAVE_DIR}/official_last_hidden.pt")
        log(f"Saved. Logits shape: {list(logits.shape)}")
        log(f"Last hidden shape: {list(captured['last_hidden_state'].shape)}")
        top5 = logits[0].topk(5)
        log(f"Top5: {top5.indices.tolist()}")

elif MODE == "megatron":
    log("=== Megatron: capture last hidden state ===")
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

    bridge = AutoBridge.from_hf_pretrained(MODEL_PATH)
    hf = PreTrainedCausalLM(MODEL_PATH)
    prov = bridge._model_bridge.provider_bridge(hf)
    prov.tensor_model_parallel_size = 1
    prov.pipeline_model_parallel_size = 1
    prov.expert_model_parallel_size = 1
    prov.expert_tensor_parallel_size = 4
    prov.finalize()
    mg_model = prov.provide_distributed_model(wrap_with_ddp=False)
    sd = torch.load(f"{WKDIR}/dsv4_flash_megatron_ckpt_tp1/rank_{rank}.pt", map_location="cpu", weights_only=False)
    mg_model[0].load_state_dict(sd, strict=False)
    del sd
    dist.barrier()
    for m in mg_model:
        m.eval()
        disable_mtp_for_inference(m)

    # Hook the decoder's final_layernorm output
    mg = mg_model[0]
    if hasattr(mg, "module"):
        mg = mg.module
    decoder = mg.decoder

    captured = {}
    if decoder.final_layernorm is not None:

        def ln_hook(mod, inp, out):
            if rank == 0:
                t = out if isinstance(out, torch.Tensor) else out[0]
                captured["last_hidden_state"] = t.detach().float().cpu()
                log(f"  last_hidden_state (post final_layernorm): {list(t.shape)}")
                # Also capture pre-layernorm (= post contraction)
                if isinstance(inp, tuple) and len(inp) > 0:
                    captured["post_contraction"] = inp[0].detach().float().cpu()
                    log(f"  post_contraction: {list(inp[0].shape)}")

        decoder.final_layernorm.register_forward_hook(ln_hook)

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

    log("Forward...")
    with torch.no_grad():
        out = get_forward_backward_func()(
            forward_step_func=fwd,
            data_iterator=It(),
            model=mg_model,
            num_microbatches=1,
            forward_only=True,
            seq_length=len(INPUT_IDS),
            micro_batch_size=1,
            collect_non_loss_data=True,
        )
    if rank == 0:
        if isinstance(out, list) and out:
            captured["logits"] = out[0].detach().float().cpu()
        torch.save(captured, f"{SAVE_DIR}/megatron_last_hidden.pt")
        log(f"Saved. Logits shape: {list(captured.get('logits', torch.tensor([])).shape)}")

elif MODE == "compare":
    log("=== Comparing last hidden states ===")
    off = torch.load(f"{SAVE_DIR}/official_last_hidden.pt", map_location="cpu", weights_only=False)
    mg = torch.load(f"{SAVE_DIR}/megatron_last_hidden.pt", map_location="cpu", weights_only=False)

    for key in ["post_contraction", "last_hidden_state"]:
        if key in off and key in mg:
            o = off[key].flatten().float()
            m = mg[key].flatten().float()
            v = min(o.shape[0], m.shape[0])
            cos = F.cosine_similarity(o[:v].unsqueeze(0), m[:v].unsqueeze(0)).item()
            log(f"{key:25s}: cos={cos:.6f}  off_shape={list(off[key].shape)}  mg_shape={list(mg[key].shape)}")

            # Per-position cosine sim
            o_t = off[key].squeeze()  # remove batch dim
            m_t = mg[key].squeeze()
            if o_t.dim() == 2 and m_t.dim() == 2:
                seq = min(o_t.shape[0], m_t.shape[0])
                log("  Per-position:")
                for p in range(seq):
                    pcos = F.cosine_similarity(o_t[p].unsqueeze(0), m_t[p].unsqueeze(0)).item()
                    log(f"    pos {p:2d}: cos={pcos:.6f}")

    # Logit comparison (full vocab now since official gathers all ranks)
    if "logits" in off and "logits" in mg:
        o_logits = off["logits"].squeeze().float()  # [vocab] (last token from official)
        m_logits = mg["logits"].squeeze().float()  # [seq, vocab] from MCore
        if m_logits.dim() == 2:
            m_last = m_logits[-1]  # last token
        else:
            m_last = m_logits
        if o_logits.dim() == 1:
            o_last = o_logits
        else:
            o_last = o_logits[-1]
        v = min(o_last.shape[0], m_last.shape[0])
        cos = F.cosine_similarity(o_last[:v].unsqueeze(0), m_last[:v].unsqueeze(0)).item()
        log(f"\nLast-token logit cosine sim (full vocab): {cos:.6f}")
        log(f"  Official shape: {list(o_logits.shape)}, MCore shape: {list(m_logits.shape)}")
        o_top5 = o_last[:v].topk(5)
        m_top5 = m_last[:v].topk(5)
        log(f"  Official top5: {o_top5.indices.tolist()}")
        log(f"  MCore top5:    {m_top5.indices.tolist()}")
        overlap = len(set(o_top5.indices.tolist()) & set(m_top5.indices.tolist()))
        log(f"  Top-5 overlap: {overlap}/5")

log("\nDone")
if dist.is_initialized():
    dist.destroy_process_group()
