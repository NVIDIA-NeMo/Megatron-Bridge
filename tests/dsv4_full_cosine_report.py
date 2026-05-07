"""Full cosine similarity report: per-layer + sub-layer breakdown.
Uses saved official_states.pt + loads new checkpoint for Megatron.
MODE=megatron: capture all layer outputs + sub-layer for selected layers
MODE=compare: compare against official
"""

import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


MODE = os.environ.get("MODE", "megatron")
WKDIR = "/lustre/fsw/portfolios/coreai/users/weijiac"
SAVE_DIR = f"{WKDIR}/dsv4_cosine_analysis"
MODEL_PATH = f"{WKDIR}/models/deepseek-ai/DeepSeek-V4-Flash"
CKPT_DIR = f"{WKDIR}/dsv4_flash_megatron_ckpt_05062026"
INPUT_IDS = [0, 128803, 3085, 344, 223, 19, 13, 19, 33, 128804, 128822]
SUBLAYER_LAYERS = {0, 2, 10, 25, 42}

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "4"))


def log(msg):
    if rank == 0:
        print(msg, flush=True)


os.makedirs(SAVE_DIR, exist_ok=True)
torch.cuda.set_device(local_rank)

if MODE == "megatron":
    log("=== Megatron: full cosine report (new ckpt, fused mHC) ===")
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
    model = prov.provide_distributed_model(wrap_with_ddp=False)
    log("Loading checkpoint...")
    sd = torch.load(f"{CKPT_DIR}/rank_{rank}.pt", map_location="cpu", weights_only=False)
    model[0].load_state_dict(sd, strict=False)
    del sd
    dist.barrier()
    for m in model:
        m.eval()
        disable_mtp_for_inference(m)

    mg = model[0]
    if hasattr(mg, "module"):
        mg = mg.module
    decoder = mg.decoder

    saved = {}

    # Hook all layer outputs
    for i, layer in enumerate(decoder.layers):

        def make_hook(idx):
            def fn(mod, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor) and rank == 0:
                    saved[f"layer_{idx}_output"] = t.detach().float().cpu()

            return fn

        layer.register_forward_hook(make_hook(i))

    # Sub-layer hooks for selected layers
    for i in SUBLAYER_LAYERS:
        if i >= len(decoder.layers):
            continue
        layer = decoder.layers[i]
        for name, mod in layer.named_children():
            if name in (
                "self_attention",
                "mlp",
                "input_layernorm",
                "pre_mlp_layernorm",
                "self_attention_hyper_connection",
                "mlp_hyper_connection",
            ):

                def make_sub(idx, n):
                    def fn(mod, inp, out):
                        t = out[0] if isinstance(out, (tuple, list)) else out
                        if isinstance(t, torch.Tensor) and rank == 0:
                            saved[f"layer_{idx}_{n}_output"] = t.detach().float().cpu()
                        if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor) and rank == 0:
                            saved[f"layer_{idx}_{n}_input"] = inp[0].detach().float().cpu()

                    return fn

                mod.register_forward_hook(make_sub(i, name))

    # Hook final_layernorm
    if decoder.final_layernorm is not None:

        def ln_hook(mod, inp, out):
            if rank == 0:
                t = out if isinstance(out, torch.Tensor) else out[0]
                saved["last_hidden_state"] = t.detach().float().cpu()
                if isinstance(inp, tuple) and len(inp) > 0:
                    saved["post_contraction"] = inp[0].detach().float().cpu()

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
            model=model,
            num_microbatches=1,
            forward_only=True,
            seq_length=len(INPUT_IDS),
            micro_batch_size=1,
            collect_non_loss_data=True,
        )
    if rank == 0:
        if isinstance(out, list) and out:
            saved["logits"] = out[0].detach().float().cpu()
        torch.save(saved, f"{SAVE_DIR}/megatron_full_report_05062026.pt")
        log(f"Saved {len(saved)} tensors")

elif MODE == "compare":
    log("=== Full cosine similarity report ===")
    off = torch.load(f"{SAVE_DIR}/official_states.pt", map_location="cpu", weights_only=False)
    mg = torch.load(f"{SAVE_DIR}/megatron_full_report_05062026.pt", map_location="cpu", weights_only=False)

    with open(f"{WKDIR}/models/deepseek-ai/DeepSeek-V4-Flash/config.json") as f:
        cfg = json.load(f)
    cr = cfg.get("compress_ratios", cfg.get("compress_rates", []))

    def cosine(a, b):
        a_f, b_f = a.flatten().float(), b.flatten().float()
        v = min(a_f.shape[0], b_f.shape[0])
        return F.cosine_similarity(a_f[:v].unsqueeze(0), b_f[:v].unsqueeze(0)).item()

    # Per-layer
    log(f"\n{'Lyr':>4} {'CR':>4} {'CosSim':>10} {'Status'}")
    log("-" * 45)
    for i in range(43):
        ok = f"layer_{i}_output"
        if ok not in off or ok not in mg:
            continue
        cs = cosine(off[ok], mg[ok])
        c = cr[i] if i < len(cr) else -1
        if cs >= 0.999:
            status = "Perfect"
        elif cs >= 0.995:
            status = "Good"
        elif cs >= 0.99:
            status = "Degrading"
        elif cs >= 0.95:
            status = "<- drift starts"
        elif cs >= 0.8:
            status = "<- MoE cliff"
        elif cs >= 0.5:
            status = "<- cascade"
        elif cs >= 0.3:
            status = "<- worst"
        else:
            status = "<- near-random"
        log(f"{i:4d} {c:4d} {cs:10.6f}    {status}")

    # Post-layer
    log("\nPost-layer metrics:")
    for key in ["post_contraction", "last_hidden_state"]:
        if key in off and key in mg:
            cs = cosine(off[key], mg[key])
            log(f"  {key:25s}: cos={cs:.6f}")
    if "logits" in off and "logits" in mg:
        o_l = off["logits"].squeeze().float()
        m_l = mg["logits"].squeeze().float()
        if m_l.dim() == 2:
            m_l = m_l[-1]
        if o_l.dim() == 2:
            o_l = o_l[-1]
        v = min(o_l.shape[0], m_l.shape[0])
        cs = F.cosine_similarity(o_l[:v].unsqueeze(0), m_l[:v].unsqueeze(0)).item()
        log(f"  {'last-token logits':25s}: cos={cs:.6f}")
        o_t5 = o_l[:v].topk(5).indices.tolist()
        m_t5 = m_l[:v].topk(5).indices.tolist()
        overlap = len(set(o_t5) & set(m_t5))
        log(f"  Official top5: {o_t5}")
        log(f"  MCore top5:    {m_t5}")
        log(f"  Top-5 overlap: {overlap}/5")

    # Sub-layer breakdown
    log("\nSub-layer breakdown (cosine sim):")
    log(
        f"{'':>28s} {'L0 (dense)':>10s} {'L2 (CSA=4)':>10s} {'L10 (MoE)':>10s} {'L25 (cliff)':>11s} {'L42 (last)':>10s}"
    )
    log("-" * 82)

    # Map official sub-layer names to MCore names
    rows = [
        ("Attn input (HC+norm)", "attn_input", "input_layernorm_output"),
        ("Attn output", "attn_output", "self_attention_output"),
        ("MLP input (HC+norm)", "ffn_input", "pre_mlp_layernorm_output"),
        ("MLP/MoE output", "ffn_output", "mlp_output"),
        ("Full layer output", "output", "output"),
    ]
    for label, off_suffix, mg_suffix in rows:
        vals = []
        for i in sorted(SUBLAYER_LAYERS):
            if off_suffix == "output":
                ok = f"layer_{i}_output"
                mk = f"layer_{i}_output"
            else:
                ok = f"layer_{i}_{off_suffix}"
                mk = f"layer_{i}_{mg_suffix}"
            if ok in off and mk in mg:
                cs = cosine(off[ok], mg[mk])
                vals.append(f"{cs:.3f}")
            else:
                vals.append("  N/A")
        log(f"{label:>28s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s} {vals[3]:>11s} {vals[4]:>10s}")

log("\nDone")
if dist.is_initialized():
    dist.destroy_process_group()
