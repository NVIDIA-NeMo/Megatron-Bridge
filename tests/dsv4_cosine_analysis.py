"""Per-layer and sub-layer COSINE SIMILARITY comparison.
Runs official (MODE=official) and Megatron (MODE=megatron) separately,
saves actual hidden state tensors for rank 0. Then MODE=compare computes cosine sim.

For MoE layers, also saves sub-layer outputs (HC pre, attn, MLP, HC post).
"""

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

# Layers to capture sub-layer detail (dense + representative MoE)
SUBLAYER_LAYERS = {0, 2, 10, 25, 42}

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "4"))


def log(msg):
    if rank == 0:
        print(msg, flush=True)


os.makedirs(SAVE_DIR, exist_ok=True)
torch.cuda.set_device(local_rank)

if MODE == "official":
    log("=== Official BF16: per-layer + sub-layer hidden states ===")
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

    log("Creating model...")
    with torch.device("cuda"):
        model = Transformer(args)
    from safetensors.torch import load_model

    load_model(model, f"{WKDIR}/dsv4_flash_converted_bf16_mp4/model{rank}-mp{world_size}.safetensors", strict=False)
    torch.set_default_device("cuda")

    saved = {}
    # Hook every layer output
    for i, layer in enumerate(model.layers):

        def make_hook(idx):
            def fn(mod, inp, out):
                t = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (tuple, list)) else None)
                if t is not None and rank == 0:
                    saved[f"layer_{idx}_output"] = t.detach().float().cpu()

            return fn

        layer.register_forward_hook(make_hook(i))

    # Sub-layer hooks for selected layers
    for i in SUBLAYER_LAYERS:
        if i >= len(model.layers):
            continue
        layer = model.layers[i]
        for name, mod in [("attn", layer.attn), ("ffn", layer.ffn)]:

            def make_sub(idx, n):
                def fn(mod, inp, out):
                    t = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (tuple, list)) else None)
                    if t is not None and rank == 0:
                        saved[f"layer_{idx}_{n}_output"] = t.detach().float().cpu()
                    if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor) and rank == 0:
                        saved[f"layer_{idx}_{n}_input"] = inp[0].detach().float().cpu()

                return fn

            mod.register_forward_hook(make_sub(i, name))

    tokens = torch.tensor([INPUT_IDS], dtype=torch.long, device="cuda")
    log("Forward...")
    with torch.no_grad():
        logits = model(tokens, start_pos=0)

    if rank == 0:
        saved["logits"] = logits.detach().float().cpu()
        torch.save(saved, f"{SAVE_DIR}/official_states.pt")
        log(f"Saved {len(saved)} tensors")
        for k in sorted(saved.keys()):
            log(f"  {k}: {list(saved[k].shape)}")

elif MODE == "megatron":
    log("=== Megatron: per-layer + sub-layer hidden states ===")
    TP = 1
    ETP = int(os.environ.get("ETP_SIZE", "4"))
    dist.init_process_group(
        "nccl",
        init_method="tcp://" + os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"],
        world_size=world_size,
        rank=rank,
    )
    from megatron.core import parallel_state, tensor_parallel

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=ETP,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(42)

    from megatron.bridge import AutoBridge
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
    from megatron.bridge.utils.common_utils import disable_mtp_for_inference

    bridge = AutoBridge.from_hf_pretrained(MODEL_PATH)
    hf = PreTrainedCausalLM(MODEL_PATH)
    prov = bridge._model_bridge.provider_bridge(hf)
    prov.tensor_model_parallel_size = TP
    prov.pipeline_model_parallel_size = 1
    prov.expert_model_parallel_size = 1
    prov.expert_tensor_parallel_size = ETP
    prov.finalize()
    model = prov.provide_distributed_model(wrap_with_ddp=False)
    sd = torch.load(f"{WKDIR}/dsv4_flash_megatron_ckpt_tp1/rank_{rank}.pt", map_location="cpu", weights_only=False)
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
    # Hook every layer output
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
        torch.save(saved, f"{SAVE_DIR}/megatron_states.pt")
        log(f"Saved {len(saved)} tensors")
        for k in sorted(saved.keys()):
            log(f"  {k}: {list(saved[k].shape)}")

elif MODE == "compare":
    log("=== Comparing cosine similarity ===")
    off = torch.load(f"{SAVE_DIR}/official_states.pt", map_location="cpu", weights_only=False)
    mg = torch.load(f"{SAVE_DIR}/megatron_states.pt", map_location="cpu", weights_only=False)

    with open(f"{WKDIR}/models/deepseek-ai/DeepSeek-V4-Flash/config.json") as f:
        cfg = json.load(f)
    cr = cfg.get("compress_ratios", cfg.get("compress_rates", []))

    def cosine(a, b):
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        v = min(a_flat.shape[0], b_flat.shape[0])
        return F.cosine_similarity(a_flat[:v].unsqueeze(0), b_flat[:v].unsqueeze(0)).item()

    # Per-layer output cosine sim
    log(f"\n{'Lyr':>4} {'CR':>4} {'CosSim':>10} {'NormRatio':>10} {'Gap':>8}")
    log("-" * 44)
    for i in range(43):
        ok = f"layer_{i}_output"
        mk = f"layer_{i}_output"
        if ok not in off or mk not in mg:
            log(f"{i:4d}  -- MISSING")
            continue
        o, m = off[ok], mg[mk]
        cs = cosine(o, m)
        nr = m.norm().item() / o.norm().item() if o.norm().item() > 0 else float("inf")
        c = cr[i] if i < len(cr) else -1
        flag = " <<<" if cs < 0.99 else ""
        log(f"{i:4d} {c:4d} {cs:10.6f} {nr:10.4f} {1 - cs:8.6f}{flag}")

    # Sub-layer breakdown for selected layers
    log(f"\n=== Sub-layer cosine similarity for layers {sorted(SUBLAYER_LAYERS)} ===")
    for i in sorted(SUBLAYER_LAYERS):
        log(f"\n  Layer {i} (CR={cr[i] if i < len(cr) else '?'}):")
        # Find matching keys
        off_keys = sorted(k for k in off if k.startswith(f"layer_{i}_") and k != f"layer_{i}_output")
        mg_keys = sorted(k for k in mg if k.startswith(f"layer_{i}_") and k != f"layer_{i}_output")
        log(f"    Official sub-keys: {off_keys}")
        log(f"    MCore sub-keys:    {mg_keys}")

        # Map official sub-layer names to MCore names
        mappings = [
            ("attn_output", "self_attention_output"),
            ("attn_input", "input_layernorm_output"),  # attn input = layernorm output
            ("ffn_output", "mlp_output"),
            ("ffn_input", "pre_mlp_layernorm_output"),
        ]
        for off_suffix, mg_suffix in mappings:
            ok = f"layer_{i}_{off_suffix}"
            mk = f"layer_{i}_{mg_suffix}"
            if ok in off and mk in mg:
                cs = cosine(off[ok], mg[mk])
                flag = " <<<" if cs < 0.99 else ""
                log(f"    {off_suffix:20s} vs {mg_suffix:35s}: cos={cs:.6f}{flag}")
            elif ok in off:
                log(f"    {off_suffix:20s} — no MCore match ({mg_suffix})")
            # Also try direct name match
            for mk2 in mg_keys:
                if mg_suffix in mk2 and mk2 not in [mk]:
                    cs2 = cosine(off[ok], mg[mk2])
                    log(f"    {off_suffix:20s} vs {mk2:35s}: cos={cs2:.6f}")

        # Layer output
        ok = f"layer_{i}_output"
        mk = f"layer_{i}_output"
        if ok in off and mk in mg:
            cs = cosine(off[ok], mg[mk])
            log(f"    {'FULL_output':20s} vs {'FULL_output':35s}: cos={cs:.6f}")

    # Logit cosine sim (all positions)
    if "logits" in off and "logits" in mg:
        log("\n=== Logit cosine similarity ===")
        o_logits = off["logits"].float()
        m_logits = mg["logits"].float()
        if o_logits.dim() == 3:
            o_logits = o_logits[0]  # [seq, vocab_chunk]
        if m_logits.dim() == 3:
            m_logits = m_logits[0]
        log(f"Official logits: {list(o_logits.shape)}, MCore logits: {list(m_logits.shape)}")

        # Note: official with MP=4 only has vocab/4 on rank 0
        seq = min(o_logits.shape[0], m_logits.shape[0])
        v = min(o_logits.shape[-1], m_logits.shape[-1])
        for p in range(seq):
            cs = F.cosine_similarity(o_logits[p, :v].unsqueeze(0), m_logits[p, :v].unsqueeze(0)).item()
            log(f"  pos {p:2d}: cos={cs:.6f}")

log("\nDone")
if dist.is_initialized():
    dist.destroy_process_group()
