import os
import sys
import traceback


MODEL_PATH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/nemo_home/models/deepseek/DeepSeek-V4-Flash"

PASS = "\033[92m\u2713\033[0m"
FAIL = "\033[91m\u2717\033[0m"
SKIP = "\033[93m-\033[0m"


def section(title):
    print(f"\n{chr(61) * 60}\n  {title}\n{chr(61) * 60}", flush=True)


def ok(msg):
    print(f"  {PASS} {msg}", flush=True)


def fail(msg, exc=None):
    print(f"  {FAIL} {msg}", flush=True)
    if exc:
        traceback.print_exc()


# Phase 1
section("Phase 1 -- Bridge registration")
try:
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(MODEL_PATH)
    ok("AutoBridge loaded: " + type(bridge).__name__)
    ok("model_type = " + bridge.hf_pretrained.config.model_type)
except Exception as e:
    fail("AutoBridge.from_hf_pretrained failed", e)
    sys.exit(1)

# Phase 2
section("Phase 2 -- Provider config (no GPU)")
try:
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

    hf_pretrained = PreTrainedCausalLM(MODEL_PATH)
    provider = bridge._model_bridge.provider_bridge(hf_pretrained)
    checks = {
        "experimental_attention_variant": "dsv4_hybrid",
        "enable_hyper_connections": True,
        "num_residual_streams": 4,
        "moe_n_hash_layers": 3,
        "moe_router_score_function": "sqrtsoftplus",
        "v_head_dim": 512,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "csa_window_size": 128,
        "mtp_num_layers": 1,
    }
    all_ok = True
    for attr, expected in checks.items():
        val = getattr(provider, attr, "__missing__")
        if val == expected:
            ok("provider." + attr + " = " + repr(val))
        else:
            fail("provider." + attr + " = " + repr(val) + " (expected " + repr(expected) + ")")
            all_ok = False
    csa = getattr(provider, "csa_compress_ratios", None)
    ok("csa_compress_ratios[:4] = " + repr(csa[:4] if csa else None))
    if all_ok:
        ok("Provider config looks correct")
except Exception as e:
    fail("provider_bridge() failed", e)
    sys.exit(1)

# Phase 3
section("Phase 3 -- Mapping registry")
try:
    registry = bridge._model_bridge.mapping_registry()
    mappings = list(registry)
    ok("Total mappings: " + str(len(mappings)))
    for m in mappings[:8]:
        print("    megatron=" + repr(m.megatron_param) + "  hf=" + repr(m.hf_param))
    print("    ...")
    for m in mappings[-4:]:
        print("    megatron=" + repr(m.megatron_param) + "  hf=" + repr(m.hf_param))
    hf_params = set()
    for m in mappings:
        if isinstance(m.hf_param, dict):
            hf_params.update(m.hf_param.values())
        else:
            hf_params.add(m.hf_param)
    for key in [
        "embed.weight",
        "head.weight",
        "norm.weight",
        "layers.*.attn.wq_a.weight",
        "layers.*.attn.wkv.weight",
        "layers.*.attn.wo_a.weight",
        "layers.*.attn.attn_sink",
        "layers.*.attn.compressor.wkv.weight",
        "layers.*.attn.compressor.ape",
        "layers.*.attn.indexer.wq_b.weight",
        "layers.*.attn.indexer.compressor.wkv.weight",
        "layers.*.ffn.gate.tid2eid",
        "layers.*.hc_attn_fn",
        "layers.*.hc_attn_scale",
        "hc_head_fn",
        "mtp.0.e_proj.weight",
    ]:
        if key in hf_params:
            ok("mapping: " + repr(key))
        else:
            fail("MISSING: " + repr(key))
except Exception as e:
    fail("mapping_registry() failed", e)
    sys.exit(1)

# Phases 4-7: use a tiny synthetic model to test code paths without OOM
# (Full model needs 16+ H100-80GB GPUs; tiny model tests mechanics on 1 GPU)
section("Phase 4 -- Tiny model init (3 layers: hash/4x/128x, 1 GPU)")
model = None
try:
    import torch

    if not torch.cuda.is_available():
        print("  " + SKIP + " No GPU -- skipping")
    else:
        import torch.distributed as dist

        if not dist.is_initialized():
            port = int(os.environ.get("MASTER_PORT", "29500"))
            dist.init_process_group(backend="nccl", init_method=f"tcp://localhost:{port}", world_size=1, rank=0)
        from megatron.core import parallel_state, tensor_parallel

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
        tensor_parallel.model_parallel_cuda_manual_seed(42)

        # Shrink to 3 layers: one of each CSA type (hash / 4x / 128x)
        provider.num_layers = 3
        provider.moe_n_hash_layers = 1
        provider.mtp_num_layers = None  # skip MTP for tiny test
        provider.csa_compress_ratios = [0, 4, 128]
        provider.moe_layer_freq = [1, 1, 1]
        provider.tensor_model_parallel_size = 1
        provider.pipeline_model_parallel_size = 1
        provider.expert_model_parallel_size = 1
        provider.finalize()

        model = provider.provide_distributed_model(wrap_with_ddp=False)
        ok("Tiny model instantiated: " + str(type(model[0]).__name__) + " (list of " + str(len(model)) + ")")
        num_params = sum(p.numel() for p in model[0].parameters())
        ok("Parameter count: " + f"{num_params:,}")
except Exception as e:
    fail("Model init failed", e)

section("Phase 5 -- Export param names from tiny model")
if model is None:
    print("  " + SKIP + " Skipped (no model)")
else:
    try:
        # Check that named parameters match expected Megatron naming
        param_names = [n for n, _ in model[0].named_parameters()]
        ok("Total named parameters: " + str(len(param_names)))
        for name in param_names[:10]:
            print("    " + name)
        print("    ...")
        # Verify key Megatron param names exist
        mg_set = set(param_names)
        # Float16Module wraps model and adds "module." prefix to all param names
        prefix = "module."
        for key in [
            "embedding.word_embeddings.weight",
            "decoder.final_layernorm.weight",
            "decoder.layers.0.self_attention.linear_q_down_proj.weight",
            "decoder.layers.0.self_attention.linear_proj.weight",
            "decoder.layers.0.mlp.router.weight",
        ]:
            full = prefix + key
            if full in mg_set:
                ok("param exists: " + key)
            else:
                fail("MISSING param: " + key)
    except Exception as e:
        fail("Parameter inspection failed", e)

section("Phase 6+7 -- Export weights from tiny model")
if model is None:
    print("  " + SKIP + " Skipped (no model)")
else:
    try:
        exported = {}
        for name, tensor in bridge.export_hf_weights(model, cpu=True):
            exported[name] = tensor
        ok("export_hf_weights() produced " + str(len(exported)) + " tensors")
        # Check a few expected HF keys exist in export
        for key in [
            "embed.weight",
            "head.weight",
            "norm.weight",
            "layers.0.attn.wq_a.weight",
            "layers.0.attn.wo_a.weight",
            "layers.0.ffn.gate.weight",
            "layers.1.attn.compressor.wkv.weight",
            "layers.1.attn.compressor.ape",
            "layers.1.attn.indexer.wq_b.weight",
            "layers.2.attn.compressor.wkv.weight",
        ]:
            if key in exported:
                ok("exported: " + repr(key) + " shape=" + str(tuple(exported[key].shape)))
            else:
                fail("MISSING exported key: " + repr(key))
    except Exception as e:
        fail("Export failed", e)

section("Done")
