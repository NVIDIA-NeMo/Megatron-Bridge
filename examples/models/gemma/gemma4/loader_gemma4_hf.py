# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# HuggingFace Gemma-4 → Megatron checkpoint converter.
#
# Usage (via convert.py):
#  PYTHONPATH=/path/to/Megatron-Bridge/src:/path/to/Megatron-Bridge/examples/models/gemma/gemma4:$PYTHONPATH \
#  CUDA_DEVICE_MAX_CONNECTIONS=1 python /path/to/Megatron-LM/tools/checkpoint/convert.py \
#   --model-type GPT \
#   --loader gemma4_hf \
#   --saver core \
#   --load-dir ~/models/gemma-4-E4B-it \
#   --save-dir /path/to/gemma4-e4b-megatron \
#   --model-size gemma4-e4b \
#   --tokenizer-model ~/models/gemma-4-E4B-it \
#   --bf16 \
#   --target-tensor-parallel-size 2 \
#   --target-pipeline-parallel-size 1 \
#   --no-checking
#
# Weight layout differences between HF Gemma-4 and Megatron-core:
#
#   HF layer norms (4 per layer):
#     input_layernorm, post_attention_layernorm,
#     pre_feedforward_layernorm, post_feedforward_layernorm
#
#   Megatron Gemma4 (4 per layer, different names):
#     input_layernorm, post_self_attn_layernorm,
#     pre_mlp_layernorm, post_mlp_layernorm
#
#   HF attention weights (separate Q/K/V):
#     self_attn.q_proj, self_attn.k_proj, self_attn.v_proj,
#     self_attn.q_norm, self_attn.k_norm, self_attn.o_proj
#
#   Megatron attention weights (fused QKV, interleaved by GQA group):
#     self_attention.linear_qkv   (fused, shape [ng*(nh/ng+2)*hd, hs])
#     self_attention.q_layernorm  (per-head-group Q norm)
#     self_attention.k_layernorm  (per-head-group K norm)
#     self_attention.linear_proj  (output projection)
#
#   HF MLP:
#     mlp.gate_proj, mlp.up_proj, mlp.down_proj
#
#   Megatron MLP:
#     mlp.linear_fc1  (gate_proj and up_proj concatenated along dim-0)
#     mlp.linear_fc2  (down_proj)

import gc
import json
import os
import sys
import types

import torch
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BRIDGE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../../.."))
_BRIDGE_SRC = os.path.join(_BRIDGE_ROOT, "src")
if _BRIDGE_SRC not in sys.path:
    sys.path.insert(0, _BRIDGE_SRC)

try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("The 'transformers' package is required. Install with: pip install transformers")


# ---------------------------------------------------------------------------
# Argument definitions (consumed by convert.py)
# ---------------------------------------------------------------------------

def add_arguments(parser):
    group = parser.add_argument_group(title='Gemma-4 HuggingFace loader')
    group.add_argument(
        '--model-size',
        type=str,
        required=True,
        choices=['gemma4-9b', 'gemma4-27b', 'gemma4-mo-9b', 'gemma4-e4b'],
        help='Gemma-4 model variant to convert.',
    )
    group.add_argument(
        '--bf16',
        action='store_true',
        help='Load and convert weights in bfloat16 (recommended).',
    )
    group.add_argument(
        '--fp16',
        action='store_true',
        help='Load and convert weights in float16.',
    )
    group.add_argument(
        '--tokenizer-model',
        required=True,
        help='Path to (or HF repo name of) the Gemma-4 tokenizer / model directory.',
    )
    group.add_argument(
        '--megatron-path',
        type=str,
        default=None,
        help='Root directory of the Megatron-LM repository (added to sys.path).',
    )
    group.add_argument(
        '--make-vocab-size-divisible-by',
        type=int,
        default=None,
        help='Pad vocab size to a multiple of this value.',
    )
    group.add_argument(
        '--loader-transformer-impl',
        default='local',
        choices=['local', 'transformer_engine'],
        help='Transformer implementation to use when building the Megatron model.',
    )


# ---------------------------------------------------------------------------
# Per-variant architecture constants
# ---------------------------------------------------------------------------

# (num_layers, hidden_size, num_attention_heads, num_kv_heads, head_dim, ffn_hidden_size)
GEMMA4_CONFIGS = {
    'gemma4-9b':    (30, 2304, 8,  4,  256,  9216),
    'gemma4-27b':   (46, 4096, 16, 8,  256, 36864),
    'gemma4-mo-9b': (30, 2304, 8,  4,  256,  9216),   # MoE variant; same text config
    'gemma4-e4b':   (42, 2560, 8,  2,  256, 10240),   # google/gemma-4-E4B-it
}

# Attention pattern: every 6th layer is full attention, others are sliding-window.
# Matches Gemma-4's (i+1) % 6 != 0 → sliding rule.
SLIDING_WINDOW_SIZE = 512
WINDOW_ATTN_SKIP_FREQ = 6   # one full-attention layer every 6


# ---------------------------------------------------------------------------
# Utility: fuse Q/K/V weights into Megatron's GQA layout
# ---------------------------------------------------------------------------

def _fuse_qkv_gqa(q_weight, k_weight, v_weight, num_attention_heads, num_kv_heads, head_dim):
    """Interleave Q, K, V weights into Megatron's grouped-query layout.

    Megatron stores the fused QKV weight as:
        [ Q_group0_head0, Q_group0_head1, ..., K_group0, V_group0,
          Q_group1_head0, Q_group1_head1, ..., K_group1, V_group1,
          ... ]
    where each group shares one K and one V head.

    Args:
        q_weight : Tensor [num_attention_heads * head_dim, hidden_size]
        k_weight : Tensor [num_kv_heads * head_dim, hidden_size]
        v_weight : Tensor [num_kv_heads * head_dim, hidden_size]

    Returns:
        Tensor  [num_kv_heads * (num_q_per_group + 2) * head_dim, hidden_size]
    """
    hidden_size = q_weight.shape[1]
    num_q_per_group = num_attention_heads // num_kv_heads

    # Reshape to (num_kv_heads, num_q_per_group, head_dim, hidden_size)
    q = q_weight.view(num_kv_heads, num_q_per_group, head_dim, hidden_size)
    # Reshape to (num_kv_heads, 1, head_dim, hidden_size) for K and V
    k = k_weight.view(num_kv_heads, 1, head_dim, hidden_size)
    v = v_weight.view(num_kv_heads, 1, head_dim, hidden_size)

    # Concatenate along dim-1: [Q_heads, K_head, V_head] per group
    qkv = torch.cat([q, k, v], dim=1)  # (num_kv_heads, num_q_per_group+2, head_dim, hidden_size)

    return qkv.view(-1, hidden_size).contiguous()


# ---------------------------------------------------------------------------
# Metadata extraction from HF config
# ---------------------------------------------------------------------------

def _load_args_from_checkpoint(args, hf_config):
    """Populate Megatron args from HF Gemma-4 config dict."""

    args.seq_length = min(hf_config.get('max_position_embeddings', 131072), 8192)
    args.max_position_embeddings = hf_config['max_position_embeddings']
    args.hidden_size = hf_config['hidden_size']
    args.num_attention_heads = hf_config['num_attention_heads']
    args.num_layers = hf_config['num_hidden_layers']
    args.norm_epsilon = hf_config['rms_norm_eps']
    args.layernorm_epsilon = hf_config['rms_norm_eps']
    args.ffn_hidden_size = hf_config['intermediate_size']
    args.vocab_size = hf_config['vocab_size']
    args.padded_vocab_size = hf_config['vocab_size']
    args.kv_channels = hf_config.get('head_dim', args.hidden_size // args.num_attention_heads)
    args.global_kv_channels = hf_config.get('global_head_dim', None)
    args.global_batch_size = 1024
    args.iteration = 1
    args.position_embedding_type = 'rope'
    args.rotary_base = hf_config.get('rope_theta', 10000)
    args.normalization = 'RMSNorm'
    args.swiglu = False
    args.geglu = False
    args.geglu_tanh = True
    args.quick_geglu = False
    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = not hf_config.get('tie_word_embeddings', False)
    args.softmax_scale = 1.0
    args.scale_embeddings_by_hidden_size = True

    rope_parameters = hf_config.get('rope_parameters') or {}
    sliding_rope = rope_parameters.get('sliding_attention', {})
    full_rope = rope_parameters.get('full_attention', {})
    args.sliding_window_rope_base = sliding_rope.get('rope_theta', 10000.0)
    args.full_attention_rope_base = full_rope.get('rope_theta', 1000000.0)
    args.full_attention_rope_partial_factor = full_rope.get('partial_rotary_factor', 0.25)

    # Sliding window attention
    sliding_window = hf_config.get('sliding_window', SLIDING_WINDOW_SIZE)
    # HF causal sliding-window attention allows the current token and the previous
    # ``sliding_window - 1`` tokens. Megatron's tuple is (left, right), inclusive.
    args.window_size = (sliding_window - 1, 0)
    layer_types = hf_config.get('layer_types')
    if layer_types is not None:
        args.window_attn_skip_freq = [
            1 if layer_type == 'sliding_attention' else 0 for layer_type in layer_types
        ]
    else:
        args.window_attn_skip_freq = WINDOW_ATTN_SKIP_FREQ

    # GQA
    num_kv_heads = hf_config.get('num_key_value_heads', args.num_attention_heads)
    args.num_global_query_groups = None
    if num_kv_heads != args.num_attention_heads:
        args.group_query_attention = True
        args.num_query_groups = num_kv_heads
    else:
        args.group_query_attention = False
        args.num_query_groups = None

    # Per-layer embeddings
    args.per_layer_embed_vocab_size = hf_config.get(
        'vocab_size_per_layer_input', hf_config['vocab_size']
    )
    args.per_layer_embed_dim = hf_config.get('hidden_size_per_layer_input', 0)

    # Step 4: attention_k_eq_v — full-attention layers use K projection for V
    args.attention_k_eq_v = hf_config.get('attention_k_eq_v', False)

    # Step 3: Shared KV cache — last N layers reuse K/V from source layers
    args.num_kv_shared_layers = hf_config.get('num_kv_shared_layers', 0)

    # Step 5: MoE block
    args.enable_moe_block = hf_config.get('enable_moe_block', False)
    if args.enable_moe_block:
        args.num_experts = hf_config.get('num_experts', 1)
        args.moe_intermediate_size = hf_config.get('moe_intermediate_size', args.hidden_size)
        args.top_k_experts = hf_config.get('top_k_experts', 1)

    # qk_layernorm is always enabled in Gemma-4
    args.qk_layernorm = True


# ---------------------------------------------------------------------------
# Weight copying helpers
# ---------------------------------------------------------------------------

def _set_preprocess_state(model, hf_model):
    """Copy word-embedding weights."""
    model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight
    )
    if getattr(model, 'per_layer_embedding', None) is not None:
        model.per_layer_embedding.weight.data.copy_(hf_model.model.embed_tokens_per_layer.weight)
        model.per_layer_model_proj.weight.data.copy_(hf_model.model.per_layer_model_projection.weight)
        model.per_layer_proj_norm.weight.data.copy_(hf_model.model.per_layer_projection_norm.weight)


def _is_full_attention_layer(args, layer_idx):
    """Return True for full-attention layers. ``layer_idx`` is 0-based."""
    skip_freq = args.window_attn_skip_freq
    if isinstance(skip_freq, int):
        return (layer_idx + 1) % skip_freq == 0
    if isinstance(skip_freq, list):
        return not bool(skip_freq[layer_idx])
    return args.window_size is None


def _set_postprocess_state(args, model, hf_model):
    """Copy final norm and output-layer weights."""
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    if args.untie_embeddings_and_output_weights:
        model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def _is_kv_shared_layer(args, layer_idx):
    """Return True if layer_idx (0-based) is a shared-KV layer."""
    num_kv_shared = getattr(args, 'num_kv_shared_layers', 0)
    if num_kv_shared <= 0:
        return False
    num_layers = args.num_layers
    return layer_idx >= (num_layers - num_kv_shared)


def _set_layer_state(args, model, hf_model, layer_idx):
    """Copy all parameters for one transformer layer.

    Maps HF Gemma-4 naming → Megatron Gemma4TransformerLayer naming.

    Handles:
    - Step 3 (shared KV): shared layers have no k_proj/v_proj/k_norm/v_norm;
      their fused QKV in Megatron has zero K/V rows.
    - Step 4 (attention_k_eq_v): full-attention layers share K and V projections;
      the V rows of fused QKV are zeroed (unused at runtime).
    - Step 5 (MoE): copies router + expert weights when enable_moe_block=True.
    """
    megatron_layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    num_attention_heads = args.num_attention_heads
    is_full_attention = _is_full_attention_layer(args, layer_idx)
    is_shared = _is_kv_shared_layer(args, layer_idx)
    # Step 4: k_eq_v applies to full-attention non-shared layers
    k_eq_v = getattr(args, 'attention_k_eq_v', False) and is_full_attention and not is_shared

    num_kv_heads = args.num_query_groups if args.group_query_attention else num_attention_heads
    if is_full_attention and args.num_global_query_groups is not None:
        num_kv_heads = args.num_global_query_groups
    head_dim = (
        args.global_kv_channels
        if is_full_attention and args.global_kv_channels is not None
        else args.kv_channels
    )

    # --- Layer norms ---
    megatron_layer.input_layernorm.weight.data.copy_(
        hf_layer.input_layernorm.weight
    )
    megatron_layer.post_self_attn_layernorm.weight.data.copy_(
        hf_layer.post_attention_layernorm.weight
    )
    megatron_layer.pre_mlp_layernorm.weight.data.copy_(
        hf_layer.pre_feedforward_layernorm.weight
    )
    megatron_layer.post_mlp_layernorm.weight.data.copy_(
        hf_layer.post_feedforward_layernorm.weight
    )

    # --- Attention: fused QKV ---
    hidden_size = hf_layer.self_attn.q_proj.weight.shape[1]

    if is_shared:
        # Step 3: shared-KV layers have only q_proj (no k_proj/v_proj in HF).
        # Build fused QKV with real Q weights and zero K/V rows.
        q_weight = hf_layer.self_attn.q_proj.weight
        k_zero = torch.zeros(num_kv_heads * head_dim, hidden_size,
                             dtype=q_weight.dtype, device=q_weight.device)
        v_zero = torch.zeros_like(k_zero)
        fused_qkv = _fuse_qkv_gqa(q_weight, k_zero, v_zero,
                                   num_attention_heads, num_kv_heads, head_dim)
    elif k_eq_v:
        # Step 4: k_eq_v — V uses K projection; V rows in fused QKV are zero.
        q_weight = hf_layer.self_attn.q_proj.weight
        k_weight = hf_layer.self_attn.k_proj.weight
        v_zero = torch.zeros_like(k_weight)
        fused_qkv = _fuse_qkv_gqa(q_weight, k_weight, v_zero,
                                   num_attention_heads, num_kv_heads, head_dim)
    else:
        fused_qkv = _fuse_qkv_gqa(
            hf_layer.self_attn.q_proj.weight,
            hf_layer.self_attn.k_proj.weight,
            hf_layer.self_attn.v_proj.weight,
            num_attention_heads,
            num_kv_heads,
            head_dim,
        )
    megatron_layer.self_attention.linear_qkv.weight.data.copy_(fused_qkv)

    # --- Attention: qk layernorms ---
    megatron_layer.self_attention.q_layernorm.weight.data.copy_(
        hf_layer.self_attn.q_norm.weight
    )
    if not is_shared:
        # Shared layers have no k_norm in HF
        megatron_layer.self_attention.k_layernorm.weight.data.copy_(
            hf_layer.self_attn.k_norm.weight
        )

    # --- Attention: output projection ---
    megatron_layer.self_attention.linear_proj.weight.data.copy_(
        hf_layer.self_attn.o_proj.weight
    )

    # --- MLP: fused gate + up (linear_fc1) ---
    # Megatron concatenates gate_proj and up_proj along dim-0 for SwiGLU/GeGLU.
    fused_fc1 = torch.cat([
        hf_layer.mlp.gate_proj.weight,
        hf_layer.mlp.up_proj.weight,
    ], dim=0)
    megatron_layer.mlp.linear_fc1.weight.data.copy_(fused_fc1)

    # --- MLP: down projection (linear_fc2) ---
    megatron_layer.mlp.linear_fc2.weight.data.copy_(hf_layer.mlp.down_proj.weight)

    # --- Step 5: MoE block ---
    if getattr(megatron_layer, 'moe_router', None) is not None:
        hf_router = hf_layer.router
        hf_experts = hf_layer.experts
        # Router weights (norm has no weight — it's scaleless)
        megatron_layer.moe_router.scale.data.copy_(hf_router.scale)
        megatron_layer.moe_router.proj.weight.data.copy_(hf_router.proj.weight)
        megatron_layer.moe_router.per_expert_scale.data.copy_(hf_router.per_expert_scale)
        # Expert weights (stored as 3D tensors: [E, out, in])
        megatron_layer.moe_experts.gate_up_proj.data.copy_(hf_experts.gate_up_proj)
        megatron_layer.moe_experts.down_proj.data.copy_(hf_experts.down_proj)
        # Extra norms
        megatron_layer.post_feedforward_layernorm_1.weight.data.copy_(
            hf_layer.post_feedforward_layernorm_1.weight
        )
        megatron_layer.post_feedforward_layernorm_2.weight.data.copy_(
            hf_layer.post_feedforward_layernorm_2.weight
        )
        megatron_layer.pre_feedforward_layernorm_2.weight.data.copy_(
            hf_layer.pre_feedforward_layernorm_2.weight
        )

    # --- Phase 4: Per-Layer Embedding (PLE) weights ---
    if getattr(megatron_layer, 'per_layer_input_gate', None) is not None:
        megatron_layer.per_layer_input_gate.weight.data.copy_(hf_layer.per_layer_input_gate.weight)
        megatron_layer.per_layer_projection.weight.data.copy_(hf_layer.per_layer_projection.weight)
        megatron_layer.post_per_layer_input_norm.weight.data.copy_(
            hf_layer.post_per_layer_input_norm.weight
        )
        megatron_layer.layer_scalar.data.copy_(hf_layer.layer_scalar)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _load_checkpoint_to_model(margs):
    """Build a Megatron mcore GPT model and fill it with HF weights."""

    from gpt_builders import gpt_builder
    from model_provider import model_provider

    # Load HF model on CPU
    dtype = (
        torch.bfloat16 if margs.bf16
        else torch.float16 if margs.fp16
        else torch.float32
    )
    print(f"Loading HuggingFace model from {margs.load} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        margs.load,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )

    # Multimodal Gemma4 (e.g. gemma-4-E4B-it): text weights are under model.language_model.
    # Redirect hf_model.model to the text sub-model so all downstream accessors are uniform.
    if hasattr(hf_model.model, 'language_model'):
        hf_model.model = hf_model.model.language_model

    # Build Megatron mcore model (uses our Gemma4TransformerLayer via --spec)
    print("Building Megatron model ...")
    model = model_provider(gpt_builder, pre_process=True, post_process=True).to(dtype)

    # Step 3: wire up shared-KV references so shared layers can access source KV
    from megatron.bridge.models.gemma.gemma4_layer_specs import wire_gemma4_kv_sharing
    wire_gemma4_kv_sharing(model)

    # Copy weights
    print("Copying weights ...")
    _set_preprocess_state(model, hf_model)
    _set_postprocess_state(margs, model, hf_model)
    for layer_idx in tqdm(range(margs.num_layers), desc='layer'):
        _set_layer_state(margs, model, hf_model, layer_idx)

    del hf_model
    gc.collect()
    return model


# ---------------------------------------------------------------------------
# Main entry-point for convert.py
# ---------------------------------------------------------------------------

def _load_checkpoint(queue, args):
    """Load HF Gemma-4 checkpoint and emit tensors over the queue."""

    # ---- Path setup ----
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    ))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from utils import _ConverterFakeProcessGroup

        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.core.models.common.language_module.language_module import LanguageModule
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables
    except ModuleNotFoundError as exc:
        print(f"Unable to import Megatron ({exc}). Use --megatron-path to specify its location.")
        queue.put("exit")
        return

    # ---- Read HF config ----
    hf_config_path = os.path.join(args.load_dir, 'config.json')
    if not os.path.isfile(hf_config_path):
        print(f"config.json not found at {hf_config_path}")
        queue.put("exit")
        return
    with open(hf_config_path) as fh:
        hf_config = json.load(fh)

    # Multimodal Gemma4 (e.g. gemma-4-E4B-it) wraps text params under text_config.
    if 'text_config' in hf_config:
        hf_config = hf_config['text_config']

    # ---- Build sys.argv for Megatron's argument parser ----
    sys.argv = [
        'script.py',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-rope-fusion',
        '--no-persist-layer-norm',
        '--use-cpu-initialization',
        '--micro-batch-size', '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--mock-data',
        '--no-initialization',
        '--load', args.load_dir,
        '--no-one-logger',
        # Custom Gemma-4 layer spec
        '--spec', 'megatron.bridge.models.gemma.gemma4_layer_specs', 'gemma4_layer_spec',
        '--use-mcore-models',
        '--transformer-impl', args.loader_transformer_impl,
    ]
    if args.make_vocab_size_divisible_by is not None:
        sys.argv += ['--make-vocab-size-divisible-by', str(args.make_vocab_size_divisible_by)]

    margs = parse_args()

    # Populate architecture from HF config
    _load_args_from_checkpoint(margs, hf_config)

    margs.tokenizer_type = 'HuggingFaceTokenizer'
    margs.tokenizer_model = args.tokenizer_model
    margs.model_type = ModelType.encoder_or_decoder
    margs.params_dtype = (
        torch.bfloat16 if args.bf16
        else torch.float16 if args.fp16
        else torch.float32
    )
    margs.bf16 = args.bf16
    margs.fp16 = args.fp16
    margs.world_size = 1  # single-process conversion

    margs = validate_args(margs)
    margs.use_legacy_models = False   # use mcore

    # Suppress distributed-init warnings
    LanguageModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(1)
    mpu.set_pipeline_model_parallel_world_size(1)
    mpu.set_virtual_pipeline_model_parallel_world_size(None)
    fake_tp = _ConverterFakeProcessGroup(size=1)
    fake_ep = _ConverterFakeProcessGroup(size=1)
    fake_dp = _ConverterFakeProcessGroup(size=1)
    mpu._TENSOR_MODEL_PARALLEL_GROUP = fake_tp
    mpu._EXPERT_MODEL_PARALLEL_GROUP = fake_ep
    # ProcessGroupCollection.use_mpu_process_groups() requires these three DP groups.
    mpu._DATA_PARALLEL_GROUP = fake_dp
    mpu._DATA_PARALLEL_GROUP_WITH_CP = fake_dp
    mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = fake_dp
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)

    # ---- Build model and load weights ----
    margs.load = args.load_dir
    model = _load_checkpoint_to_model(margs)

    # ---- Metadata ----
    md = types.SimpleNamespace()
    md.model_type = 'GPT'
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = False
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = 'rope'
    md.linear_bias = False
    md.qkv_bias = False
    md.norm_has_bias = False
    md.swiglu = False
    md.previous_tensor_parallel_size = 1
    md.previous_pipeline_parallel_size = 1
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.true_vocab_size = margs.vocab_size
    # Gemma-4 specific metadata (consumed by compatible savers)
    md.gemma4 = True
    md.geglu = True  # gate+up fused weight needs interleaved TP split (not contiguous)
    md.qk_layernorm = True
    md.window_size = margs.window_size
    md.window_attn_skip_freq = margs.window_attn_skip_freq

    queue.put(md)

    def queue_put(name, msg):
        print(f"  sending: {name}")
        msg['name'] = name
        queue.put(msg)

    # ---- Embeddings ----
    emb_msg = {'word embeddings': model.embedding.word_embeddings.weight.data}
    if getattr(model, 'per_layer_embedding', None) is not None:
        emb_msg['per layer embeddings'] = model.per_layer_embedding.weight.data
        emb_msg['per layer model proj'] = model.per_layer_model_proj.weight.data
        emb_msg['per layer proj norm'] = model.per_layer_proj_norm.weight.data
    queue_put('embeddings', emb_msg)

    # ---- Transformer layers ----
    for layer_num in range(margs.num_layers):
        layer = model.decoder.layers[layer_num]
        attn = layer.self_attention

        msg = {
            # Layer norms
            'input norm weight':     layer.input_layernorm.weight.data,
            'post attn norm weight': layer.post_self_attn_layernorm.weight.data,
            'pre mlp norm weight':   layer.pre_mlp_layernorm.weight.data,
            'post mlp norm weight':  layer.post_mlp_layernorm.weight.data,
            # Attention
            'qkv weight':            attn.linear_qkv.weight.data,
            'q norm weight':         attn.q_layernorm.weight.data,
            'k norm weight':         attn.k_layernorm.weight.data,
            'dense weight':          attn.linear_proj.weight.data,
            # MLP
            'mlp l0 weight':         layer.mlp.linear_fc1.weight.data,
            'mlp l1 weight':         layer.mlp.linear_fc2.weight.data,
        }
        # Per-Layer Embedding (PLE) weights — only present when per_layer_embed_dim > 0
        if getattr(layer, 'per_layer_input_gate', None) is not None:
            msg['ple gate weight'] = layer.per_layer_input_gate.weight.data
            msg['ple proj weight'] = layer.per_layer_projection.weight.data
            msg['ple norm weight'] = layer.post_per_layer_input_norm.weight.data
            msg['ple scalar']      = layer.layer_scalar.data
        queue_put(f'transformer layer {layer_num}', msg)

    # ---- Final norm ----
    queue_put('final norm', {
        'weight': model.decoder.final_layernorm.weight.data,
    })

    # ---- Output layer ----
    if md.output_layer:
        queue_put('output layer', {
            'weight': model.output_layer.weight.data,
        })

    queue.put('done')


def load_checkpoint(queue, args):
    """Entry-point called by convert.py (wraps _load_checkpoint for error handling)."""
    try:
        _load_checkpoint(queue, args)
    except Exception:
        import traceback
        traceback.print_exc()
        queue.put('exit')
