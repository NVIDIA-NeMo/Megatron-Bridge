# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forward-pass parity test for LLaDA1.5: HF reference vs Megatron Bridge.

Loads the trust_remote_code ``LLaDAModelLM`` reference and a Megatron
``GPTModel`` built via :class:`LLaDA15Bridge` from the same checkpoint,
runs the same prompt through both, and reports full-tensor logit similarity.

Usage::

    python examples/conversion/test_llada15_parity.py \
        --hf-path /path/to/GSAI-ML/LLaDA-1.5/snapshot \
        --prompt "The capital of France is"

Single-GPU; an 8B model in bf16 fits twice in 80 GB.
"""

import argparse
import gc
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Make sure the bridge module registers itself with AutoBridge.
from megatron.bridge import AutoBridge  # noqa: E402
from megatron.bridge.diffusion.conversion.llada15 import llada15_bridge  # noqa: F401, E402


# Forward-pass parity thresholds (bf16, single GPU). For an 8B model summed
# over 32 layers + a 126k-way LM head matmul, max absolute logit drift in
# the 0.1-1.0 range is normal bf16 noise — argmax / top-k agreement is the
# stronger correctness signal. We require both.
COS_SIM_THRESHOLD = 0.999
REL_MAX_DIFF_THRESHOLD = 0.05  # max_diff / logit range
TOP5_OVERLAP_THRESHOLD = 4  # out of 5


def patch_llada_for_transformers5(cls):
    """Add transformers-5.x compat shims to the trust_remote_code LLaDA class.

    The bundled ``modeling_llada.py`` was written for transformers 4.46 and
    fails to load under 5.x because:
      - finalize_model_loading looks for ``all_tied_weights_keys``
      - ``tie_weights`` is now called with ``missing_keys``/``recompute_mapping``
      - ``config.use_cache`` is no longer auto-populated

    ``weight_tying=False`` on LLaDA1.5, so the tied-weights bookkeeping is a no-op.
    """
    cls.all_tied_weights_keys = {}
    orig_tie = cls.tie_weights

    def _tie_weights_compat(self, *args, **kwargs):
        kwargs.pop("missing_keys", None)
        kwargs.pop("recompute_mapping", None)
        return orig_tie(self, *args, **kwargs)

    cls.tie_weights = _tie_weights_compat


def load_hf_reference(hf_path: str, device: str = "cuda:0"):
    """Load the HF reference model with the transformers-5.x compat shims."""
    from transformers import AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(
        "modeling_llada.LLaDAModelLM",
        hf_path,
        trust_remote_code=True,
    )
    patch_llada_for_transformers5(cls)

    model = (
        AutoModelForCausalLM.from_pretrained(
            hf_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    model.config.use_cache = False
    if not hasattr(model.config, "use_return_dict"):
        model.config.use_return_dict = True
    return model


def load_megatron_via_bridge(hf_path: str):
    """Convert HF → Megatron via AutoBridge and return the GPTModel."""
    bridge = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True)
    megatron_model = bridge.to_megatron_model(wrap_with_ddp=False)
    if isinstance(megatron_model, list):
        # Pipeline-parallel returns a list; single-GPU path returns one model.
        assert len(megatron_model) == 1
        megatron_model = megatron_model[0]
    return bridge, megatron_model.eval()


def hf_forward(hf_model, input_ids):
    with torch.no_grad():
        out = hf_model(input_ids=input_ids)
    return (out.logits if hasattr(out, "logits") else out[0]).float()


def megatron_forward(megatron_model, input_ids):
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    with torch.no_grad():
        # Bidirectional attention is enforced by LLaDA15TEDotProductAttention,
        # so we don't need to pass an explicit attention_mask.
        out = megatron_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
    logits = out if isinstance(out, torch.Tensor) else out[0]
    return logits.float()


def compare_logits(hf_logits: torch.Tensor, mcore_logits: torch.Tensor, *, hf_vocab_size: int):
    """Report cosine sim, max/mean diff, argmax agreement, and top-k overlap."""
    # Megatron pads vocab — truncate to HF vocab so we compare the same slice.
    if mcore_logits.shape[-1] != hf_vocab_size:
        mcore_logits = mcore_logits[..., :hf_vocab_size]
    assert hf_logits.shape == mcore_logits.shape, (
        f"shape mismatch: hf={tuple(hf_logits.shape)} mcore={tuple(mcore_logits.shape)}"
    )

    diff = (hf_logits - mcore_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos = F.cosine_similarity(hf_logits.flatten().unsqueeze(0), mcore_logits.flatten().unsqueeze(0)).item()

    hf_argmax = hf_logits.argmax(dim=-1)
    mc_argmax = mcore_logits.argmax(dim=-1)
    n_agree = int((hf_argmax == mc_argmax).sum().item())
    n_total = int(hf_argmax.numel())

    # Top-5 agreement at the last position is a useful sanity check beyond argmax.
    hf_top5 = torch.topk(hf_logits[0, -1], 5).indices.tolist()
    mc_top5 = torch.topk(mcore_logits[0, -1], 5).indices.tolist()
    top5_overlap = len(set(hf_top5) & set(mc_top5))

    # Scale-relative max diff: max_diff normalized by the HF logit range. Useful
    # for understanding whether large absolute diffs are large in *relative* terms.
    hf_range = (hf_logits.max() - hf_logits.min()).item()
    rel_max_diff = max_diff / hf_range if hf_range > 0 else float("nan")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_max_diff": rel_max_diff,
        "cos_sim": cos,
        "argmax_all_match": (n_agree == n_total),
        "argmax_agreement": f"{n_agree}/{n_total}",
        "top5_overlap_last_pos": f"{top5_overlap}/5",
        "hf_top5_last": hf_top5,
        "mcore_top5_last": mc_top5,
        "hf_logit_range": hf_range,
    }


def setup_distributed_single_gpu():
    """Initialize a 1-rank process group so Megatron's parallel state works."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", type=str, required=True, help="Local path to LLaDA1.5 HF snapshot.")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument(
        "--mask-some",
        action="store_true",
        help="Replace the last token of the prompt with <MASK> before forwarding. "
        "Makes the comparison exercise the MDM denoising path.",
    )
    args = parser.parse_args()

    setup_distributed_single_gpu()

    # ------------------------------------------------------------------
    # 1) Tokenize once on CPU
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    if args.mask_some:
        mask_id = 126336
        ids[0, -1] = mask_id
    print(f"prompt='{args.prompt}' -> tokens={ids[0].tolist()}")

    # ------------------------------------------------------------------
    # 2) HF reference forward pass
    # ------------------------------------------------------------------
    print("Loading HF reference...")
    hf_model = load_hf_reference(args.hf_path, device="cuda:0")
    hf_logits = hf_forward(hf_model, ids.to("cuda:0"))
    hf_vocab_size = hf_model.config.vocab_size
    print(f"HF logits shape={tuple(hf_logits.shape)} vocab={hf_vocab_size}")

    # Free HF model so we have headroom for Megatron model on the same GPU
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3) Megatron forward pass via the bridge
    # ------------------------------------------------------------------
    print("Building Megatron model via AutoBridge (loads weights)...")
    bridge, megatron_model = load_megatron_via_bridge(args.hf_path)
    mc_logits = megatron_forward(megatron_model, ids.to("cuda:0"))
    print(f"Megatron logits shape={tuple(mc_logits.shape)} dtype={mc_logits.dtype}")

    # ------------------------------------------------------------------
    # 4) Compare
    # ------------------------------------------------------------------
    metrics = compare_logits(hf_logits, mc_logits, hf_vocab_size=hf_vocab_size)
    print("\n=== Parity result ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    top5_overlap_int = int(metrics["top5_overlap_last_pos"].split("/")[0])
    ok = (
        metrics["cos_sim"] >= COS_SIM_THRESHOLD
        and metrics["rel_max_diff"] <= REL_MAX_DIFF_THRESHOLD
        and metrics["argmax_all_match"]
        and top5_overlap_int >= TOP5_OVERLAP_THRESHOLD
    )
    print(
        f"\nverdict: {'PASS' if ok else 'FAIL'}  "
        f"(cos>={COS_SIM_THRESHOLD}, rel_max_diff<={REL_MAX_DIFF_THRESHOLD}, "
        f"argmax match, top5_overlap>={TOP5_OVERLAP_THRESHOLD})"
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
