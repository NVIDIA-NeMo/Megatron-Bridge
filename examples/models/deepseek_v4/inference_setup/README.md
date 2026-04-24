# DeepSeek-V4-Flash inference on cw

Pragmatic path to coherent output: use DeepSeek's own `inference/` code with
their FP8/FP4 tilelang kernels. This directory holds the wrapper scripts; the
actual DS scripts are fetched from the HF repo at setup time.

## Status — ✅ coherent inference achieved

With tilelang 0.1.9 and the `fast_hadamard_transform` shim, an 8×H100 torchrun
of the converted MP=8 checkpoint produces coherent text on a chat prompt.
Sample on 2026-04-24:

```
Prompt: Write a short poem about the moon.

Completion: A silver coin in velvet night,
A ghostly ship of silent light.
It pulls the tide with patient grace,
And paints a shadow on your face.

A constant watch, a steady gleam,
The keeper of a dreamer's dream.
```

## State on cw (as of branch e0eac0fb-plus)

1. **HF checkpoint present** —
   `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/yuya/HF_HOME/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6e763230a9d263eca2023f1d4a5ce1bfe126cf48`
   (149 GB, 46 FP8/FP4 safetensors + tokenizer files). Downloaded from a
   compute-node egress IP because the login node hit HF XET rate limits.

2. **Converted MP=8 checkpoint present** —
   `/lustre/fsw/portfolios/coreai/users/yuya/dsv4_mp8`
   (24 GB, `model{0..7}-mp8.safetensors` + tokenizer). Produced by
   `inference/convert.py --n-experts 256 --model-parallel 8`.

3. **DS inference scripts staged** —
   `/lustre/fsw/portfolios/coreai/users/yuya/dsv4_infer/{inference,encoding}/*.py`
   (fetched verbatim from the HF repo).

## Gotchas encountered and resolved

1. **Wrong HF snapshot directory.** `huggingface_hub` cached two snapshot
   commits (`6e763230…` and `6c858e71…`); only the *older* one had all 46
   safetensors shards symlinked in. Using the newer snapshot pulled in only
   7 shards → `convert.py` produced ~24 GB of MP-sharded weights containing
   only layers 0–5, no LM head, no MTP. Generation then emitted pure
   `<|begin_of_sentence|>` / gibberish because the output-layer weights
   didn't load. Fix: point `--hf-ckpt-path` at the snapshot dir with 46
   symlinks. Expected MP=8 output is ~165 GB.

2. **`tilelang==0.1.8` crashes with**
   `AttributeError: '_NestedLoopCheckVisitor' object has no attribute '_inst'`
   during JIT. **0.1.9 works** on the same container (mbridge-260321, Python
   3.12, torch 2.11).

3. **`fast_hadamard_transform` wheel fails to compile** in-container (nvcc
   toolchain mismatch). The pure-PyTorch shim in this directory is a drop-in;
   put it on `PYTHONPATH` and DS's `from fast_hadamard_transform import
   hadamard_transform` resolves to it.

4. **Each `srun` starts a fresh container.** `pip install` does not persist
   between `srun` invocations. `run_generate.sh` re-installs on every run.

## Files in this directory

| File | Purpose |
|------|---------|
| `run_generate.sh` | Wraps the `srun` + pip-install + torchrun invocation. Ready to run when the tilelang blocker clears. |
| `fast_hadamard_transform_shim.py` | Pure-PyTorch Fast Walsh-Hadamard stand-in; fast_hadamard_transform's CUDA wheel fails to compile in-container. Copy to `fast_hadamard_transform.py` on the PYTHONPATH. |

## Repro command once the env is fixed

```bash
# from cw:aa, inside salloc
srun --container-image=/lustre/fsw/portfolios/coreai/users/yuya/containers/mbridge-260321.sqsh \
     --container-mounts=/lustre:/lustre --no-container-mount-home \
     --gpus-per-task=8 --pty \
     bash /lustre/fsw/portfolios/coreai/users/yuya/dsv4_infer/run_generate.sh
```
