# Kimi K3

[Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) is a large sparse MoE model from Moonshot AI. Megatron Bridge supports the **language backbone** of the published multimodal checkpoint through the `KimiK3Bridge`.

The historical public-checkpoint results below were collected with the legacy
GPT implementation. They do not certify the new Hybrid path. Native migration
coverage and its remaining limitations are described here; the existing
[verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/kimi-k3/card.yaml)
retains those earlier results.

## Hybrid migration

`AutoBridge` selects Core's `HybridModel` with `KimiK3HybridModelConfig` and a
Bridge-owned Kimi stack supplied through the existing module-spec API. Each HF
block becomes two entries: KDA (`K`) or MLA (`+`), followed by dense MLP (`-`)
or MoE (`E`). Placement, dimensions, gates, normalization, vocabulary and AttnRes
boundaries come from HF. Recipes consume this shared configuration.

This is a **Bridge-only implementation**. It works with the unchanged pinned
Core main `9261f941cb0a1882b052cdaac950248121dcaf76`; no companion MLM change or
repository-wide pin update is required. The stack reuses Bridge's existing Kimi
attention, expert normalization, activation and AttnRes operations. Core owns
embeddings, the output/loss, pipeline allocation and distributed checkpoint
traversal. Performance optimization is outside this migration's current scope.

The stack passes the partial sum and snapshot bank as explicit differentiable
tensors through checkpoint boundaries. Complete HF blocks stay together on
pipeline stages; 93 blocks become 24/23/23/23 with PP4. Set
`model.pipeline_block_counts` for an explicit alternative. Native layer counts
are twice HF block counts; `kimi_attn_res_block_size` remains in HF block units.

### Checkpoints and recipes

Existing serialized `KimiK3ModelConfig` and `KimiK3ModelProvider` remain GPT-based
so they reconstruct old checkpoints. `KimiK3GPTBridge` retains their conversion
mapping. To migrate weights, restore the GPT checkpoint with its saved config,
export through `KimiK3GPTBridge` to HF, and import through `KimiK3Bridge`.
The Hybrid layout splits attention and FFN entries and moves the final AttnRes
head to the stack. A GPT distributed checkpoint cannot be loaded directly into
Hybrid. Migration starts a new optimizer/scheduler; old moments are not
converted. Published MXFP4 expert conversion retains the existing lossy
dequantize/requantize behavior.

`kimi_k3_pretrain_config()` obtains the model from `KIMI_K3_HF_PATH` and keeps
the HF tokenizer configuration. The explicitly named
`kimi_k3_pretrain_64gpu_gb300_bf16_proxy_config()` takes the first 24 attention
blocks and replaces their FFN entries with experts. For three-KDA/one-MLA
ordering this gives `KEKEKE+E` repeated six times, as in PR #7066. Reduced depth,
random initialization, mock data, NullTokenizer and forced routing are confined
to the proxy. Its BF16 Adam setup does not reproduce that PR's MXFP8/MoK/Muon
performance results. Neither recipe establishes full-scale fit or throughput.

### Numerical verification

Before/after tests compare the immutable GPT implementation and the Hybrid
candidate under the same unchanged Core/runtime, weights, inputs and actual
routing. The synthetic six-block fixture includes four KDA layers, two MLA
layers, a dense MLP, five routed/shared-expert layers and three AttnRes blocks.
Norm and KDA gate weights are nontrivial. Every logical parameter and persistent
buffer must have complete mapping and exact state/shape/dtype coverage. Tests
compare every parameter gradient, None/zero presence, logits, cross-entropy
loss and embedding gradients. FP32 uses `atol=2e-5, rtol=2e-4`; BF16 uses
`atol=2e-3, rtol=2e-2` plus a per-tensor relative-L2 limit of 0.02.

On two RTX 6000 Ada GPUs with unchanged Core main, FP32/BF16 packed tests,
BF16 full/selective recomputation, TP2 with sequence parallelism, EP2, and PP2
with 3/3 and 4/2 source-block splits produced exact forward/backward matches.
The PP full-recompute case crosses an AttnRes boundary and runs two scheduler
calls with four distinct microbatches each. A DP2 Core DDP/Adam update also
matched gradients, parameter groups, updated weights, FP32 masters and both
moments exactly. Same-topology TP2/EP2 checkpoints restored 195 local tensors
per rank; DP2 model/optimizer checkpoints restored 730 tensor values per rank.
These are bounded synthetic tests, not full-model convergence results.

Frozen-embedding block recomputation is checked against the GPT model with
recomputation disabled. The old nonquantized reentrant checkpoint path loses
early parameter gradients when none of its inputs require gradients; the
Hybrid stack runs eagerly until it has a differentiable checkpoint input.
The frozen nonrecompute control and this separate recompute check match exactly.
Use `--freeze-embeddings --recompute full --recompute-method block
--reference-recompute none` to run that comparison.

The standalone regression is manually runnable in a compatible CUDA/TE/FLA
environment:

```bash
NVIDIA_TF32_OVERRIDE=0 FLA_DISABLE_BACKEND_DISPATCH=1 \
uv run python -m torch.distributed.run --nproc_per_node=1 \
  tests/functional_tests/training/kimi_k3_hybrid_parity.py \
  --dtype bf16 --packed --report kimi-parity.json
```

Use `--dtype fp32` for the control and `--recompute full` or
`--recompute selective` for recomputation. Kernel choices are pinned for a
reproducible comparison. This is not a throughput measurement or a public HF
checkpoint execution. Historical verification-card results below remain GPT
results.

### Supported scope

The current scope is unquantized language-backbone training. CP/dynamic CP,
VPP, MTP, nonzero hidden dropout, CUDA graphs, offloading, alternate residual
streams and overlapped expert communication are rejected. Native cached
generation is not implemented; use the retained GPT configuration and
`KimiK3GPTBridge` for the legacy generation path. Multimodal execution, arbitrary
padding masks, quantized training and full-scale convergence are unverified.

TP greater than one with MoE requires sequence parallelism in the pinned Core,
which is incompatible with distributed saved activations. Do not combine these
settings.

Keep DDP gradient overlap disabled: the source model has unused first-attention
AttnRes parameters, which the pinned overlap implementation cannot handle across
iterations. Raw gradients remain `None`; Core DDP provides zero main-gradient
buffers to its optimizer. Custom name-based optimizer policies need translation
to split-entry names. Different construction order does not promise
seed-identical fresh initialization; parity starts from identical mapped weights.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-kimi-k3"></a>
### Run a configuration

Choose a workflow, precision, and exact recorded combination. The command and expected result update below.

<div class="verification-model-explorer" data-model-explorer>
  <div class="verification-model-controls" hidden>
    <div class="verification-capability-tabs" role="tablist" aria-label="Workflow">
      <button type="button" role="tab" aria-selected="true" data-capability-tab="import-export">Import & Export</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="pretrain">Pretrain</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="benchmark" disabled>Benchmark</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="sft">SFT</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="lora">LoRA</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="long-context">Long Context</button>
    </div>
    <div class="verification-filter-row">
      <div class="verification-precision-controls" aria-label="Precision filter">
        <span>Precision</span>
        <button type="button" class="is-active" data-precision="">All</button>
        <button type="button" data-precision="bf16">BF16</button>
        <button type="button" data-precision="fp8_mx">FP8 MX</button>
        <button type="button" data-precision="nvfp4">NVFP4</button>
      </div>
      <div class="verification-hardware-controls" aria-label="GPU filter">
        <span>GPU</span>
        <button type="button" class="is-active" data-hardware="">All</button>
        <button type="button" data-hardware="GB300">GB300</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="kimi-k3-hf-to-megatron-cpu" aria-controls="kimi-k3-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="kimi-k3-hf-to-megatron-gpu" aria-controls="kimi-k3-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="kimi-k3-megatron-to-hf-cpu" aria-controls="kimi-k3-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="kimi-k3-megatron-to-hf-gpu" aria-controls="kimi-k3-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="GB300" data-status="unverified" data-entry="kimi-k3-pretrain-gb300" aria-controls="kimi-k3-pretrain-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · GB300</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="GB300" data-status="unverified" data-entry="kimi-k3-sft-gb300" aria-controls="kimi-k3-sft-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · GB300</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="GB300" data-status="unverified" data-entry="kimi-k3-sft-long-context-gb300" aria-controls="kimi-k3-sft-long-context-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · GB300</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="GB300" data-status="unverified" data-entry="kimi-k3-peft-gb300" aria-controls="kimi-k3-peft-gb300" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · GB300</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="kimi-k3-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="kimi-k3-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>No CPU Hugging Face-to-Megatron conversion result is claimed. Future verification must import the complete pinned language backbone with the public conversion launcher, persist iter_0000000, reload it, and audit all mapped keys, shapes, dtypes, and values.
</p>
      </section>
    </article>
    <article id="kimi-k3-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="kimi-k3-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-11-06</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 12 --gpus-per-node 4 --hf-model moonshotai/Kimi-K3 --hf-revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 --megatron-path work/model-verification/kimi-k3/imported-megatron --torch-dtype bfloat16 --tp 2 --pp 3 --ep 8 --etp 2 --distributed-timeout-minutes 180 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>On 48x GB200, the pinned complete checkpoint imports at TP2/PP3/EP8/ETP2 and exits successfully. Published MXFP4 routed-expert weights are dequantized to BF16, and iter_0000000 is persisted with 48 distributed-checkpoint shards, metadata, and the recorded 93-layer run configuration. Exact Hugging Face round-trip parity is tracked separately by megatron_to_hf_gpu.
</p>
      </section>
    </article>
    <article id="kimi-k3-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="kimi-k3-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>No CPU Megatron-to-Hugging-Face conversion result is claimed. Future verification depends on a complete persisted Megatron import and must write, strictly reload, and exactly audit the exported Hugging Face language-backbone weights against the pinned source revision.
</p>
      </section>
    </article>
    <article id="kimi-k3-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="kimi-k3-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>No GPU Megatron-to-Hugging-Face conversion result is claimed. All 69 source KDA A_log tensors were audited: each stores 96 active heads plus 32 zero-valued inactive entries, for 2,208 checked padding values. Import validation drops only this zero padding, and export unit tests restore it with zeros. A four-layer diagnostic proxy completed low-memory GPU import and distributed streaming export. Its seven safetensors shards and index, totaling 58,482,941,904 bytes, were byte-identical to an export from the previous default-save import; this validates the low-memory I/O behavior, not full-model parity. Against the published proxy source, all 16,402 keys and shapes matched and A_log padding was valid. Three o_norm tensors changed from FP32 to BF16, and 16,128 value mismatches were concentrated in MXFP4 expert packed weights and scales because import dequantizes them to BF16 before export requantizes them. Full verification therefore still requires the pinned complete model to export and strictly reload, exact comparison of lossless tensors, and a declared numerical criterion for lossy MXFP4 reconstruction.
</p>
      </section>
    </article>
    <article id="kimi-k3-pretrain-gb300" class="verification-model-detail" data-entry-detail="kimi-k3-pretrain-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · GB300</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>None ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>None TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>None tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Training was intentionally excluded from this model-only verification pass. A future public Kimi K3 language-backbone recipe must complete a bounded 100-step pretrain run with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable final checkpoint.
</p>
      </section>
    </article>
    <article id="kimi-k3-sft-gb300" class="verification-model-detail" data-entry-detail="kimi-k3-sft-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · GB300</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>None ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>None TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>None tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Training was intentionally excluded from this model-only verification pass. A future public Kimi K3 language-backbone recipe must complete a pinned-data 100-step full-SFT run with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable final checkpoint.
</p>
      </section>
    </article>
    <article id="kimi-k3-sft-long-context-gb300" class="verification-model-detail" data-entry-detail="kimi-k3-sft-long-context-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · GB300</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>None ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>None TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>None tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Training was intentionally excluded from this model-only verification pass. A future public long-context SFT recipe must define supported sequence packing and context parallelism, complete 100 steps with finite loss and no skipped or NaN iterations, and record all five metrics.
</p>
      </section>
    </article>
    <article id="kimi-k3-peft-gb300" class="verification-model-detail" data-entry-detail="kimi-k3-peft-gb300" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · GB300</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>GB300</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>None ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>None TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>None tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Training was intentionally excluded from this model-only verification pass. A future Kimi K3 PEFT recipe with an audited adapter target set must complete 100 steps with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable adapter checkpoint.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
