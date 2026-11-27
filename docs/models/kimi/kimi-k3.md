# Kimi K3

[Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) is a sparse MoE model from
Moonshot AI. Megatron Bridge supports its **language backbone** through
`KimiK3Bridge`, `KimiK3ModelConfig` and the existing `BridgeHybridModelBuilder`.

| Implementation | Availability |
| --- | --- |
| **Bridge** | Default, unoptimized implementation using Core's `HybridModel` with Bridge-owned KDA, MLA, SiTU, latent MoE and attention residual layers. |
| **Optimized MCore components** | Available on the Megatron-Core `dev` branch. Integration and performance work are tracked in [#6451](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/6451). |

Use `AutoBridge.from_hf_pretrained(...).get_model_config()` to obtain the model
config. The shared builder constructs the model from the Kimi stack spec.
HF configuration determines the architecture and weight mapping; each HF block
maps to an attention layer followed by a dense or expert layer.

**EARLY UNOPTIMIZED — library recipe.**
`kimi_k3_pretrain_512gpu_gb300_bf16_config` lives in
`megatron.bridge.recipes.kimi.gb300.kimi_k3`; `kimi_k3_pretrain_config` remains
an alias. It follows [#6429](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6429)
with BF16/Adam, TP4/PP4/EP128, 8K sequences and complete-block recomputation.
It requires the pinned MCore `dev` stack and has no full-model performance or
convergence result. A small-model check on eight GB300 GPUs completed five
updates with TP2/PP2/EP2, 1K sequences and mock data, with finite gradients and
confirmed weight updates on every rank. This checks execution of the recipe's
settings, not full-model memory fit, throughput or convergence.

The library recipe carries over compatible settings from the performance
recipe: fused gradient accumulation and cross entropy, grouped expert GEMMs,
HybridEP dispatch, chunked Adam-state
offload, periodic garbage collection, expandable allocator segments and the
NCCL/Transformer Engine environment settings. Gradient reduction and parameter
gathers remain synchronous: the first attention-residual branch has unused score
parameters, and Core requires reduction overlap for parameter-gather overlap.
HF architecture and real expert routing are preserved.

MXFP8, MoK and FLA attention residuals require further integration with the
bridge's model components. Native KDA/MLA fusion and selective-recompute flags
are not consumed by those components. The standalone permutation-fusion flag
is not consumed by HybridEP dispatch. The library recipe therefore retains
BF16 and full-block recomputation instead of advertising inactive flags.

The separate performance recipe `kimi_k3_pretrain_64gpu_gb300_fp8mx_proxy_config`
is unchanged from Ze Yang’s [#7066](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/7066).
It requires the pinned MCore `dev` stack and uses MXFP8, MoK, FLA attention
residuals and Muon for a 24-block, random-init mock-data proxy with forced routing.

**Conversion scope:** language-backbone import/export, including MXFP4 expert
conversion and unchanged multimodal-weight passthrough during export.
The [verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/kimi-k3/card.yaml)
covers import/export only. Full-checkpoint conversion with this implementation
remains unverified; see the recorded status and acceptance criteria below.

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
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="kimi-k3-hf-to-megatron-gpu" aria-controls="kimi-k3-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
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
        <p>Import the complete pinned language backbone on CPU, persist and reload the Megatron checkpoint, and check every mapped key, shape, dtype and value. No complete CPU import result is recorded for this implementation.
</p>
      </section>
    </article>
    <article id="kimi-k3-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="kimi-k3-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
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
        <p>Import the complete pinned language backbone on GPU into HybridModel, dequantize MXFP4 routed-expert weights to BF16, and persist and reload the split-layer checkpoint. Check every mapped key, shape, dtype and value. Full-checkpoint import remains unverified for this implementation.
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
        <p>Export the complete persisted Megatron checkpoint on CPU and strictly reload the Hugging Face files. Check keys, shapes, dtypes and lossless values exactly, including restored zero padding in KDA A_log tensors. Declare a numerical tolerance for lossy MXFP4 reconstruction before comparing it. No complete CPU export result is recorded.
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
        <p>Export the complete persisted Megatron checkpoint on GPU and strictly reload the Hugging Face files. Check keys, shapes, dtypes and lossless values exactly, including restored zero padding in KDA A_log tensors. Declare a numerical tolerance for lossy MXFP4 reconstruction before comparing it. Small-model round trips do not establish full-checkpoint export verification.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
