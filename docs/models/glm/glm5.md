# GLM-5

[GLM-5](https://huggingface.co/zai-org/GLM-5), [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1), [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2), and [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) use the shared `GLM5Bridge` for their MoE, Multi-Latent Attention, and Dynamic Sparse Attention architecture. GLM-5.3 shares GLM-5.2's architecture and import mappings; full-model GLM-5.3 verification remains pending.


## GLM-5.2 / GLM-5.3 compatibility

The pinned publisher configs for [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json) and [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/config.json) have identical architecture fields: 78 decoder layers, 256 routed experts with top-8 routing, MLA, and the same DSA IndexShare pattern. All 59,585 base checkpoint tensor names and shapes also match. AutoBridge resolves both to `GlmMoeDsaForCausalLM` and `GLM5Bridge`; no separate provider is needed. **GLM-5.3-Flash is a different architecture and is outside this support statement.**

The checkpoints have important differences:

- **Precision:** GLM-5.2 stores BF16 weights (plus FP32 router biases). GLM-5.3 stores 59,044 weights as E4M3 FP8 with FP32 scales per 128×128 block. The existing GLM bridge dequantizes these weights to BF16 on import. This is distinct from enabling FP8 training.
- **Config metadata:** GLM-5.3 adds `quantization_config` and records Transformers 5.15.0 instead of 5.12.0. Use the repository's supported dependency versions.
- **Tokenizer and generation defaults:** `tokenizer.json`, `tokenizer_config.json`, and `generation_config.json` are identical at those revisions, including token IDs and sampling defaults.
- **Chat formatting:** GLM-5.3's template adds low reasoning effort, retains historical reasoning by default, always opens a thinking response (it does not honor `enable_thinking=False`), and changes structured tool-result handling and ordering. Load the template from the GLM-5.3 checkpoint; do not substitute GLM-5.2's template.

For config/provider loading without weights:

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(
    "zai-org/GLM-5.3", revision="aca966e4e02791568aa6a4ced368624b3d897f42"
)
provider = bridge.to_megatron_provider(load_weights=False)
```

For **BF16 export** using the FP8 source as the reference, explicitly select `--export-weight-dtype bfloat16` with the GPU or distributed-CPU conversion backend (API: `weight_dtype=torch.bfloat16`). This removes FP8 scale tensors from strict source-key validation and writes a BF16 artifact rather than claiming a bitwise FP8 round trip. The serial-CPU backend does not support this option. MTP is disabled by default; the appended MTP layer is outside that default inference graph.

### Verification limits

The recorded commands and results below belong to the checkpoint named in each verification card. GLM-5.2 results do **not** establish GLM-5.3 full-checkpoint conversion/export, forward parity, generation, SFT/PEFT, resume, long-context behavior, or performance. Those GLM-5.3 checks remain unverified. The `glm52_*` recipes are pinned to GLM-5.2 and retain that name and attribution; changing only their display name would not select GLM-5.3 weights or its chat template.

[Production qualification tracker #5476](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/5476) remains open. It includes repeated-MTP training semantics, precision controls, dynamic context parallelism, production recipes, checkpoint continuity, convergence, and hardware-specific performance qualification. See also the open [export context-length fix #5996](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/5996), [router-bias fix #6036](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6036), and [recipe-selection fix #5608](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/5608).

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-glm5"></a>
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
        <button type="button" data-hardware="H100">H100</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="unverified" data-entry="glm5-hf-to-megatron-cpu" aria-controls="glm5-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="glm5-hf-to-megatron-gpu" aria-controls="glm5-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="glm5-megatron-to-hf-cpu" aria-controls="glm5-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="glm5-megatron-to-hf-gpu" aria-controls="glm5-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="glm5-pretrain-h100" aria-controls="glm5-pretrain-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="glm5-sft-h100" aria-controls="glm5-sft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="glm5-sft-long-context-h100" aria-controls="glm5-sft-long-context-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H100" data-status="unverified" data-entry="glm5-peft-h100" aria-controls="glm5-peft-h100" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H100</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="glm5-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="glm5-hf-to-megatron-cpu" tabindex="-1">
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
        <p>A CPU import of the pinned Hugging Face revision must complete every mapping and create a reloadable Megatron checkpoint. This workflow is deferred by this card.
</p>
      </section>
    </article>
    <article id="glm5-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="glm5-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-10</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 4 --gpus-per-node 8 --hf-model zai-org/GLM-5 --hf-revision 4e6698ba8e85059d749020e3c4d2123719f23926 --megatron-path work/model-verification/glm5/gpu-megatron --torch-dtype bfloat16 --tp 1 --pp 2 --ep 8 --etp 2 --distributed-timeout-minutes 60 --low-memory-save</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The pinned 32-H100 import exits successfully at TP1/PP2/EP8/ETP2, completes all 6,201 distributed mapping tasks, and persists a reloadable iter_0000000 checkpoint. Reload plus exact export projection covers all 59,079 tensors and 1,487,822,475,264 tensor-payload bytes in the 78-layer inference graph with matching keys, shapes, dtypes, and values. The 791 source tensors under model.layers.78 belong only to the intentionally disabled appended MTP auxiliary layer and are outside this item.
</p>
      </section>
    </article>
    <article id="glm5-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="glm5-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-26</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 4 --cpu-processes-per-node 8 --cpus-per-task 16 --mem 0 --exclusive --hf-model zai-org/GLM-5 --hf-revision 4e6698ba8e85059d749020e3c4d2123719f23926 --megatron-path work/model-verification/glm5/gpu-megatron/iter_0000000 --hf-path work/model-verification/glm5/cpu-hf-export --torch-dtype bfloat16 --tp 1 --pp 2 --ep 8 --etp 2 --distributed-timeout-minutes 240 --distributed-save --save-every-n-ranks 1 --no-progress</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 32-process distributed CPU export exits successfully and writes 280 safetensors shards. The exhaustive projection audit covers all 59,079 tensors and 1,487,822,475,264 tensor-payload bytes in the 78-layer inference graph with zero missing, unexpected, shape, dtype, or value mismatches. The 791 tensors under model.layers.78 belong only to the intentionally disabled appended MTP auxiliary layer and remain outside this item. Transformers 5.12.1 strictly reloads the output as GlmMoeDsaForCausalLM with 1,629 state tensors, 743,911,199,232 parameters, and no loading discrepancies.
</p>
      </section>
    </article>
    <article id="glm5-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="glm5-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-08-10</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 4 --gpus-per-node 8 --hf-model zai-org/GLM-5 --hf-revision 4e6698ba8e85059d749020e3c4d2123719f23926 --megatron-path work/model-verification/glm5/gpu-megatron/iter_0000000 --hf-path work/model-verification/glm5/gpu-hf-export --torch-dtype bfloat16 --tp 1 --pp 2 --ep 8 --etp 2 --distributed-timeout-minutes 60 --distributed-save --save-every-n-ranks 1</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 32-H100 distributed export exits successfully after all 6,201 mapping tasks and writes 280 safetensors shards. Of those, 278 are byte-for-byte identical to the pinned source shards; the two shards that also contain excluded MTP keys match all 63 common tensors exactly. The composite audit covers all 59,079 inference-graph tensors and 1,487,822,475,264 tensor-payload bytes with zero key, shape, dtype, or value mismatch. Transformers 5.12.1 strictly reloads the output as GlmMoeDsaForCausalLM with 1,629 state tensors, 743,911,199,232 parameters, and no loading discrepancies.
</p>
      </section>
    </article>
    <article id="glm5-pretrain-h100" class="verification-model-detail" data-entry-detail="glm5-pretrain-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H100</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
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
        <p>A public GLM-5 H100 recipe must complete a bounded 100-step run with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable final checkpoint. Training is deferred by this card.
</p>
      </section>
    </article>
    <article id="glm5-sft-h100" class="verification-model-detail" data-entry-detail="glm5-sft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H100</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
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
        <p>A pinned-data 100-step full-SFT run must finish with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable final checkpoint. Training is deferred by this card.
</p>
      </section>
    </article>
    <article id="glm5-sft-long-context-h100" class="verification-model-detail" data-entry-detail="glm5-sft-long-context-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · H100</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
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
        <p>A dedicated packed long-context SFT run must complete a bounded run with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable checkpoint. Training is deferred by this card.
</p>
      </section>
    </article>
    <article id="glm5-peft-h100" class="verification-model-detail" data-entry-detail="glm5-peft-h100" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H100</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H100</dd></div>
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
        <p>A GLM-5 PEFT recipe with an audited adapter target set must complete a bounded run with finite loss, all five metrics, and a reloadable adapter checkpoint. Training is deferred by this card.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
