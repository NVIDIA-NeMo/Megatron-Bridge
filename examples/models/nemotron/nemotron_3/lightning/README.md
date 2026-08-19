# Nemotron 3.5 Lightning

Nemotron 3.5 Lightning uses the same architecture as Nemotron 3 Nano, but its
checkpoint was trained natively with multi-token prediction (MTP).

Day-0 support for Nemotron 3.5 Lightning is available through the
`nvcr.io/nvidia/nemo:26.06.01` container plus the
[custom Megatron Bridge 0.5.1 branch and release README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/nemotron-3.5-lightning-mb-0.5.1/examples/models/nemotron/nemotron_3_5_lightning/README.md).
The model is also available on the
[`main`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main) and
[`r0.6.0`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/r0.6.0)
branches through the `nvcr.io/nvidia/nemo:26.08` container.

See the
[Nemotron 3.5 Lightning model verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/nemotron-3.5-lightning/card.yaml)
for verification scripts and results.

## Chat SFT formatting

Use the model's Hugging Face chat template through the direct HF SFT path. Nemotron 3.5 Lightning accepts
`reasoning_content` alongside assistant `content` and uses OpenAI-style top-level `tools` plus assistant
`tool_calls`. The official template renders thinking and tool calls in the format used to post-train the model;
do not replace it with a generic ChatML or Hermes tool template.

Per-row template controls belong under `chat_template_kwargs`:

```json
{
  "messages": [
    {"role": "user", "content": "What is 1 + 1?"},
    {"role": "assistant", "reasoning_content": "Add the two values.", "content": "2"}
  ],
  "chat_template_kwargs": {
    "enable_thinking": true,
    "truncate_history_thinking": true
  }
}
```

`enable_thinking` selects the prefix for a new generation. For completed SFT assistant turns, include or omit
`reasoning_content` to represent thinking or non-thinking data. `truncate_history_thinking=true`, the official
default, removes earlier reasoning text while retaining its empty `<think></think>` boundary; set it to `false`
to preserve reasoning from all historical assistant turns. The assistant thinking prefix `<think>\n` and an empty
`<think></think>` pair are masked. For non-empty reasoning, the reasoning text and closing `</think>` token are
supervised, along with final-answer and structured tool-call tokens. System, user, and tool-response tokens are masked.
