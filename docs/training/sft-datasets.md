# SFT Dataset Support

This guide describes the dataset formats supported by Megatron Bridge for
**Supervised Fine-Tuning (SFT)**, how loss masking works (so the model only
learns from assistant responses), and how the available configuration
options map to the equivalents in Hugging Face TRL.

For PEFT-specific guidance (LoRA, DoRA) on top of these formats, see
[`peft.md`](peft.md). For sequence-packing details, see
[`packed-sequences.md`](packed-sequences.md).

## Loss masking: the `answer_only_loss` flag

The single most important knob for SFT is whether loss is computed over the
entire example or only over the assistant's response.

`create_sft_dataset` (and the underlying `GPTSFTDataset`, `GPTSFTChatDataset`,
and packed variants) takes an `answer_only_loss: bool = True` argument.

| Value | Effect |
|---|---|
| `True` (default) | `loss_mask` is `1` on the answer tokens and `0` on the prompt / system / user tokens. The model only learns from the assistant's output. |
| `False` | `loss_mask` is `1` on every non-pad token. The model is trained to reproduce the prompt as well as the answer. |

The Bridge default matches what users typically want when migrating from
TRL's `DataCollatorForCompletionOnlyLM` or the implicit assistant-only mask
produced by `apply_chat_template(..., return_assistant_tokens_mask=True)`.

## Supported dataset formats

### 1. Standard JSONL (prompt / response)

Each line is a JSON object with at least an `input` field and an `output`
field:

```jsonl
{"input": "Translate to French: Hello, how are you?", "output": "Bonjour, comment ça va ?"}
{"input": "What is 2 + 2?", "output": "4"}
```

Configure via `create_sft_dataset`:

```python
from megatron.bridge.data.datasets.sft import create_sft_dataset

dataset = create_sft_dataset(
    path="path/to/train.jsonl",
    tokenizer=tokenizer,
    seq_length=2048,
    answer_only_loss=True,                # mask everything except `output`
    label_key="output",                   # which field is the assistant response
    truncation_field="input",             # truncate the prompt if needed
    prompt_template="{input} {output}",   # how prompt + response are joined
)
```

The `prompt_template` is an `f-string`-style template that determines how
the input and output fields are concatenated before tokenization.

### 2. Hugging Face chat format

Standard HF multi-turn messages — each example is a list of `{role, content}`
turns. Bridge applies the tokenizer's chat template and uses it to produce
the loss mask. Enable this with `chat=True`:

```python
dataset = create_sft_dataset(
    path="path/to/chat.jsonl",
    tokenizer=tokenizer,
    seq_length=2048,
    chat=True,
    use_hf_tokenizer_chat_template=True,  # delegate templating to the tokenizer
    answer_only_loss=True,
)
```

Example JSONL line:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."}
  ]
}
```

With `use_hf_tokenizer_chat_template=True`, Bridge calls
`tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)` so
the loss mask comes directly from the tokenizer's chat template — no
hand-rolled regex. The chat template must support the
`{% generation %}{% endgeneration %}` block for this to work; most modern
instruct tokenizers (Llama 3 Instruct, Qwen2/3 Instruct, Gemma 2/3 Instruct,
etc.) ship with one.

### 3. ShareGPT-style conversations

ShareGPT-format datasets list turns under a `conversations` key with
`from` / `value` instead of `role` / `content`. Convert to the HF chat
format above before training:

```python
def sharegpt_to_hf(example):
    return {
        "messages": [
            {"role": {"human": "user", "gpt": "assistant"}[turn["from"]],
             "content": turn["value"]}
            for turn in example["conversations"]
        ]
    }
```

You can plug this into an `HFDatasetConfig` as `process_example_fn`, or
preprocess once to disk and use the standard chat-format path.

### 4. Packed sequences

For throughput on small / variable-length SFT samples, pre-pack many
examples into each sequence up to `seq_length`. Bridge supports two
on-disk packed formats:

| Path pattern | Class chosen |
|---|---|
| `*.npy` | `GPTSFTPackedDataset` (legacy NumPy memmap) |
| `*.parquet`, `*.pq`, directory, glob → `*.parquet` | `GPTSFTPackedParquetDataset` |

Pass `pack_metadata_file_path` (and optionally `pad_cu_seqlens=True` for
CUDA-graph compatibility) when constructing the dataset. The packing
arithmetic, the `pad_seq_to_mult` rounding, and the difference between
histogram-key (pre-pad) and stored-tensor-length (post-pad) are described
in [`packed-sequences.md`](packed-sequences.md).

Loss masking continues to work per-example inside the pack — each example's
prompt tokens still get `loss_mask=0` and the assistant span gets `1`.

### 5. Hugging Face Hub datasets via `HFDatasetConfig`

When the dataset is on the Hub, use `HFDatasetConfig` so Bridge handles the
download, caching, and per-example processing:

```python
from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.hf_processors.squad import process_squad_example

dataset_cfg = HFDatasetConfig(
    dataset_name="rajpurkar/squad",
    process_example_fn=process_squad_example,
    val_proportion=0.05,
    split_val_from_train=True,
)
```

The `process_example_fn` is the bridge between the dataset's native schema
and Bridge's `{input, output}` (or chat `messages`) format. Three reference
processors are shipped in `megatron.bridge.data.hf_processors`:

- `process_squad_example` — SQuAD-style `(context, question, answers)` →
  `Context: ... Question: ... Answer:`
- `process_gsm8k_example` — math word problems with `####`-delimited final
  answer
- `process_openmathinstruct2_example` — OpenMathInstruct-2 multi-turn
  reasoning

These are pure dict-in / dict-out functions — write your own for any
dataset that doesn't fit, no tokenizer or I/O is required.

## Common SFT configuration knobs

These are the `create_sft_dataset` arguments most users override:

| Argument | Purpose | Typical value |
|---|---|---|
| `answer_only_loss` | Mask everything but the assistant span | `True` |
| `label_key` | Which JSON field is the assistant response | `"output"` |
| `prompt_template` | F-string joining prompt fields | `"{input} {output}"` |
| `truncation_field` | Which field to truncate when over `seq_length` | `"input"` |
| `truncation_method` | Where to truncate (`"left"` or `"right"`) | `"right"` |
| `add_bos` / `add_eos` | Whether to add BOS/EOS tokens around the example | model-specific |
| `pad_to_max_length` | Pre-pad every example to `seq_length` | `False` unless required by the kernel |
| `chat` | Switch to `GPTSFTChatDataset` | `True` for multi-turn |
| `use_hf_tokenizer_chat_template` | Use the tokenizer's chat template | `True` for modern instruct models |

## Migrating from Hugging Face TRL

Users coming from TRL will find these equivalences useful:

| TRL concept | Megatron Bridge equivalent |
|---|---|
| `SFTTrainer(..., dataset_text_field="text")` with a custom collator | `create_sft_dataset(..., prompt_template="...", label_key=...)` |
| `DataCollatorForCompletionOnlyLM(response_template=...)` | `answer_only_loss=True` (default) |
| `apply_chat_template(..., return_assistant_tokens_mask=True)` | `chat=True, use_hf_tokenizer_chat_template=True` |
| Multi-turn ShareGPT loading | Convert to HF chat format (see above) |
| Packing via TRL's `packing=True` | `GPTSFTPackedParquetDataset` (parquet) or `GPTSFTPackedDataset` (`.npy`) — see [`packed-sequences.md`](packed-sequences.md) |
| `max_seq_length` | `seq_length` |
| `dataset_num_proc` | `memmap_workers` |

The main shape difference: TRL exposes a single `SFTTrainer` with collator
plug-ins, while Bridge selects between purpose-built dataset classes
(`GPTSFTDataset` / `GPTSFTChatDataset` / packed variants) based on
constructor args. The factory function `create_sft_dataset` hides that
selection — pass the format-specific args and you get the right class back.

## Where to look next

- [`peft.md`](peft.md) — applying LoRA / DoRA on top of any of these formats
- [`packed-sequences.md`](packed-sequences.md) — packing internals, the
  `pad_seq_to_mult` knob, and the histogram-key vs. stored-tensor-length
  distinction
- [`training-loop-settings.md`](training-loop-settings.md) — how the SFT
  dataset plugs into the main training loop
- [`../recipe-usage.md`](../recipe-usage.md) — how the family recipes (e.g.
  `llama3_8b_sft_config`) wire all of this together end-to-end
