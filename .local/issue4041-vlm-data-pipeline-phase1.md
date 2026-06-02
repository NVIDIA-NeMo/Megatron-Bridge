# Issue #4041 Phase 1: VLM Data Pipeline Current-State Map

Owner: Mira

Issue: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/4041

Scope: report only. No refactor started and no local tests run.

## Source Anchors Read

- `src/megatron/bridge/data/energon/base_energon_datamodule.py`
- `src/megatron/bridge/data/energon/energon_provider.py`
- `src/megatron/bridge/data/energon/hf_encoder_task_encoder.py`
- `src/megatron/bridge/data/energon/task_encoder_utils.py`
- `src/megatron/bridge/recipes/qwen_vl/data/energon/task_encoder.py`
- `src/megatron/bridge/data/vlm_datasets/conversation_dataset.py`
- `src/megatron/bridge/data/vlm_datasets/hf_provider.py`
- `src/megatron/bridge/data/vlm_datasets/hf_dataset_makers.py`
- `src/megatron/bridge/data/vlm_datasets/mock_provider.py`
- `src/megatron/bridge/data/vlm_datasets/preloaded_provider.py`
- `src/megatron/bridge/data/vlm_datasets/collate.py`
- `src/megatron/bridge/data/vlm_datasets/token_utils.py`
- `src/megatron/bridge/data/vlm_datasets/step37_flickr8k/*`
- `src/megatron/bridge/training/vlm_step.py`
- `src/megatron/bridge/training/gpt_step.py`
- `src/megatron/bridge/training/utils/packed_seq_utils.py`
- `src/megatron/bridge/training/utils/padding_utils.py`
- `src/megatron/bridge/training/utils/visual_inputs.py`
- `src/megatron/bridge/training/config.py`
- `src/megatron/bridge/models/transformer_config.py`
- `src/megatron/bridge/models/qwen_vl/qwen3_vl_step.py`
- Selected model forward signatures for Qwen2.5-VL / Qwen3-VL / GLM-4.5V.
- Existing focused unit tests under `tests/unit_tests/data/energon/`, `tests/unit_tests/data/vlm_datasets/`, and `tests/unit_tests/training/test_vlm_step.py`.

Skill context loaded before analysis:

- `skills/nemo-mbridge-perf-sequence-packing/SKILL.md`
- `skills/testing/SKILL.md`

## Responsibility Map

| Responsibility | Current file / function / class | Energon path behavior | HF path behavior | Duplication / layering issue | Proposed post-refactor owner |
|---|---|---|---|---|---|
| Loading | `EnergonProvider.build_datasets`; `EnergonMultiModalDataModule.datasets_provider`; `ChatMLWebdataset` | Webdataset samples decode into `ChatMLSample` with JSON conversation plus optional `imgs`, `videos`, `audio`. Worker config uses pure DP rank so CP ranks see identical samples. | `HFDatasetConversationProvider.build_datasets` loads examples through maker functions; `MockVLMConversationProvider` synthesizes examples; `PreloadedVLMConversationProvider` parses local JSON/JSONL into conversation examples. | Mostly source-specific and acceptable. Providers expose the same `pack_sequences_in_batch` flag, but not all pass it consistently into `VLMConversationDataset`. | Keep source-specific dataset providers. Normalize their output into a shared "raw VLM sample" adapter before processing. |
| Conversation normalization | `cook_chatml_sample`; `QwenVLTaskEncoder.encode_sample`; `HFEncoderVLMTaskEncoder.encode_sample`; HF maker functions in `hf_dataset_makers.py`; collates in `collate.py` | Energon normalizes GPT-style or OpenAI-style JSON via `cook_chatml_sample`. Qwen Energon then rewrites user content around `<image>` / `<video>` placeholders. Generic Energon rewrites `<image>` placeholders to structured content for processor templates. | HF makers already emit structured `{"role", "content"}` conversations. Some collates pass them directly to `apply_chat_template`; Nemotron/Kimi perform custom rewrites. | Same semantic step is split across maker functions, Energon encoders, and HF collates. Placeholder policy differs by model and source. | Shared processor-layer normalizer with source adapters: Energon adapter parses `ChatMLSample`; HF adapter accepts already-structured examples. Model-specific placeholder hooks stay explicit. |
| Processing: chat template, tokenize, vision processor | `QwenVLTaskEncoder.encode_sample`; `process_vision`; `HFEncoderVLMTaskEncoder.encode_sample`; `qwen2_5_collate_fn`; `default_collate_fn`; `ministral3_collate_fn`; `glm4v_collate_fn`; `kimi_k25_vl_collate_fn`; `nemotron_*_collate_fn` | Energon does per-sample processing before batch: PIL conversion, chat template/tokenization, vision processor call, visual-token expansion or visual tensor collection. | HF path does processing inside collate on a list of raw examples; model-specific collates call processor/apply_chat_template differently. | This is the main Step-1 target: the same model processing is implemented once in Energon task encoders and again in HF collates. HF path combines processing with batching. | New shared VLM processing component with `encode_sample` / `encode_example` returning per-sample tensors and modality metadata. Energon task encoders and HF collates call this component. |
| Assistant masking | `find_pattern_indices`; `QwenVLTaskEncoder.encode_sample`; `HFEncoderVLMTaskEncoder.encode_sample`; `create_multiturn_loss_mask_by_search`; `qwen2_audio_collate_fn` | Energon searches tokenized assistant answer text inside `input_ids`. Qwen asserts a match; generic Energon silently leaves unmatched spans masked. | HF collates use `create_multiturn_loss_mask_by_search` with a few string variants. Qwen2-Audio uses a backward search. No path currently uses HF `return_assistant_tokens_mask`. | Fragile duplicate text-search masking. Shift timing also varies by path. This should be centralized in Step 1 but not semantically replaced until Step 3. | Shared mask builder preserving current behavior first, behind one API. Step 3 can replace internals with generation-tag or token-boundary masking. |
| Label / loss-mask construction | `QwenVLTaskEncoder.encode_sample` and `batch`; `HFEncoderVLMTaskEncoder.encode_sample`; collate functions in `collate.py` | Energon builds labels/targets per sample, then batch pads and masks labels to `IGNORE_INDEX`. Qwen expands visual placeholders before final target shift. | HF collates build labels from `input_ids[:, 1:]`, shift masks, mask skipped tokens, then attach `loss_mask`. | Repeated shift/mask logic with model-specific exceptions. A Step-1 central processor needs to preserve current shifted-label convention exactly. | Shared processing output should include `input_ids`, `labels`, and `loss_mask` with one documented next-token convention. |
| Visual metadata / tensors | `Qwen2_5_VLVisualInputs`; `GenericVisualInputs`; `QwenVLTaskEncoder.encode_batch`; `HFEncoderVLMTaskEncoder.encode_batch`; all VLM collates | Energon wraps visual tensors into `Qwen2_5_VLVisualInputs` or `GenericVisualInputs`. Qwen expands placeholders and stores `image_grid_thw` / `video_grid_thw`. | HF collates wrap processor outputs. Generic wrapper can include `pixel_values`, `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`, `image_sizes`, `image_position_ids`, `mm_token_type_ids`. Qwen wrapper normalizes 5D tensors and 3D grid tensors. | `visual_inputs.normalized_for_model()` is the current implicit Bridge-internal contract. Model-specific required metadata is not declared in one place. | Keep `visual_inputs` as the boundary for now, but define each processor's output contract explicitly in the shared processor class. |
| Batching / collation | `QwenVLTaskEncoder.batch`; `HFEncoderVLMTaskEncoder.batch`; `VLMConversationDataset.collate_fn`; collates in `collate.py` | Energon task encoder batches processed samples and pads to batch max length. It builds attention mask and position IDs. | HF `VLMConversationDataset` selects collate by processor type; collate processes and batches in one function. Most collates rely on processor padding; Kimi/Nemotron have custom padding/token expansion. | Processing and batching are intertwined in HF collates. Energon has a cleaner `encode_sample` / `batch` split but still owns model-specific processing. | After Step 1, shared processing owns per-sample encode. Existing collates/task encoders can continue owning batching until Step 2. |
| Padding | Energon task `batch`; HF collate processor padding; `vlm_step.get_batch`; `padding_utils.py`; GLM forward alignment | Energon pads to current microbatch max. Qwen/generic Energon replace pad token IDs with 0 for model input. | HF collates use processor padding or custom right-padding. Some paths create a 2D attention mask. | `vlm_step.get_batch` pads again: fixed `cfg.model.seq_length` for PP/EP, otherwise ceil-to-128 capped by model seq length. This causes the `pad in collator -> pad/truncate again in step` layer leak. GLM forward pads `mm_token_type_ids` because `vlm_step` pads `input_ids` but not all metadata. | Step 1 should not move padding yet. Step 2 should move final padding policy into collate/batching and make metadata padding explicit. |
| In-batch packing | `vlm_step.pack_batch_sequences`; `vlm_step.get_batch`; `nemotron_omni_collate_fn(pack_sequences=True)`; `VLMConversationDataset.__init__` pack binding; `EnergonProvider.build_datasets` pack flag handoff | Generic Energon task encoders do not implement collate packing. `EnergonProvider` only toggles `task_encoder.pack_sequences` if the task encoder exposes that attribute. `vlm_step` still reads `cfg.dataset.pack_sequences_in_batch` and can pack padded Energon batches at runtime. | `HFDatasetConversationProvider` passes `pack_sequences_in_batch` to `VLMConversationDataset`; dataset raises unless the selected collate accepts `pack_sequences`. Currently this is supported by `nemotron_omni_collate_fn`, not by the common Qwen/default/GLM/Ministral collates. `vlm_step` also reads the same flag. | One config flag has multiple meanings and owners. Runtime packing in `vlm_step` removes padding and concatenates only within the current microbatch. It is not offline sequence packing. Collate-level packing plus `vlm_step` packing could collapse sample boundaries if both run on the same step path. | Step 1 should leave packing unchanged. Step 2 should choose one batching/packing owner: the VLM collate/batch layer. Long-term offline packing should be separate from in-batch packing. |
| Packed metadata | `vlm_step.pack_batch_sequences`; `get_packed_seq_params`; `nemotron_omni_collate_fn`; `training/gpt_step.py` THD CP guard | Energon runtime packing builds `cu_seqlens` and `max_seqlen` in `vlm_step`, not in task encoder, except adjacent Omni-specific code. | HF runtime packing same as Energon when it reaches `vlm_step`. Nemotron Omni collate can emit `cu_seqlens`, `cu_seqlens_unpadded`, argmins, and `max_seqlen`. | `vlm_step.get_batch` ignores preexisting collator `cu_seqlens` when `enable_packing=True` and creates fresh packed metadata. `vlm_step` creates padded `cu_seqlens` under CP/SP but does not emit separate `cu_seqlens_unpadded`. | Batching owner should emit final packed metadata once. `vlm_step` should only route it into `PackedSeqParams`. |
| CP / SP constraints | `vlm_step.get_batch`; `training/config.py`; `models/transformer_config.py`; `training/gpt_step.py` | CP ranks read identical data because Energon worker config uses DP rank. Runtime packing pads each sequence to `lcm(2*CP, CP*TP if sequence_parallel else 1)`. | Same runtime constraint applies after HF collate. Config only enforces `micro_batch_size > 1` for `pack_sequences_in_batch`. | CP/SP divisibility is enforced late in `vlm_step`, not validated at dataset/collate time. Packed THD batches are expected to have microbatch size 1 after packing. | Step 2 batching owner should compute CP/SP-aware pad multiples. Config validation should remain a guard but not be the sole owner. |
| Position IDs / M-RoPE | `get_ltor_masks_and_position_ids`; collates creating `position_ids`; `vlm_step.forward_step`; Qwen2.5-VL / Qwen3-VL / GLM model forwards | Energon batches create ordinary 2D position IDs. Qwen2.5/GLM model forwards recompute 3D M-RoPE from `input_ids`, media token IDs, and grid metadata. | HF collates create ordinary 2D position IDs. GLM passes `mm_token_type_ids` through `GenericVisualInputs`; Qwen2.5 computes type IDs from `input_ids`; Qwen3 has a separate step that deliberately sets `position_ids=None` and recomputes in model. | Data path emits 2D position IDs, but several VLM models ignore or replace them. Padding/packing before model forward can affect M-RoPE unless all required visual metadata and masks are padded/sliced consistently. | Shared processing should declare whether it emits plain 2D positions or only enough metadata for model-side M-RoPE. Step 1 should not change M-RoPE behavior. |
| Model-specific forward args | `vlm_step.forward_step`; `visual_inputs.normalized_for_model`; model forward signatures | Energon `encode_batch` supplies `visual_inputs`; `vlm_step` moves contained tensors to CUDA and expands to forward kwargs. | HF collates also supply `visual_inputs`; `vlm_step` treats both sources identically after batch fetch. | `forward_args.update(visual_inputs.normalized_for_model())` is broad and implicit. Different models require different keys: Qwen uses grids and pixel tensors; GLM needs `mm_token_type_ids`; Gemma may use `image_position_ids`; Kimi maps `grid_thws` to `image_grid_thw`. | Shared processor classes should own and document model-specific `visual_inputs` construction. `vlm_step` should remain a router. |
| Adjacent legacy packed VLM path | `data/vlm_datasets/step37_flickr8k/*`; separate Step37 forward step | Not an Energon path. | Uses a dedicated packed dataloader, identity collate, and per-step image preprocess. | It is a separate packed-sample pipeline and should not be confused with `vlm_step` in-batch packing. | Leave out of Step 1 unless Step37 is explicitly brought into scope. |

## Call-Flow Diagrams

### Energon VLM Path

Generic flow:

```text
Config.dataset = EnergonProvider(...)
  -> EnergonProvider.build_datasets(context)
     -> if pack_sequences_in_batch and task_encoder has pack_sequences:
          task_encoder.pack_sequences = True
     -> EnergonMultiModalDataModule(...)
        -> train_dataloader()
           -> _build_worker_config() using DP rank, CP rank only for logging
           -> get_train_dataset(path, batch_size=micro_batch_size, task_encoder=...)
           -> get_savable_loader(...)
           -> EnergonDataloader cyclic iterator
              -> task_encoder.encode_sample(ChatMLSample)
              -> task_encoder.batch(list[TaskSample])
              -> task_encoder.encode_batch(TaskBatch)
              -> dict consumed by vlm_step.get_batch()
```

Qwen Energon processing:

```text
ChatMLSample(conversation, imgs, videos)
  -> QwenVLTaskEncoder.encode_sample()
     -> _images_to_pil / _videos_to_pil
     -> process_vision(image_processor, images, videos)
        -> pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
     -> cook_chatml_sample()
     -> convert_to_qwenvl_content() for user placeholders
     -> tokenizer.apply_chat_template(..., tokenize=True)
     -> find_pattern_indices() for assistant answer tokens
     -> expand image/video placeholder token count from grid_thw / merge_size
     -> build image_input_mask / video_input_mask and shifted target
  -> QwenVLTaskEncoder.batch()
     -> pad to microbatch max length
     -> replace pad token with 0 in input ids
     -> labels IGNORE_INDEX and loss_mask
     -> get_ltor_masks_and_position_ids()
     -> stack visual tensors and grids
  -> encode_batch()
     -> attach Qwen2_5_VLVisualInputs
```

Generic HF-encoder Energon processing:

```text
ChatMLSample(conversation, imgs, videos)
  -> HFEncoderVLMTaskEncoder.encode_sample()
     -> _images_to_pil / _videos_to_pil
     -> cook_chatml_sample()
     -> convert <image> text into structured content parts
     -> processor.apply_chat_template(..., tokenize=False)
     -> processor(text=..., images=..., videos=..., return_tensors="pt")
     -> find_pattern_indices() for assistant answer text
     -> build shifted labels/loss_mask
     -> truncate to seq_length
     -> remove partial image token blocks when truncation splits a block
     -> collect configured visual_keys
  -> HFEncoderVLMTaskEncoder.batch()
     -> pad to microbatch max length
     -> replace pad token with 0 in input ids
     -> get_ltor_masks_and_position_ids()
     -> concatenate visual tensors
  -> encode_batch()
     -> attach GenericVisualInputs
```

### HF VLM Path

```text
Config.dataset = HFDatasetConversationProvider(...)
  -> build_datasets(context)
     -> AutoProcessor.from_pretrained(hf_processor_path)
     -> maker(**maker_kwargs)
        -> list[{"conversation": structured turns, optional modality payloads}]
     -> VLMConversationDataset(base_examples, processor, collate_impl, pack_sequences)
        -> __getitem__ returns raw example only
        -> collate_fn selected from COLLATE_FNS[type(processor).__name__]
           -> qwen2_5_collate_fn / default_collate_fn / ministral3_collate_fn /
              glm4v_collate_fn / kimi_k25_vl_collate_fn / nemotron_*_collate_fn / ...
              -> apply_chat_template and/or processor(...)
              -> processor padding or custom padding
              -> create_multiturn_loss_mask_by_search()
              -> shifted labels and loss_mask
              -> position_ids if processor did not provide them
              -> wrap visual tensors in Qwen2_5_VLVisualInputs or GenericVisualInputs
              -> dict consumed by vlm_step.get_batch()
```

Provider variants:

```text
MockVLMConversationProvider
  -> synthesizes base_examples
  -> VLMConversationDataset(..., collate_impl=None)
  -> currently does not pass pack_sequences_in_batch into VLMConversationDataset

PreloadedVLMConversationProvider
  -> parses local JSON/JSONL records into conversations
  -> VLMConversationDataset(..., processor=processor)
  -> currently does not pass pack_sequences_in_batch into VLMConversationDataset
```

### `vlm_step.get_batch` / `forward_step`

```text
forward_step(state, data_iterator, model)
  -> get_pg_collection(model)
  -> get_batch(data_iterator, cfg, pg_collection)
     -> is_pp_first_stage / is_pp_last_stage
     -> get_batch_from_iterator()
        -> next(data_iterator)
        -> move required tensors to CUDA
        -> move visual_inputs container tensors to CUDA
        -> keep labels/loss_mask only on last PP stage
        -> preserve 2D attention_mask as _padding_mask for packing length detection
     -> enable_packing = cfg.dataset.pack_sequences_in_batch
     -> if not enable_packing:
          if PP or EP:
             pad/truncate input_ids, labels, loss_mask, position_ids, attention_mask to cfg.model.seq_length
          else:
             pad/truncate to ceil(current_seq_len, 128), capped at cfg.model.seq_length
        else:
          pad_multiple = lcm(2*CP if CP>1 else 1,
                             CP*TP if sequence_parallel and TP>1 else 1)
          pack_batch_sequences()
             -> infer per-row lengths from _padding_mask or token != 0
             -> concatenate rows into [1, total_len]
             -> pad each row length to pad_multiple
             -> build cu_seqlens and max_seqlen
          -> attention_mask = None
     -> return tokens, labels, loss_mask, attention_mask, position_ids,
        cu_seqlens, max_seqlen, visual_inputs
  -> forward_args = {
       input_ids, position_ids, attention_mask, labels, loss_mask
     }
  -> if visual_inputs:
       forward_args.update(visual_inputs.normalized_for_model())
  -> if cu_seqlens:
       forward_args["packed_seq_params"] = get_packed_seq_params(...)
  -> model(**forward_args)
  -> loss_function = create_masked_next_token_loss_function(loss_mask, ...)
```

## Risks And Unknowns

### Assistant Masking

- Both Energon and HF paths search tokenized assistant text in the final `input_ids`. This can fail with leading-space tokens, chat-template control tokens, BOS/EOS behavior, tokenizer version changes, or assistant content that appears earlier in the prompt.
- The paths differ on failure behavior:
  - Qwen Energon asserts the assistant span is found.
  - Generic Energon leaves missing spans masked.
  - HF collates warn only when the entire mask is zero.
  - Qwen2-Audio uses a backward search and does not mask skipped tokens the same way as other VLM collates.
- Shift timing must be preserved during Step 1. Several paths build a mask on unshifted input IDs, then shift it to align with next-token labels.
- HF `apply_chat_template(..., return_assistant_tokens_mask=True)` is not currently used. Adopting it belongs to Step 3, but Step 1 should centralize current mask construction so Step 3 has one replacement point.

### Image / Video Metadata

- `image_grid_thw` and `video_grid_thw` are both vision-encoder inputs and M-RoPE inputs. If processing or padding changes, grid row count must continue to match media placeholder count.
- Truncation behavior differs:
  - Generic Energon can replace partial image-token blocks with pad and slice visual tensors.
  - Qwen Energon drops samples when visual tokens alone exceed sequence length and otherwise warns about text truncation.
  - HF collates mostly rely on processor truncation and do not share one visual-token truncation policy.
- GLM carries `mm_token_type_ids` in `GenericVisualInputs`, and model forward pads it because `vlm_step` pads `input_ids` later. Moving padding out of `vlm_step` later must explicitly pad this metadata with the same target length.
- Kimi maps processor `grid_thws` into `Qwen2_5_VLVisualInputs.image_grid_thw` after pre-expanding image placeholder tokens.
- Gemma/Ministral generic visual inputs can include `image_position_ids` and `image_sizes`; these should not be dropped by a Qwen-centric processor abstraction.

### M-RoPE / Position IDs

- Data collators and Energon batchers emit ordinary 2D `position_ids`, but Qwen2.5-VL, Qwen3-VL, GLM, Ernie-style models compute 3D M-RoPE position IDs in model forward from `input_ids`, media token IDs, and grid metadata.
- `vlm_step.forward_step` currently passes 2D `position_ids` and then also passes visual kwargs. Some model forwards ignore or replace the data `position_ids`; others may not.
- Qwen3-VL has a separate `qwen3_vl_step.py` that deliberately recomputes position IDs in model forward and preserves original `input_ids` for packing/CP handling. Generic `vlm_step.py` is not equivalent.
- If Step 2 later moves padding/packing into collate, M-RoPE parity needs checks for:
  - padded non-packed BSHD batches,
  - packed THD batches,
  - PP/EP fixed sequence shapes,
  - media metadata padded or sliced consistently with tokens.

### CP And Packed THD Constraints

- Per the sequence-packing skill, VLM in-batch packing is different from offline packed SFT. It only concatenates samples within the current microbatch.
- Config validation enforces `micro_batch_size > 1` for `pack_sequences_in_batch=True`, but CP/SP pad multiples are computed only at runtime in `vlm_step`.
- `vlm_step.pack_batch_sequences` pads each per-sample packed segment to:
  - `2 * context_parallel_size` when CP is enabled,
  - `context_parallel_size * tensor_parallel_size` when sequence parallel is also enabled,
  - `lcm` of both constraints when both apply.
- Length detection falls back to `tokens != 0` unless a 2D `_padding_mask` is preserved. Energon batches replace pad tokens with 0, so any real token ID 0 would be ambiguous in fallback mode.
- Existing collate-emitted packed metadata is not the same contract as `vlm_step` runtime packing. If a collate emits `cu_seqlens` and `vlm_step` also packs, original sample boundaries can be lost.
- Packed THD CP slicing elsewhere expects the effective packed batch to have microbatch size 1. `vlm_step` satisfies this after packing by returning `[1, total_len]`.

## Recommended Step-1 Plan: Unify Processing Only

Do not move padding, batching, in-batch packing, CP slicing, or `vlm_step.py` behavior in Step 1.

1. Add a shared VLM processing module under `src/megatron/bridge/data/`, for example `vlm_processing.py` or `vlm_datasets/processing.py`.
2. Define a small output object, tentatively `VLMProcessedSample`, with:
   - `input_ids`
   - `labels`
   - `loss_mask`
   - optional `attention_mask`
   - optional plain `position_ids`
   - visual tensor/metadata payload or a `visual_inputs` object
   - source metadata such as sample key where needed
3. Define source adapters:
   - Energon adapter: `ChatMLSample -> normalized conversation + PIL images/videos`.
   - HF adapter: structured example dict -> normalized conversation + modality payloads.
4. Move shared logic first, without changing semantics:
   - conversation normalization,
   - `apply_chat_template` / processor invocation,
   - current search-based assistant mask,
   - shifted label construction,
   - skipped-token label masking,
   - visual tensor wrapper construction.
5. Keep model-specific hooks explicit:
   - Qwen image/video placeholder expansion and `image_grid_thw` / `video_grid_thw`,
   - Kimi media token pre-expansion,
   - GLM `mm_token_type_ids`,
   - Gemma/Ministral `image_position_ids` / generic visual fields.
6. Convert one low-risk pair first:
   - `HFEncoderVLMTaskEncoder` and `default_collate_fn`/`qwen2_5_collate_fn` should call the same shared processor.
   - Keep `QwenVLTaskEncoder` as a model-specific wrapper until parity tests prove the shared processor handles its placeholder expansion.
7. Leave `VLMConversationDataset`, provider pack flags, and `vlm_step.get_batch` unchanged in Step 1 except for wiring them to call the shared processor.
8. Document the shared processor contract in code: which model-specific forward kwargs are produced and whether positions are plain 2D positions or deferred to model-side M-RoPE.

## Suggested Tests To Add Later

Do not add these in phase 1.

- Unit: same simple image+assistant example through Energon `ChatMLSample` and HF structured example produces identical `input_ids`, `labels`, `loss_mask`, and visual metadata after shared processing.
- Unit: multi-turn conversation with two assistant spans preserves current shifted label/loss-mask convention.
- Unit: assistant text with leading/trailing whitespace and repeated answer text documents current text-search behavior before Step 3 changes masking.
- Unit: no-image, single-image, multi-image, and mixed image/text batches preserve `image_grid_thw` row counts and `visual_inputs.normalized_for_model()` shapes.
- Unit: truncation that cuts a visual-token block is either handled consistently or rejected consistently per model.
- Unit: GLM-style `mm_token_type_ids` and Gemma-style `image_position_ids` survive shared processing and batching.
- Unit: `pack_sequences_in_batch` flag behavior for `HFDatasetConversationProvider`, `MockVLMConversationProvider`, and `PreloadedVLMConversationProvider` is explicit and consistent.
- Unit: `vlm_step.pack_batch_sequences` with `_padding_mask` vs `tokens != 0` length detection, including a row containing token ID 0 as real content if the tokenizer can produce it.
- Unit: CP/SP pad multiple calculation for packed VLM batches: CP-only, SP-only, and CP+SP `lcm` cases.
- Integration or functional later: small Qwen2.5-VL / GLM M-RoPE parity check after Step 1 and again after Step 2. Prefer unit tests first; functional tests should be limited to the relevant model and at most 2 GPUs.

## Post-Refactor Validation Plan

Do not run these jobs during phase 1. There is no refactor branch yet, so there is no baseline/refactor pair to compare.

Goal after implementation: verify that the VLM data refactor does not change training loss for one HF-style VLM dataset path and one Energon-style VLM dataset path. Run the baseline commit and refactor branch with the same container image, same model checkpoint, same dataset, same seed, same overrides, and same process count.

### Validation Matrix

| Check | Baseline branch | Refactor branch | Dataset path exercised | Candidate recipe / step | Determinism controls | Iterations and loss samples | Expected tolerance |
|---|---|---|---|---|---|---|---|
| HF-style VLM data | Pre-refactor commit, usually `main` or the parent commit of the refactor branch | Refactor branch under review | `HFDatasetConversationProvider` -> `VLMConversationDataset` -> `qwen2_5_collate_fn` via processor type `Qwen3VLProcessor` | `scripts/training/run_recipe.py --recipe qwen3_vl_8b_peft_config --dataset vlm-hf --step_func qwen3_vl_step` | `rng.seed=1234`; fixed model/processor source; fixed pretrained checkpoint; fixed HF dataset revision/cache; `dataset.maker_name=rdr`; `dataset.pack_sequences_in_batch=false`; same `train.global_batch_size`, `train.micro_batch_size`, TP/PP/CP/EP | 10 train iterations, `logger.log_interval=1`; compare first 5 logged `lm loss` values and also the full 10-value series if all are present | Expected exact or near-exact match. Flag if any pair differs by `abs > 1e-5` and `rel > 1e-5`; any `abs > 1e-4`, NaN, skipped iteration, or missing value is a regression unless separately explained. |
| Energon-style VLM data | Same baseline commit as above | Same refactor branch as above | `QwenVLEnergonProvider` -> `QwenVLTaskEncoder` -> Energon dataloader -> `qwen3_vl_step` | `scripts/training/run_recipe.py --recipe qwen3_vl_8b_peft_energon_config --dataset vlm-energon --step_func qwen3_vl_step` | `rng.seed=1234`; same model/processor source; same pretrained checkpoint; same prebuilt Energon shard directory; `dataset.pack_sequences_in_batch=false`; same task encoder knobs; same `train.global_batch_size`, `train.micro_batch_size`, TP/PP/CP/EP | 10 train iterations, `logger.log_interval=1`; compare first 5 logged `lm loss` values and also the full 10-value series if all are present | Same tolerance as HF check. If Energon sample order is not stable, first rerun baseline twice to establish baseline noise; do not accept the refactor until baseline-vs-refactor is no worse than baseline-vs-baseline. |

Notes:

- Start with `dataset.pack_sequences_in_batch=false` to isolate processing-layer parity for Step 1. If the implemented refactor also moves batching/packing, add a second matrix row for each dataset path with `dataset.pack_sequences_in_batch=true`, `train.micro_batch_size=2`, and the CP/SP pad-multiple settings relevant to the changed code.
- Prefer `dataset.maker_name=rdr` for the HF check because the built-in `make_rdr_dataset` formatting is deterministic. Avoid `make_cord_v2_dataset` for loss parity unless a fixed preprocessed subset is used, because that maker chooses among multiple parses with Python `random.choice`.
- The Energon check should use one prebuilt shard directory for both baseline and refactor. `examples/models/qwen/qwen3_vl/prepare_mantis_energon.py` is the repo-native converter candidate for creating Qwen-VL-compatible Energon shards from a prepared Mantis-Instruct-style source.
- The selected Qwen3-VL PEFT recipe functions hardcode the model source as `Qwen/Qwen3-VL-8B-Instruct`. If Morgan/Yu require a local model snapshot for fully offline cluster execution, use an identical private runner or approved recipe update on both baseline and refactor; do not compare a hardcoded-source run against a local-snapshot run.

### Candidate Commands

These are sanitized command templates. Replace placeholders privately when running; do not put expanded internal paths in a public issue comment.

Shared setup inside the container for each worktree:

```bash
export UV_CACHE_DIR=<shared_uv_cache>
cd <baseline_or_refactor_worktree>
uv sync --locked
```

If launching multiple local ranks from a scheduler allocation with a bind-mounted source tree, sync once per node before `uv run`:

```bash
if [ "${LOCAL_RANK_INDEX:-0}" = "0" ]; then
  uv sync --locked
fi
```

HF-style command candidate:

```bash
uv run python -m torch.distributed.run --nproc_per_node=<nproc> scripts/training/run_recipe.py \
  --recipe qwen3_vl_8b_peft_config \
  --dataset vlm-hf \
  --step_func qwen3_vl_step \
  --peft_scheme lora \
  checkpoint.pretrained_checkpoint=<pretrained_checkpoint> \
  checkpoint.save=null \
  checkpoint.load=null \
  train.train_iters=10 \
  train.global_batch_size=<global_batch_size> \
  train.micro_batch_size=<micro_batch_size> \
  validation.eval_interval=null \
  validation.eval_iters=null \
  scheduler.lr_warmup_iters=1 \
  scheduler.lr_decay_iters=10 \
  logger.log_interval=1 \
  logger.tensorboard_dir=<run_output_dir>/tb_logs \
  logger.wandb_project=null \
  rng.seed=1234 \
  dataset.maker_name=rdr \
  dataset.hf_processor_path=Qwen/Qwen3-VL-8B-Instruct \
  dataset.pack_sequences_in_batch=false \
  model.tensor_model_parallel_size=<tp> \
  model.pipeline_model_parallel_size=<pp> \
  model.context_parallel_size=<cp> \
  model.expert_model_parallel_size=1
```

Energon-style command candidate:

```bash
uv run python -m torch.distributed.run --nproc_per_node=<nproc> scripts/training/run_recipe.py \
  --recipe qwen3_vl_8b_peft_energon_config \
  --dataset vlm-energon \
  --step_func qwen3_vl_step \
  --peft_scheme lora \
  checkpoint.pretrained_checkpoint=<pretrained_checkpoint> \
  checkpoint.save=null \
  checkpoint.load=null \
  train.train_iters=10 \
  train.global_batch_size=<global_batch_size> \
  train.micro_batch_size=<micro_batch_size> \
  validation.eval_interval=null \
  validation.eval_iters=null \
  scheduler.lr_warmup_iters=1 \
  scheduler.lr_decay_iters=10 \
  logger.log_interval=1 \
  logger.tensorboard_dir=<run_output_dir>/tb_logs \
  logger.wandb_project=null \
  rng.seed=1234 \
  dataset.path=<energon_dataset_root> \
  dataset.pack_sequences_in_batch=false \
  model.tensor_model_parallel_size=<tp> \
  model.pipeline_model_parallel_size=<pp> \
  model.context_parallel_size=<cp> \
  model.expert_model_parallel_size=1
```

Recommended initial parallelism: use the smallest fixed `<nproc>`, TP, PP, and CP that fits the selected checkpoint on the allocated GPUs. Keep the exact same values for baseline and refactor. Start with CP=1 and packing disabled; add CP/packing rows only when the refactor touches those owners.

### Loss Collection And Comparison

Use `logger.log_interval=1` so every train iteration prints one loss line. Capture stdout/stderr to one log file per run.

Collect `lm loss` values from log lines matching:

```text
iteration ... | ... lm loss: <float> |
```

Comparison procedure:

1. Parse the baseline and refactor logs into ordered `(iteration, lm_loss)` pairs.
2. Require at least 5 matched training iterations; prefer all 10.
3. Fail immediately on missing `lm loss`, NaN/Inf, nonzero `number of nan iterations`, or unexpected nonzero skipped iterations.
4. Compare by iteration number. For each pair compute absolute and relative differences.
5. Flag regression when any value exceeds `abs > 1e-5` and `rel > 1e-5`. Escalate anything above `abs > 1e-4` even if a rerun is planned.
6. If nondeterminism is suspected, run baseline twice with the exact same command and compare baseline-vs-baseline before accepting a wider tolerance.

Minimal parser shape for the private validation run:

```python
import math
import re
from pathlib import Path

LOSS_RE = re.compile(r"iteration\s+(\d+)/\s+\d+.*?lm loss:\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)")

def read_losses(path: str) -> list[tuple[int, float]]:
    pairs = []
    for line in Path(path).read_text().splitlines():
        match = LOSS_RE.search(line)
        if match:
            value = float(match.group(2))
            if not math.isfinite(value):
                raise ValueError(f"Non-finite loss in {path}: {value}")
            pairs.append((int(match.group(1)), value))
    return pairs
```

### Cluster Execution Notes

- Run inside the project container, not on a bare host.
- Use `uv sync --locked` in each worktree before the run. Use a shared `UV_CACHE_DIR` on persistent storage so baseline and refactor use the same wheel cache and do not fill the container cache.
- If using a bind-mounted source tree with multiple local ranks, sync once per node before launching the distributed command, then use `uv run python -m torch.distributed.run ...`.
- Keep baseline and refactor jobs on the same GPU type and same process topology.
- Disable checkpoint saves for parity runs unless a checkpoint side effect is explicitly being tested.
- Store full logs, resolved command lines, git SHAs, and environment metadata privately. Public comments must not include internal cluster names, accounts, hostnames, job IDs, or full filesystem paths.

### Private Report Format For Morgan / Yu

When jobs are eventually run, send a private report with:

```text
Issue: #4041 VLM data refactor loss parity

Baseline:
- git SHA:
- worktree:
- container image:
- uv command:
- exact training command:
- log file:

Refactor:
- git SHA:
- worktree:
- container image:
- uv command:
- exact training command:
- log file:

HF-style validation:
- recipe / step:
- dataset maker and revision/cache note:
- model checkpoint:
- seed:
- nproc / TP / PP / CP / EP:
- first 10 lm loss values, baseline:
- first 10 lm loss values, refactor:
- max abs diff:
- max rel diff:
- result: PASS / FAIL

Energon-style validation:
- recipe / step:
- Energon dataset shard location:
- model checkpoint:
- seed:
- nproc / TP / PP / CP / EP:
- first 10 lm loss values, baseline:
- first 10 lm loss values, refactor:
- max abs diff:
- max rel diff:
- result: PASS / FAIL

Sanitized public summary:
- HF-style VLM path: PASS/FAIL, loss tolerance summary only.
- Energon-style VLM path: PASS/FAIL, loss tolerance summary only.
- No internal paths, cluster names, accounts, hostnames, or job IDs.
```

## Risky Files / Functions For Step 1

- `src/megatron/bridge/data/vlm_datasets/collate.py`
  - Many model-specific collates repeat processing and masking. A broad edit here can affect Qwen, GLM, Gemma, Ministral, Kimi, Nemotron, and Qwen2-Audio at once.
- `src/megatron/bridge/recipes/qwen_vl/data/energon/task_encoder.py`
  - Qwen Energon expands visual placeholders from grid metadata and builds labels before/after expansion. This is easy to break.
- `src/megatron/bridge/data/energon/hf_encoder_task_encoder.py`
  - Generic Energon has partial visual-token truncation handling that HF collates do not share.
- `src/megatron/bridge/data/vlm_datasets/conversation_dataset.py`
  - It currently binds collate implementation and optionally overloads `pack_sequences`; touching it can change pack flag behavior.
- `src/megatron/bridge/training/vlm_step.py`
  - Should not be modified in Step 1. It remains the current owner of final padding and runtime in-batch packing until Step 2.
- `src/megatron/bridge/training/utils/visual_inputs.py`
  - This is the implicit model-forward contract. Changes here affect multiple model families.

## Sanitized GitHub Issue Comment Draft

Phase 1 current-state map is ready locally as `.local/issue4041_vlm_data_pipeline_phase1.md`.

High-signal findings:

- The core duplication is between Energon task encoders (`QwenVLTaskEncoder`, `HFEncoderVLMTaskEncoder`) and HF collates in `data/vlm_datasets/collate.py`; both own chat templating, processor calls, assistant masking, labels, and visual metadata wrapping.
- `vlm_step.py` still owns final padding and runtime in-microbatch THD packing, so batches can be padded in the collator and then padded/truncated or unpacked/repacked again in the step.
- `pack_sequences_in_batch` currently has multiple owners: provider/dataset collate binding, optional collate-level packing for some paths, and runtime packing in `vlm_step.py`.
- Assistant masking is still text-search based across the main paths and should be centralized in Step 1 without changing semantics; replacing it with generation-tag or token-boundary masking should remain Step 3.
- M-RoPE-sensitive metadata (`image_grid_thw`, `video_grid_thw`, `mm_token_type_ids`, `image_position_ids`) is passed through an implicit `visual_inputs.normalized_for_model()` contract. Step 1 should make that contract explicit, but not change model-side M-RoPE behavior.

Recommended Step 1: introduce a shared VLM processing component used by both Energon task encoders and HF collates, limited to per-sample conversation normalization, processor invocation, current masking/label construction, and visual metadata packaging. Leave batching, padding, in-batch packing, CP constraints, and `vlm_step.py` behavior unchanged until Step 2.
