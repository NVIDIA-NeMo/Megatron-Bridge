"""
Dataset types and providers for conversation-style VLM datasets.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoProcessor

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider

from .collate import COLLATE_FNS, default_collate_fn
from .makers import (
    make_cord_v2_dataset,
    make_cv17_dataset,
    make_medpix_dataset,
    make_rdr_dataset,
)


class VLMConversationDataset(torch.utils.data.Dataset):
    """Repeating wrapper over a list of HF-style conversation examples.

    - Each base example is expected to contain a "conversation" key following
      processor.apply_chat_template conventions. Optional modality fields like
      "audio" are passed through and consumed by the collate function.
    - Dataset length is set to a target length and indexes wrap around the
      underlying list to meet the requested size.
    - A `collate_fn` attribute is exposed so the framework can pass it to the
      DataLoader.
    """

    def __init__(
        self,
        base_examples: List[Dict[str, Any]],
        target_length: int,
        processor: Any,
        collate_impl: Optional[Callable[[list, Any], Dict[str, torch.Tensor]]] = None,
        start_of_response_token: Optional[Union[str, List[int]]] = None,
    ) -> None:
        assert isinstance(base_examples, list) and len(base_examples) > 0, "base_examples must be a non-empty list"
        self._base_examples = base_examples
        self._length = int(max(0, target_length))
        self._processor = processor
        # Choose collate implementation by processor type name when not provided
        collate_key = type(processor).__name__ if processor is not None else "default"
        selected_impl = collate_impl or COLLATE_FNS.get(collate_key, COLLATE_FNS["default"])  # type: ignore[index]

        def _bound_collate(batch: list) -> Dict[str, torch.Tensor]:
            # Some collate functions accept start_of_response_token as third arg
            if selected_impl is default_collate_fn:
                return selected_impl(batch, self._processor, start_of_response_token)  # type: ignore[arg-type]
            return selected_impl(batch, self._processor)  # type: ignore[call-arg]

        self.collate_fn = _bound_collate

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._length == 0:
            raise IndexError("Empty dataset")
        base = self._base_examples[idx % len(self._base_examples)]
        return base


@dataclass(kw_only=True)
class HFDatasetConversationProvider(DatasetProvider):
    """DatasetProvider that builds VLM conversation datasets from HF datasets.

    This provider leverages simple maker functions that return lists of examples
    with a "conversation" schema understood by model processors. It binds a
    HuggingFace `AutoProcessor` for the specified model and selects an
    appropriate collate function for batching.
    """

    # Required to match model.seq_length (enforced by ConfigContainer.validate)
    sequence_length: int

    # HF processor/model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
    hf_processor_path: str

    # Select which maker to use. Must match a function defined in makers module
    # like `make_rdr_dataset`, `make_cord_v2_dataset`, `make_medpix_dataset`, `make_cv17_dataset`.
    maker_name: str

    # Optional parameters forwarded to the selected maker
    maker_kwargs: Optional[Dict[str, Any]] = None

    # Optional collate override. If None, inferred from processor type.
    collate_impl: Optional[Callable[[list, Any], Dict[str, torch.Tensor]]] = None

    # Token or token-ids marking start of response for loss masking when supported
    start_of_response_token: Optional[Union[str, List[int]]] = None

    # Keep parity with GPTDatasetConfig usage in batching utilities
    skip_getting_attention_mask_from_dataset: bool = True

    # DataloaderConfig fields are inherited (num_workers, dataloader_type, etc.)

    def _get_maker(self) -> Callable[..., List[Dict[str, Any]]]:
        registry: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
            "make_rdr_dataset": make_rdr_dataset,
            "make_cord_v2_dataset": make_cord_v2_dataset,
            "make_medpix_dataset": make_medpix_dataset,
            "make_cv17_dataset": make_cv17_dataset,
        }
        if self.maker_name in registry:
            return registry[self.maker_name]
        # Allow passing function name alias without prefix, e.g., "rdr" -> make_rdr_dataset
        alias_map = {
            "rdr": "make_rdr_dataset",
            "cord_v2": "make_cord_v2_dataset",
            "medpix": "make_medpix_dataset",
            "cv17": "make_cv17_dataset",
        }
        if self.maker_name in alias_map and alias_map[self.maker_name] in registry:
            return registry[alias_map[self.maker_name]]
        raise ValueError(f"Unknown maker_name: {self.maker_name}")

    def _build_split_dataset(
        self,
        split: str,
        target_length: int,
        processor: Any,
    ) -> Optional[VLMConversationDataset]:
        if target_length <= 0:
            return None
        maker = self._get_maker()
        kwargs = dict(self.maker_kwargs or {})
        kwargs.setdefault("split", split)
        base_examples = maker(**kwargs)  # type: ignore[misc]
        if not isinstance(base_examples, list) or len(base_examples) == 0:
            raise ValueError(f"Maker '{self.maker_name}' returned no examples for split='{split}'")
        return VLMConversationDataset(
            base_examples=base_examples,
            target_length=target_length,
            processor=processor,
            collate_impl=self.collate_impl,
            start_of_response_token=self.start_of_response_token,
        )

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        # Bind processor for the requested model
        processor = AutoProcessor.from_pretrained(self.hf_processor_path, trust_remote_code=True)

        train_ds = self._build_split_dataset("train", context.train_samples, processor)
        valid_ds = self._build_split_dataset("validation", context.valid_samples, processor)
        test_ds = self._build_split_dataset("test", context.test_samples, processor)

        return train_ds, valid_ds, test_ds


