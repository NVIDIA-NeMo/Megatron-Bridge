# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    ProcessorMixin,
)
from transformers.generation.utils import GenerateOutput

from megatron.bridge.models.hf_pretrained.base import PreTrainedBase
from megatron.bridge.models.hf_pretrained.safe_config_loader import safe_load_config_with_retry


# Type variable for generic model type
VLMType = TypeVar("VLMType", bound=PreTrainedModel)


class PreTrainedVLM(PreTrainedBase, Generic[VLMType]):
    """
    A generic class for Pretrained Vision-Language Models with lazy loading.

    Allows type-safe access to specific VLM implementations like LlavaForConditionalGeneration.

    Examples:
        Basic usage with image and text:
        >>> from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
        >>> from PIL import Image
        >>>
        >>> # Create instance - no model loading happens yet
        >>> vlm = PreTrainedVLM.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>>
        >>> # Load an image
        >>> image = Image.open("cat.jpg")
        >>>
        >>> # Process image and text together - processor and model load here
        >>> inputs = vlm.process_images_and_text(
        ...     images=image,
        ...     text="What do you see in this image?"
        ... )
        >>>
        >>> # Generate response
        >>> outputs = vlm.generate(**inputs, max_new_tokens=100)
        >>> print(vlm.decode(outputs[0], skip_special_tokens=True))

        Batch processing with multiple images:
        >>> # Process multiple images with questions
        >>> images = [Image.open(f"image_{i}.jpg") for i in range(3)]
        >>> questions = [
        ...     "What is the main object in this image?",
        ...     "Describe the scene",
        ...     "What colors do you see?"
        ... ]
        >>>
        >>> # Process batch
        >>> inputs = vlm.process_images_and_text(
        ...     images=images,
        ...     text=questions,
        ...     padding=True
        ... )
        >>>
        >>> # Generate responses
        >>> outputs = vlm.generate(**inputs, max_new_tokens=50)
        >>> for i, output in enumerate(outputs):
        ...     print(f"Image {i+1}: {vlm.decode(output, skip_special_tokens=True)}")

        Using specific VLM types with type hints:
        >>> from transformers import LlavaForConditionalGeneration
        >>> from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
        >>>
        >>> # Type-safe access to Llava-specific features
        >>> llava: PreTrainedVLM[LlavaForConditionalGeneration] = PreTrainedVLM.from_pretrained(
        ...     "llava-hf/llava-1.5-7b-hf",
        ...     torch_dtype=torch.float16,
        ...     device="cuda"
        ... )
        >>>
        >>> # Access model-specific attributes
        >>> vision_tower = llava.model.vision_tower  # Type-safe access

        Text-only generation (for multimodal models that support it):
        >>> # Some VLMs can also work with text-only inputs
        >>> text_inputs = vlm.encode_text("Explain what a neural network is.")
        >>> outputs = vlm.generate(**text_inputs, max_length=100)
        >>> print(vlm.decode(outputs[0], skip_special_tokens=True))

        Custom preprocessing and generation:
        >>> # Load with custom settings
        >>> vlm = PreTrainedVLM.from_pretrained(
        ...     "Qwen/Qwen-VL-Chat",
        ...     trust_remote_code=True,
        ...     device_map="auto",
        ...     load_in_4bit=True
        ... )
        >>>
        >>> # Custom generation config
        >>> from transformers import GenerationConfig
        >>> vlm.generation_config = GenerationConfig(
        ...     max_new_tokens=200,
        ...     temperature=0.8,
        ...     top_p=0.95,
        ...     do_sample=True
        ... )
        >>>
        >>> # Process with custom parameters
        >>> inputs = vlm.process_images_and_text(
        ...     images=image,
        ...     text="<image>\\nDescribe this image in detail.",
        ...     max_length=512
        ... )

        Manual component setup:
        >>> # Create empty instance
        >>> vlm = PreTrainedVLM()
        >>>
        >>> # Load components separately
        >>> from transformers import AutoProcessor, AutoModel
        >>> vlm.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base")
        >>> vlm.model = AutoModel.from_pretrained("microsoft/Florence-2-base")
        >>>
        >>> # Use for various vision tasks
        >>> task_prompt = "<OD>"  # Object detection task
        >>> inputs = vlm.process_images_and_text(images=image, text=task_prompt)
        >>> outputs = vlm.generate(**inputs)

        Conversational VLM usage:
        >>> # Multi-turn conversation with images
        >>> conversation = []
        >>>
        >>> # First turn
        >>> image1 = Image.open("chart.png")
        >>> inputs = vlm.process_images_and_text(
        ...     images=image1,
        ...     text="What type of chart is this?"
        ... )
        >>> response = vlm.generate(**inputs)
        >>> conversation.append(("user", "What type of chart is this?"))
        >>> conversation.append(("assistant", vlm.decode(response[0])))
        >>>
        >>> # Follow-up question
        >>> follow_up = "What is the highest value shown?"
        >>> # Format conversation history + new question
        >>> full_prompt = format_conversation(conversation) + f"\\nUser: {follow_up}"
        >>> inputs = vlm.process_images_and_text(images=image1, text=full_prompt)
        >>> response = vlm.generate(**inputs)
    """

    ARTIFACTS = ["processor", "tokenizer", "image_processor"]
    OPTIONAL_ARTIFACTS = ["generation_config"]

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Initialize a Pretrained VLM with lazy loading.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            device: Device to load model on (e.g., 'cuda', 'cpu')
            torch_dtype: Data type to load model in (e.g., torch.float16)
            trust_remote_code: Whether to trust remote code when loading
            **kwargs: Additional arguments passed to component loaders
        """
        self._model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        super().__init__(**kwargs)

    def _load_model(self) -> VLMType:
        """Lazy load and return the model."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load model")

        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            **self.init_kwargs,
        }

        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype

        # Use provided config if already loaded
        config = getattr(self, "_config", None)
        if config is not None:
            model_kwargs["config"] = config

        # Try AutoModel first for VLMs
        model = AutoModel.from_pretrained(self.model_name_or_path, **model_kwargs)

        # Move to device
        model = model.to(self.device)

        # Set generation config if available
        generation_config = getattr(self, "_generation_config", None)
        if generation_config is not None and hasattr(model, "generation_config"):
            model.generation_config = generation_config
        return model

    def _load_config(self) -> AutoConfig:
        """Lazy load and return the model config with thread-safety protection."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load config")

        return safe_load_config_with_retry(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            **self.init_kwargs,
        )

    def _load_processor(self) -> ProcessorMixin:
        """Lazy load and return the processor."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load processor")

        try:
            return AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **self.init_kwargs,
            )
        except Exception:
            # Some VLMs might not have a processor, fall back to manual loading
            raise ValueError(
                f"Could not load processor for {self.model_name_or_path}. "
                "This model might require manual processor setup."
            )

    def _load_tokenizer(self) -> Optional[PreTrainedTokenizer]:
        """
        Lazy load and return the tokenizer.
        For VLMs, the tokenizer might be included in the processor.
        """
        # Check if tokenizer is available through processor first
        processor = getattr(self, "_processor", None)
        if processor is not None and hasattr(processor, "tokenizer"):
            return processor.tokenizer

        # Try to load tokenizer separately
        if self.model_name_or_path is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                    **self.init_kwargs,
                )

                # Set padding token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                return tokenizer
            except Exception:
                # Some VLMs include tokenizer only in processor
                pass
        return None

    def _load_image_processor(self) -> Optional[Any]:
        """
        Lazy load and return the image processor.
        For VLMs, the image processor might be included in the processor.
        """
        # Check if image processor is available through processor first
        processor = getattr(self, "_processor", None)
        if processor is not None and hasattr(processor, "image_processor"):
            return processor.image_processor

        # Try to load image processor separately
        if self.model_name_or_path is not None:
            try:
                return AutoImageProcessor.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                    **self.init_kwargs,
                )
            except Exception:
                # Some VLMs include image processor only in processor
                pass
        return None

    def _load_generation_config(self) -> Optional[GenerationConfig]:
        """Lazy load and return the generation config."""
        if self.model_name_or_path is not None:
            try:
                return GenerationConfig.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=self.trust_remote_code,
                    **self.init_kwargs,
                )
            except Exception:
                # Not all models have generation configs
                pass
        return None

    @property
    def model_name_or_path(self) -> Optional[Union[str, Path]]:
        """Return the model name or path."""
        return self._model_name_or_path

    @property
    def model(self) -> VLMType:
        """Lazy load and return the underlying model."""
        if not hasattr(self, "_model"):
            self._model = self._load_model()
        else:
            # Ensure model is on the right device when accessed
            if hasattr(self._model, "device") and hasattr(self._model.device, "type"):
                current_device = str(self._model.device)
                target_device = str(self.device)
                if current_device != target_device:
                    self._model = self._model.to(self.device)
        return self._model

    @model.setter
    def model(self, value: VLMType):
        """Set the model manually."""
        self._model = value

    @property
    def processor(self) -> ProcessorMixin:
        """Lazy load and return the processor."""
        if not hasattr(self, "_processor"):
            self._processor = self._load_processor()
        return self._processor

    @processor.setter
    def processor(self, value: ProcessorMixin):
        """Set the processor manually."""
        self._processor = value

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizer]:
        """Lazy load and return the tokenizer."""
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: PreTrainedTokenizer):
        """Set the tokenizer manually."""
        self._tokenizer = value

    @property
    def image_processor(self) -> Optional[Any]:
        """Lazy load and return the image processor."""
        if not hasattr(self, "_image_processor"):
            self._image_processor = self._load_image_processor()
        return self._image_processor

    @image_processor.setter
    def image_processor(self, value: Any):
        """Set the image processor manually."""
        self._image_processor = value

    @property
    def generation_config(self) -> Optional[GenerationConfig]:
        """Lazy load and return the generation config."""
        if not hasattr(self, "_generation_config"):
            self._generation_config = self._load_generation_config()
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: GenerationConfig):
        """Set the generation config manually."""
        self._generation_config = value
        # Update model's generation config if model is loaded
        if hasattr(self, "_model") and self._model is not None and hasattr(self._model, "generation_config"):
            self._model.generation_config = value

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Additional initialization kwargs."""
        return self.init_kwargs

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "PreTrainedVLM[VLMType]":
        """
        Create a PreTrainedVLM instance for lazy loading.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            device: Device to load model on
            torch_dtype: Data type to load model in
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for from_pretrained methods

        Returns:
            PreTrainedVLM instance configured for lazy loading
        """
        return cls(
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def generate(self, **kwargs) -> Union[torch.LongTensor, GenerateOutput]:
        """
        Generate sequences using the model.

        Args:
            **kwargs: Arguments for the generate method

        Returns:
            Generated sequences
        """
        return self.model.generate(**kwargs)

    def __call__(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)

    def encode_text(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Encode text input using the tokenizer.

        Args:
            text: Input text or list of texts
            **kwargs: Additional tokenizer arguments

        Returns:
            Encoded inputs ready for the model
        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer available. Set tokenizer manually or ensure model has one.")
        return self.tokenizer(text, return_tensors="pt", **kwargs).to(self.device)

    def decode(self, token_ids: torch.Tensor, **kwargs) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional decoding arguments

        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer available. Set tokenizer manually or ensure model has one.")
        return self.tokenizer.decode(token_ids, **kwargs)

    def process_images_and_text(
        self,
        images: Optional[Any] = None,
        text: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Process images and text together using the processor.

        Args:
            images: Input images
            text: Input text
            **kwargs: Additional processor arguments

        Returns:
            Processed inputs ready for the model
        """
        inputs = self.processor(images=images, text=text, return_tensors="pt", **kwargs)
        # Move all tensors in the dict to the device
        if isinstance(inputs, dict):
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    inputs[key] = value.to(self.device)
        return inputs

    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        Save the model and all components to a directory.

        Args:
            save_directory: Directory to save to
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        if hasattr(self, "_model") and self._model is not None:
            self._model.save_pretrained(save_path)

        # Save artifacts through base class
        self.save_artifacts(save_path)

    def to(self, device: Union[str, torch.device]) -> "PreTrainedVLM[VLMType]":
        """
        Move model to a device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.device = device
        if hasattr(self, "_model") and self._model is not None:
            self._model = self._model.to(device)
        return self

    def half(self) -> "PreTrainedVLM[VLMType]":
        """
        Convert model to half precision.

        Returns:
            Self for chaining
        """
        if hasattr(self, "_model") and self._model is not None:
            self._model = self._model.half()
        self.torch_dtype = torch.float16
        return self

    def float(self) -> "PreTrainedVLM[VLMType]":
        """
        Convert model to full precision.

        Returns:
            Self for chaining
        """
        if hasattr(self, "_model") and self._model is not None:
            self._model = self._model.float()
        self.torch_dtype = torch.float32
        return self

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Return the dtype of the model."""
        if hasattr(self, "_model") and self._model is not None:
            return next(self._model.parameters()).dtype
        return self.torch_dtype

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Get the number of parameters in the model.

        Args:
            only_trainable: Whether to count only trainable parameters

        Returns:
            Number of parameters
        """
        if not hasattr(self, "_model") or self._model is None:
            return 0

        if only_trainable:
            return sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self._model.parameters())

    def __repr__(self) -> str:
        """String representation."""
        parts = [f"{self.__class__.__name__}("]

        if self._model_name_or_path:
            parts.append(f"  model_name_or_path='{self._model_name_or_path}',")

        parts.append(f"  device='{self.device}',")

        if self.torch_dtype:
            parts.append(f"  torch_dtype={self.torch_dtype},")

        if self.trust_remote_code:
            parts.append(f"  trust_remote_code={self.trust_remote_code},")

        # Show loaded components
        loaded = []
        if hasattr(self, "_model") and self._model is not None:
            loaded.append("model")
        if hasattr(self, "_processor") and self._processor is not None:
            loaded.append("processor")
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            loaded.append("tokenizer")
        if hasattr(self, "_config") and self._config is not None:
            loaded.append("config")

        if loaded:
            parts.append(f"  loaded_components={loaded},")

        parts.append(")")
        return "\n".join(parts)
