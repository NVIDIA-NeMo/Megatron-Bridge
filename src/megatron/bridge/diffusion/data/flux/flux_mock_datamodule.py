# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Mock data module for FLUX model training."""

from dataclasses import dataclass

import torch
from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider
from torch.utils.data import DataLoader, Dataset


class _MockT2IDataset(Dataset):
    """
    A mock dataset class for text-to-image tasks, simulating data samples for training and testing.

    This dataset generates synthetic data for both image and text inputs, with options to use
    pre-cached latent representations or raw data. The class is designed for use in testing and
    prototyping machine learning models.

    Attributes:
        image_H (int): Height of the generated images.
        image_W (int): Width of the generated images.
        length (int): Total number of samples in the dataset.
        image_precached (bool): Whether to use pre-cached latent representations for images.
        text_precached (bool): Whether to use pre-cached embeddings for text.
        prompt_seq_len (int): Sequence length for text prompts.
        pooled_prompt_dim (int): Dimensionality of pooled text embeddings.
        context_dim (int): Dimensionality of the text embedding context.
        vae_scale_factor (int): Scaling factor for the VAE latent representation.
        vae_channels (int): Number of channels in the VAE latent representation.
    """

    def __init__(
        self,
        image_H: int = 1024,
        image_W: int = 1024,
        length: int = 100000,
        image_precached: bool = True,
        text_precached: bool = True,
        prompt_seq_len: int = 512,
        pooled_prompt_dim: int = 768,
        context_dim: int = 4096,
        vae_scale_factor: int = 8,
        vae_channels: int = 16,
    ):
        super().__init__()
        self.length = length
        self.H = image_H
        self.W = image_W
        self.image_precached = image_precached
        self.text_precached = text_precached
        self.vae_channels = vae_channels
        self.vae_scale_factor = vae_scale_factor
        self.prompt_seq_len = prompt_seq_len
        self.pooled_prompt_dim = pooled_prompt_dim
        self.context_dim = context_dim

        if self.image_precached:
            self.latent_shape = (
                vae_channels,
                int(image_H // vae_scale_factor),
                int(image_W // vae_scale_factor),
            )
        if self.text_precached:
            self.prompt_embeds_shape = (prompt_seq_len, context_dim)
            self.pooled_prompt_embeds_shape = (pooled_prompt_dim,)
            self.text_ids_shape = (prompt_seq_len, 3)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset.

        The sample includes pre-cached latent representations for images and text.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the generated data sample with keys:
                - 'latents': Pre-cached latent representation of the image [C, H, W].
                - 'prompt_embeds': Pre-cached text prompt embeddings [seq_len, context_dim].
                - 'pooled_prompt_embeds': Pooled text prompt embeddings [pooled_dim].
                - 'text_ids': Text position IDs [seq_len, 3].
        """
        item = {}

        if self.image_precached:
            # Latents in [C, H, W] format - will be batched to [B, C, H, W]
            item["latents"] = torch.randn(self.latent_shape, dtype=torch.bfloat16)
        else:
            # Raw images [3, H, W]
            item["images"] = torch.randn(3, self.H, self.W, dtype=torch.bfloat16)

        if self.text_precached:
            # T5 embeddings [seq_len, context_dim]
            item["prompt_embeds"] = torch.randn(self.prompt_embeds_shape, dtype=torch.bfloat16)
            # CLIP pooled embeddings [pooled_dim]
            item["pooled_prompt_embeds"] = torch.randn(self.pooled_prompt_embeds_shape, dtype=torch.bfloat16)
            # Text position IDs [seq_len, 3]
            item["text_ids"] = torch.zeros(self.text_ids_shape, dtype=torch.bfloat16)
        else:
            item["txt"] = "This is a sample caption input"

        return item

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.length


def _collate_fn(samples):
    """
    Collate function to batch samples from _MockT2IDataset.

    Args:
        samples: List of sample dictionaries from the dataset.

    Returns:
        dict: Batched dictionary with stacked tensors.
    """
    batch = {}

    # Stack latents: [B, C, H, W]
    if "latents" in samples[0]:
        batch["latents"] = torch.stack([s["latents"] for s in samples], dim=0)
    elif "images" in samples[0]:
        batch["images"] = torch.stack([s["images"] for s in samples], dim=0)

    # Stack text embeddings
    if "prompt_embeds" in samples[0]:
        # [B, seq_len, context_dim]
        batch["prompt_embeds"] = torch.stack([s["prompt_embeds"] for s in samples], dim=0)
        # [B, pooled_dim]
        batch["pooled_prompt_embeds"] = torch.stack([s["pooled_prompt_embeds"] for s in samples], dim=0)
        # [B, seq_len, 3]
        batch["text_ids"] = torch.stack([s["text_ids"] for s in samples], dim=0)
    elif "txt" in samples[0]:
        batch["txt"] = [s["txt"] for s in samples]

    # Add loss mask (all ones)
    if "latents" in batch:
        batch_size = batch["latents"].shape[0]
        latent_h = batch["latents"].shape[2]
        latent_w = batch["latents"].shape[3]
        # Loss mask covers all latent positions
        batch["loss_mask"] = torch.ones(batch_size, latent_h * latent_w, dtype=torch.bfloat16)

    return batch


@dataclass(kw_only=True)
class FluxMockDataModuleConfig(DatasetProvider):
    """
    Configuration for FLUX mock data module.

    This data module generates synthetic data for FLUX model training,
    matching the expected input format of FluxForwardStep.

    Attributes:
        path: Unused, kept for interface compatibility.
        seq_length: Sequence length (unused for FLUX, kept for interface compatibility).
        packing_buffer_size: Packing buffer size (unused for FLUX).
        micro_batch_size: Micro batch size for training.
        global_batch_size: Global batch size for training.
        num_workers: Number of data loading workers.
        dataloader_type: Type of dataloader ("external" for mock data).
        image_H: Height of input images.
        image_W: Width of input images.
        vae_channels: Number of VAE latent channels.
        vae_scale_factor: VAE spatial downsampling factor.
        prompt_seq_len: Sequence length for T5 text embeddings.
        context_dim: Dimensionality of T5 text embeddings.
        pooled_prompt_dim: Dimensionality of CLIP pooled embeddings.
        image_precached: Whether images are pre-encoded as VAE latents.
        text_precached: Whether text is pre-encoded as embeddings.
        num_train_samples: Number of training samples.
    """

    path: str = ""
    seq_length: int = 1024
    packing_buffer_size: int = None
    micro_batch_size: int = 1
    global_batch_size: int = 4
    num_workers: int = 8
    dataloader_type: str = "external"

    # Image dimensions
    image_H: int = 1024
    image_W: int = 1024

    # VAE settings
    vae_channels: int = 16
    vae_scale_factor: int = 8

    # Text embedding settings
    prompt_seq_len: int = 512
    context_dim: int = 4096
    pooled_prompt_dim: int = 768

    # Precaching settings (FLUX typically uses precached data)
    image_precached: bool = True
    text_precached: bool = True

    # Dataset size
    num_train_samples: int = 10000

    def __post_init__(self):
        """Initialize the mock dataset and dataloader."""
        mock_ds = _MockT2IDataset(
            image_H=self.image_H,
            image_W=self.image_W,
            length=self.num_train_samples,
            image_precached=self.image_precached,
            text_precached=self.text_precached,
            prompt_seq_len=self.prompt_seq_len,
            pooled_prompt_dim=self.pooled_prompt_dim,
            context_dim=self.context_dim,
            vae_scale_factor=self.vae_scale_factor,
            vae_channels=self.vae_channels,
        )

        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 8
            kwargs["persistent_workers"] = True

        self._train_dl = DataLoader(
            mock_ds,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            **kwargs,
        )
        self._train_dl_iter = iter(self._train_dl)
        self.sequence_length = self.seq_length

    def build_datasets(self, _context: DatasetBuildContext):
        """Build and return train/val/test dataloaders."""
        # Return iterator for external dataloader type
        return self._train_dl_iter, self._train_dl_iter, self._train_dl_iter
