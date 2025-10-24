import torch
from torch import Tensor
from typing import List
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextRotaryEmbedding


class Qwen3VLTextRotaryEmbedding(Qwen3VLMoeTextRotaryEmbedding):

    def forward(self, position_ids: torch.Tensor, mrope_section: List[int]) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal,
                height and width in rope calculation.

        Returns:
            Tensor: Raw frequency embeddings for Megatron Core (shape: [seq_length, bs, 1, dim]).
                    Megatron Core will compute cos/sin internally and apply attention_scaling.
        """
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # if self.seq_len_interpolation_factor is not None:
        #     seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, mrope_section)
        
        # DON'T double the freqs for Megatron Core - it computes cos/sin internally
        # emb = torch.cat((freqs, freqs), dim=-1)  # This was causing dimension mismatch!
        
        # Apply attention scaling here since Megatron Core doesn't know about it
        # shape (seq_length, bs, 1, dim)
        emb = (freqs * self.attention_scaling)[..., None, :].transpose(0, 1).contiguous()
        return emb