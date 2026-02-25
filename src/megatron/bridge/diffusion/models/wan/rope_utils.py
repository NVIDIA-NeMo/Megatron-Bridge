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


import torch


class Wan3DRopeEmbeddings(torch.nn.Module):
    """
    Wan 3D RoPE embeddings implementation.
    Implements Wan's 3D RoPE embeddings for Mcore Attention based on Wan's implementation at https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py.
    """

    def __init__(self, dim_head, max_position_len):
        super().__init__()
        self.freqs = torch.cat(
            [
                self.rope_params(max_position_len, dim_head - 4 * (dim_head // 6)),
                self.rope_params(max_position_len, 2 * (dim_head // 6)),
                self.rope_params(max_position_len, 2 * (dim_head // 6)),
            ],
            dim=1,
        )
        if torch.cuda.is_available():
            self.freqs = self.freqs.cuda()

    def rope_params(self, max_position_len, dim_head, theta=10000):
        assert dim_head % 2 == 0
        freqs = torch.outer(
            torch.arange(max_position_len), 1.0 / torch.pow(theta, torch.arange(0, dim_head, 2).div(dim_head))
        )
        return freqs

    def forward(self, n_head, dim_head, cu_seqlens_q_padded, grid_sizes, device):
        n, c = n_head, dim_head // 2

        # split freqs
        freqs = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        freqs_real = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            freqs_real_i = torch.cat(
                [
                    freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(seq_len, 1, 1, -1)  # <-- add 1,1 for batch/head broadcasting

            # Double dimension from c -> 2c with rotating angles as (x0, x0, x1, x1, ...), for interleaving RoPE
            freqs_real_i = freqs_real_i.unsqueeze(-1).expand(-1, -1, -1, -1, 2).reshape(seq_len, 1, 1, dim_head)

            freqs_real.append(freqs_real_i)

        # Pad freqs_real_i to (padded_seq_len, 1, 1, dim_head) with 0s
        for i, freqs_real_i in enumerate(freqs_real):
            seq_len_q_padded = cu_seqlens_q_padded[i + 1] - cu_seqlens_q_padded[i]
            if freqs_real_i.shape[0] < seq_len_q_padded:
                pad_shape = (seq_len_q_padded - freqs_real_i.shape[0], 1, 1, dim_head)
                freqs_real_i = torch.cat(
                    [freqs_real_i, torch.zeros(pad_shape, dtype=freqs_real_i.dtype, device=freqs_real_i.device)], dim=0
                )
            freqs_real[i] = freqs_real_i

        # Each freqs_real[i] is (seq_len, 1, 1, dim_head)
        # We concatenate them along dim=0 to get (concatenated_seq_len, 1, 1, dim_head)
        freqs_real = torch.cat(freqs_real, dim=0)

        # Note:
        # when running context_parallel, which must use "thd" for qkv_format,
        # we don't need to scatter the freqs to the context parallel region,
        # because mcore rope_utils will automatically retrieve the correct freqs for each context parallel region

        return freqs_real
