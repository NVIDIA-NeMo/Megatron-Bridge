# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Selectable uniform-state diffusion SFT losses for DiffusionGemma (PyTorch).

This objective is a derivative work of the Apache-2.0 licensed DiffGemma SFT
reference from Rhine-AI-Lab/diffusion-gemma-sft. The formulas are reimplemented
here against Megatron's loss contract; no reference repository code is imported
at runtime.

Port of the JAX/kauldron objectives in
``gemma/diffusion/hackable_diffusion_adapter/hd/sft_model.py``. All variants are
a masked denoising cross-entropy on the completion tokens, differing only in a
scalar time-weight ``w(t)`` and in which distribution the CE scores:

    L_SFT = -E_t[ w(t) * sum_l M_l * log <x0_l, p_theta_l(x_t, t)> ]

    variant             w(t)                prediction p_theta
    ------------------  ------------------  ------------------------------------------
    base-sft            1                   softmax(f_theta)            (raw denoiser)
    reweighted-ce       -alpha'(t)/alpha(t) softmax(f_theta)            (raw denoiser)
    loo-ce              1                   softmax(f_theta + delta*xt) (LOO->denoiser)
    reweighted-loo-ce   -alpha'(t)/alpha(t) softmax(f_theta + delta*xt) (LOO->denoiser)

where ``delta = log(1 + K*alpha/(1-alpha))`` is the leave-one-out -> denoiser
logit shift (Gourevitch et al., arXiv:2605.22765), ``K`` the vocab size, and the
rectified-flow schedule gives ``alpha(t) = 1 - t``, ``alpha'(t) = -1`` so
``reweighted-ce``'s weight is ``1/(1-t)``.

Note: ``reweighted-ce`` uses ``-alpha'/alpha`` with **no** ``1/K`` factor. The
``1/K`` only belongs to the *exact* ELBO's ``coeff = alpha'/(K*alpha)`` where it
cancels in the term structure; in a plain weighted CE it just shrinks the loss
by K (the original bug that drove eval to ~0.4%).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


VARIANTS = ("base-sft", "reweighted-ce", "loo-ce", "reweighted-loo-ce")
_ALIASES = {"loo-reweighted-ce": "reweighted-loo-ce"}
_LOO_VARIANTS = ("loo-ce", "reweighted-loo-ce")
_REWEIGHTED_VARIANTS = ("reweighted-ce", "reweighted-loo-ce")

_EPS = 1e-12


def normalize_variant(variant: str) -> str:
    """Return the canonical objective name, accepting the early branch alias."""
    variant = _ALIASES.get(variant, variant)
    if variant not in VARIANTS:
        supported = VARIANTS + tuple(_ALIASES)
        raise ValueError(f"Unknown variant {variant!r}; expected one of {supported}.")
    return variant


def rf_alpha(t: torch.Tensor) -> torch.Tensor:
    """Rectified-flow signal-retention schedule alpha(t) = 1 - t."""
    return 1.0 - t


def rf_alpha_dot(t: torch.Tensor) -> torch.Tensor:
    """Time derivative d alpha / dt = -1 for the RF schedule."""
    return -torch.ones_like(t)


def loo_to_denoiser_logits(
    logits: torch.Tensor, xt: torch.Tensor, alpha: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """Map LOO logits -> standard denoiser logits (the +delta observed-token shift).

    ``delta = log(1 + K*alpha/(1-alpha))`` added to the logit of the observed
    noisy token ``xt`` at each position. Inverse of the denoiser->LOO map.

    Args:
      logits: ``[B, L, K]`` raw (LOO) logits.
      xt: ``[B, L]`` observed noisy token ids.
      alpha: ``[B, 1]`` or ``[B, L]`` signal-retention probability.
      vocab_size: ``K``.
    """
    k = vocab_size
    a = alpha
    delta = torch.log1p(k * a / torch.clamp(1.0 - a, min=_EPS))  # [B,1] or [B,L]
    if delta.dim() == 2 and delta.shape[1] == 1:
        delta = delta.expand(-1, logits.shape[1])  # [B, L]
    onehot = F.one_hot(xt, k).to(logits.dtype)  # [B, L, K]
    return logits + delta.unsqueeze(-1) * onehot


def reweight(variant: str, t: torch.Tensor, alpha: torch.Tensor, alpha_dot: torch.Tensor) -> torch.Tensor:
    """Per-example time weight w(t) for the given variant. Shape matches ``t`` ([B])."""
    variant = normalize_variant(variant)
    if variant in _REWEIGHTED_VARIANTS:
        # MD4-style: w(t) = -alpha'(t) / alpha(t)  (no 1/K).
        return -alpha_dot / torch.clamp(alpha, min=_EPS)
    return torch.ones_like(t)


def diffusion_sft_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    variant: str,
    vocab_size: int,
    diffusion_weight_clip: float | None = None,
    alpha_fn=rf_alpha,
    alpha_dot_fn=rf_alpha_dot,
) -> torch.Tensor:
    """Masked, time-weighted denoising cross-entropy. Returns per-example loss [B].

    Args:
      logits: ``[B, L, K]`` network logits over the canvas.
      x0: ``[B, L]`` clean (target) completion tokens.
      xt: ``[B, L]`` noisy canvas tokens.
      t: ``[B]`` per-sequence diffusion time.
      loss_mask: ``[B, L]`` 1.0 on completion tokens to score, 0.0 elsewhere.
      variant: one of ``VARIANTS``.
      vocab_size: ``K``.
      diffusion_weight_clip: optional upper bound for the scalar time weight.
        This is mainly useful for ``reweighted-ce``, whose RF weight is
        ``1 / (1 - t)`` and can explode near ``t=1``.
      alpha_fn / alpha_dot_fn: schedule callables of ``t``.
    """
    variant = normalize_variant(variant)
    logits = logits.float()
    alpha = alpha_fn(t).unsqueeze(-1)  # [B, 1]
    if variant in _LOO_VARIANTS:
        logits = loo_to_denoiser_logits(logits, xt, alpha, vocab_size)
    logp = F.log_softmax(logits, dim=-1)
    per_token_ce = -logp.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
    w = reweight(variant, t, alpha_fn(t), alpha_dot_fn(t))  # [B]
    if diffusion_weight_clip is not None:
        w = torch.clamp(w, max=float(diffusion_weight_clip))
    mask = loss_mask.to(per_token_ce.dtype)
    sum_ce = (per_token_ce * mask).sum(dim=-1)  # [B]
    n = mask.sum(dim=-1).clamp(min=1.0)  # [B]
    return w * (sum_ce / n)  # [B]


def diffusion_sft_logps(
    logits: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    variant: str,
    vocab_size: int,
    diffusion_weight_clip: float | None = None,
    alpha_fn=rf_alpha,
    alpha_dot_fn=rf_alpha_dot,
) -> torch.Tensor:
    """Per-example *summed* denoise log-prob surrogate ``-( w(t) * sum_l M_l CE_l )``.

    Same masked, time-weighted denoising cross-entropy as :func:`diffusion_sft_loss`,
    but returned in **summed** (un-normalized) form as a log-probability (sign
    flipped). This is the convention the GRPO/GDSD objective expects: it divides
    the per-sequence logp by ``norm = number of scored tokens``, so

        diffusion_sft_logps(...) / n == -diffusion_sft_loss(...)

    when ``norm`` equals the SFT mask count ``n = loss_mask.sum(-1)``. Args match
    :func:`diffusion_sft_loss`. Returns ``[B]``.
    """
    variant = normalize_variant(variant)
    logits = logits.float()
    alpha = alpha_fn(t).unsqueeze(-1)  # [B, 1]
    if variant in _LOO_VARIANTS:
        logits = loo_to_denoiser_logits(logits, xt, alpha, vocab_size)
    logp = F.log_softmax(logits, dim=-1)
    per_token_ce = -logp.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
    w = reweight(variant, t, alpha_fn(t), alpha_dot_fn(t))  # [B]
    if diffusion_weight_clip is not None:
        w = torch.clamp(w, max=float(diffusion_weight_clip))
    mask = loss_mask.to(per_token_ce.dtype)
    sum_ce = (per_token_ce * mask).sum(dim=-1)  # [B]
    return -(w * sum_ce)  # [B]
