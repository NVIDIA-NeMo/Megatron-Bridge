# TP=1 vs TP=4 Logit & Log-Prob Comparison: Qwen3-1.7B

Comparison of logits and log-probabilities between TP=1 and TP=4 using real
Qwen3-1.7B weights loaded via AutoBridge.

**Script**: `compare_tp_logits.py`

## Setup

| Parameter          | Value                    |
|--------------------|--------------------------|
| Model              | Qwen/Qwen3-1.7B         |
| Layers             | 28                       |
| Hidden size        | 2048                     |
| Vocab size         | 151,936                  |
| Dtype              | bfloat16                 |
| Prompt             | "The capital of France is" (5 tokens) |
| Baseline           | TP=1 (DP=4, full model per rank) |
| Comparison         | TP=4 (DP=1, sharded across 4 ranks) |
| GPU                | H100 (sm90)              |

## Raw Logit Differences

| Metric                     | Value    |
|----------------------------|----------|
| Max absolute difference    | 0.6875   |
| Mean absolute difference   | 0.0600   |
| Median absolute difference | 0.0312   |
| Cosine similarity (min)    | 0.99991  |
| Cosine similarity (mean)   | 0.99995  |

85% of logit elements differ by more than 1e-4. 83% differ by more than 1e-2.

### Per-position logit breakdown

| Pos | Max Abs Diff | Mean Abs Diff | Cosine Sim | Top-1 Match |
|-----|-------------|---------------|------------|-------------|
| 0   | 0.6875      | 0.1532        | 0.99991    | YES         |
| 1   | 0.2500      | 0.0368        | 0.99994    | YES         |
| 2   | 0.1875      | 0.0357        | 0.99997    | YES         |
| 3   | 0.1875      | 0.0378        | 0.99996    | YES         |
| 4   | 0.2500      | 0.0365        | 0.99995    | YES         |

Position 0 has the largest divergence.

## Chosen-Token Log-Prob Error

The chosen token `tᵢ` is the argmax from the TP=1 baseline at each position.
We compute `Δᵢ = logp_TP1(tᵢ) - logp_TP4(tᵢ)` and `p_A/p_B = exp(|Δᵢ|)`.

| Metric              | Value    |
|----------------------|----------|
| mean \|Δ\|          | 0.0696   |
| max \|Δ\|           | 0.2911   |
| signed mean (bias)  | -0.0658  |
| std                  | 0.1268   |
| token_mult_prob_error | 1.079x |

`token_mult_prob_error = (1/n) Σ exp(|Δᵢ|)` — on average the chosen token's
probability differs by ~8% between TP=1 and TP=4.

### Per-position detail

| Pos | Token | logp (TP=1) | logp (TP=4) | Δ       | p_A/p_B |
|-----|-------|-------------|-------------|---------|---------|
| 0   | 25    | -2.3221     | -2.0310     | -0.2911 | 1.338x  |
| 1   | 315   | -0.4478     | -0.4513     | +0.0035 | 1.003x  |
| 2   | 279   | -0.6793     | -0.6524     | -0.0269 | 1.027x  |
| 3   | 374   | -0.0462     | -0.0521     | +0.0059 | 1.006x  |
| 4   | 12095 | -0.6568     | -0.6363     | -0.0205 | 1.021x  |

Position 0 is the worst: TP=4 assigns 34% more probability to the top token
than TP=1. Remaining positions are within 1–3%.

## Full-Distribution Divergence

| Metric               | Mean    | Max     |
|----------------------|---------|---------|
| KL(TP=1 \|\| TP=4)  | 0.00254 | 0.00997 |
| KL(TP=4 \|\| TP=1)  | 0.00260 | 0.01029 |
| Total variation dist | 0.02042 | 0.05613 |

At worst ~5.6% of probability mass shifts between TP=1 and TP=4.

## Top-k Token Agreement

| Metric          | Value               |
|-----------------|---------------------|
| Top-1 agreement | 100% (5/5 positions)|
| Top-5 agreement | 100% (25/25 slots)  |

Greedy decoding is unaffected for this 5-token prompt. Longer generation or
sampling-based decoding may diverge.

## Reproducing

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 \
    examples/tp_investigation/compare_tp_logits.py
```
