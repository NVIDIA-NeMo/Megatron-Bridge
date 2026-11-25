# Natural variation after the first nonzero update

5 original Swift runs and 4 runs of the same strict G2 Bridge recipe independently executed two GA8 optimizer updates on the eight-document input. Each comparison uses microbatch 17, immediately after the first nonzero update. All token and target coordinates matched. Within each run the four dense replicas had bitwise-identical observations, so the table counts each run once. The Bridge runs used the reference shared output-weight gradient, strict MTP backward, contiguous D backward and original-owner router counts. The main head used its standard backward. Each native update accumulated eight microbatches across 64 GB300 GPUs with TP1, PP1, CP16 and EP64. The first update used zero learning rate and the second used 2e-6.

Cells give the median and range over pairs. Each distance is 2‖A−B‖/(‖A‖+‖B‖), which weights the two tensors equally. CE covers every valid position. Logits cover all 163 measured vocabulary columns at sampled valid positions. Head-input gradients cover all 4,096 channels at those positions. Distances sharing a run are correlated.

| Measured tensor | Swift vs Swift (10 pairs) | Bridge vs Bridge (6 pairs) | Swift vs Bridge (20 pairs) |
| --- | ---: | ---: | ---: |
| Main per-token CE | 1.859% (1.809%–1.896%) | 1.833% (1.796%–1.859%) | 1.850% (1.801%–1.901%) |
| Main raw logits | 1.570% (1.416%–1.800%) | 1.640% (1.362%–1.856%) | 1.634% (1.298%–2.050%) |
| Main head-input gradient | 2.510% (2.080%–3.054%) | 2.732% (2.516%–3.233%) | 2.804% (2.307%–3.863%) |
| MTP per-token CE | 2.330% (2.308%–2.396%) | 2.372% (2.316%–2.425%) | 2.353% (2.306%–2.399%) |
| MTP raw logits | 3.061% (2.189%–6.144%) | 7.165% (2.113%–9.071%) | 6.191% (2.026%–8.787%) |
| MTP head-input gradient | 3.675% (2.247%–5.753%) | 3.528% (2.558%–5.035%) | 3.073% (2.030%–5.731%) |

For any two observed reference tensors A and B, the best achievable worst relative L2 for a single prediction against both is ‖A−B‖/(‖A‖+‖B‖), by the triangle inequality. The largest such lower bound among the original-original pairs is 0.900% on main raw logits and 3.072% on MTP raw logits. The per-comparison pass gate remains 0.5%. These restarts establish finite-sample distances and a reference-repeatability limit. Statistical equivalence and long-run training convergence require separate measurements.

The original Swift jobs were 429077, 429182, 429346, 429670 and 430086. Original run 429670 reached its scheduler limit during cleanup after completing the two updates and writing all 64 next-microbatch records. The Bridge jobs were 429417, 429671, 429935 and 430087. All used the same eight-document fixture and initial weights.

[All pairwise values and input fingerprints](post-update-repeat-distances.json) · [Original Swift and Bridge two-update capture](../OPTIMIZER-PARITY.md#capture-both-heads-after-the-first-nonzero-weight-update)

## Reproduce a pair

Use the public two-update guide to record two independent runs with the same checkpoint, fixture and optimizer settings. Set `RUN_A` and `RUN_B` to their optimizer capture directories, each containing `heads/rank00.npz` through `heads/rank63.npz`. Compare the two directions with the public head comparator below. It checks tokens and target masks and reconstructs each head in physical row order on every dense replica.

```bash
python "$EX/head_parity_256k.py" "$RUN_A/heads" "$RUN_B/heads" > forward.json
python "$EX/head_parity_256k.py" "$RUN_B/heads" "$RUN_A/heads" > reverse.json
python - <<'PY'
import json
from pathlib import Path

first, second = (json.loads(Path(name).read_text()) for name in ("forward.json", "reverse.json"))
for head in ("main", "mtp"):
    for field in ("valid_ce_relative_l2", "sample_raw_logit_relative_l2", "sample_head_input_grad_relative_l2"):
        a, b = first[head][field], second[head][field]
        distance = 2 * a * b / (a + b) if a + b else 0.0
        print(head, field, f"{distance:.4%}")
PY
```

The JSON uses the chronological jobs above for Swift A–E and Bridge A–D. Form all unique run pairs within each trainer and all pairs across trainers, and take the median and endpoints in each group. The CE, logits and gradient arrays are compared before any rounding of the summary statistics.
