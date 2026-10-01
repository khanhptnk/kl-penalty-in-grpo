# KL penalties in GRPO: which KL do you actually get?

Code for the post [GRPO's KL penalty pulls toward the forward KL](https://khanhptnk.github.io/machine-learning/kl-penalty-in-grpo).
Everything runs on a CPU in a few minutes, and every expectation in the first two scripts is computed exactly
(summed over all outcomes), so the checks are exact identities, not statistical tests.

```sh
uv run kl_single_position.py   # ~10 s: gradient identities at one position; recovery of a dropped mode
uv run kl_sequence.py          # ~5 s:  the same questions over whole sequences (tabular policy, 27 sequences)
uv run entropy_rl.py           # ~5 min: toy RL runs; which fixed point each KL implementation reaches
uv run plots.py                # figures/*.svg (light and dark versions) from results/*.json
```

`results/` holds the JSON each script writes, so `plots.py` works without re-running anything.

## What each script shows

**`kl_single_position.py`**: for a categorical policy `π` and reference `π_ref`, with `r = π_ref / π` at a token sampled
from `π`, the expected gradient of each way of using a KL estimator:

| Implementation | Expected gradient equals |
|---|---|
| `k1 = −log r` as a detached reward (REINFORCE) | `∇ KL(π ‖ π_ref)`, the reverse KL |
| k1 as a loss | 0 |
| `k2 = ½ (log r)²` as a loss | `∇ KL(π ‖ π_ref)` |
| `k3 = (r − 1) − log r` as a loss (GRPO) | `∇ KL(π_ref ‖ π)`, the **forward** KL |

It then takes a policy that has nearly dropped a mode the reference likes (`π(J) ≈ 1.5e-4`, `π_ref(J) = 0.2`) and shows that
k3 as a loss pulls it back about 175× harder than k2 in expectation, but almost entirely through the rare event that J
is sampled (noise ≈ 10× signal at 64 tokens per batch). With sampled gradients, k3 recovers the mode in 19 of 20 seeds within 300 steps, each through
one large jump at a random time (whenever J is first sampled); k2 never does.

**`kl_sequence.py`**: a tabular autoregressive policy (3 tokens, length 3). Checks, all exact:

- **A.** Per-token k3 as a loss has exactly the gradient of `Σₜ E_{prefix ~ π}[ KL(π_ref(·|prefix) ‖ π(·|prefix)) ]`: a
  forward KL at every position, on prefixes the policy generated. *(Expected output: True.)*
- **B.** That is not the sequence-level forward KL `KL(P_ref ‖ P_π)`; the gap grows with the distance between π and π_ref.
  *(Expected output: False, plus the gap table.)*
- **C.** k1 in the reward with reward-to-go returns gives exactly the sequence-level reverse-KL gradient. *(True.)*
- **D.** Per-token k2 as a loss misses the part of that gradient that flows through future tokens *(False)*; adding that
  part back recovers it exactly *(True)*.

**`entropy_rl.py`**: REINFORCE with one rewarded sequence, the same tabular policy, and each KL implementation at
β ∈ {0.05, 0.1, 0.2, 0.5, 1}, 10 seeds each. k1 in the reward lands on the closed-form optimum of the reverse-KL-regularized
objective. k3 as a loss reaches a different fixed point: much higher entropy when the reward dominates (small β), slightly
lower entropy when the regularizer dominates (β = 0.5).

## Caveats

These are toy models chosen so that every quantity can be computed exactly. They establish what each implementation
optimizes. They do not measure how much the difference matters in LLM training, where β is small, the vocabulary is
large, the policy-gradient term dominates, and gradient clipping blunts large updates.

## License

MIT
