"""KL penalties over sequences, computed exactly on a tabular autoregressive policy.

The policy has its own logits for every prefix; V = 3 tokens and T = 3 steps give 27 sequences, so every expectation
is an exact sum. Checks:
  A. per-token k3 as a loss  ==  grad of  sum_t E_{prefix ~ pi}[ KL(pi_ref(.|prefix) || pi(.|prefix)) ]
     (a forward KL at every position, on prefixes the policy generated, prefix weights not differentiated)
  B. ... which differs from the sequence-level forward KL KL(P_ref || P_pi), more so the further pi is from pi_ref
  C. k1 in the reward, with reward-to-go returns  ==  grad of the sequence-level reverse KL KL(P_pi || P_ref)
  D. per-token k2 as a loss misses exactly the part of that gradient that flows through future tokens
  E. per-token k2 as a loss  ==  grad of  sum_t E_{prefix ~ pi}[ KL(pi(.|prefix) || pi_ref(.|prefix)) ]
     (a reverse KL at every position, prefix weights not differentiated: the mirror image of A)
  F. per-token k2 as a loss  ==  k1 in the reward with each token charged only its own KL (no reward-to-go)
  G. per-token k1 as a loss has zero expected gradient

Run: uv run kl_sequence.py        (CPU, ~5 s; writes results/sequence.json)
"""

import itertools
import json
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
V, T = 3, 3
PREFIXES = [()] + [p for t in range(1, T) for p in itertools.product(range(V), repeat=t)]
IDX = {p: i for i, p in enumerate(PREFIXES)}
SEQS = list(itertools.product(range(V), repeat=T))
PIDX = torch.tensor([[IDX[y[:t]] for t in range(T)] for y in SEQS])  # (27, T): prefix of each position
TOK = torch.tensor(SEQS)  # (27, T)


def reward_to_go(x: torch.Tensor) -> torch.Tensor:
    """(..., T) -> (..., T) with out[t] = sum_{s >= t} x[s]."""
    return torch.flip(torch.cumsum(torch.flip(x, [-1]), -1), [-1])


def gradients(theta: torch.Tensor, logq: torch.Tensor) -> dict:
    theta = theta.clone().requires_grad_(True)
    lp = theta.log_softmax(-1)[PIDX, TOK]  # (27, T) per-token log pi
    lq = logq[PIDX, TOK]  # (27, T) per-token log pi_ref
    logP, logQ = lp.sum(-1), lq.sum(-1)
    P, Q = logP.exp(), logQ.exp()
    w = P.detach()  # sequences sampled from pi, sampling not differentiated

    def grad(x):
        return torch.autograd.grad(x, theta, retain_graph=True)[0]

    log_r = lq - lp
    k1 = (lp - lq).detach()
    # sum over positions of the forward KL at each prefix, weighted by P_pi(prefix) (frozen)
    fwd_per_prefix = (logq.exp() * (logq - theta.log_softmax(-1))).sum(-1)
    rev_per_prefix = (theta.softmax(-1) * (theta.log_softmax(-1) - logq)).sum(-1)  # KL(pi(.|h) || pi_ref(.|h))
    prefix_weight = torch.zeros(len(PREFIXES)).index_add_(0, PIDX.flatten(), w.repeat_interleave(T))
    return {
        "seq reverse KL": grad((P * (logP - logQ)).sum()),
        "seq forward KL": grad((Q * (logQ - logP)).sum()),
        "k3 as loss (per token)": grad((w * (log_r.exp() - 1 - log_r).sum(-1)).sum()),
        "per-position forward KL on pi's prefixes": grad((prefix_weight * fwd_per_prefix).sum()),
        "per-position reverse KL on pi's prefixes": grad((prefix_weight * rev_per_prefix).sum()),
        "k1 in reward (own token only)": grad((w * (k1 * lp).sum(-1)).sum()),
        "k1 as loss (per token)": grad((w * (lp - lq).sum(-1)).sum()),
        "k1 in reward (reward-to-go)": grad((w * (reward_to_go(k1) * lp).sum(-1)).sum()),
        "k2 as loss (per token)": grad((w * (0.5 * log_r**2).sum(-1)).sum()),
        "future-token part": grad((w * ((reward_to_go(k1) - k1) * lp).sum(-1)).sum()),
    }


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    ref_logits = torch.randn(len(PREFIXES), V, generator=g)
    direction = torch.randn(len(PREFIXES), V, generator=g)
    logq = ref_logits.log_softmax(-1)
    G = gradients(ref_logits + direction, logq)
    checks = {
        "A: k3 as loss == per-position forward KL on pi's prefixes":
            torch.allclose(G["k3 as loss (per token)"], G["per-position forward KL on pi's prefixes"]),
        "B: k3 as loss == sequence forward KL": torch.allclose(G["k3 as loss (per token)"], G["seq forward KL"]),
        "C: k1 in reward (reward-to-go) == sequence reverse KL":
            torch.allclose(G["k1 in reward (reward-to-go)"], G["seq reverse KL"]),
        "D: k2 as loss == sequence reverse KL": torch.allclose(G["k2 as loss (per token)"], G["seq reverse KL"]),
        "D: k2 as loss + future-token part == sequence reverse KL":
            torch.allclose(G["k2 as loss (per token)"] + G["future-token part"], G["seq reverse KL"]),
        "E: k2 as loss == per-position reverse KL on pi's prefixes":
            torch.allclose(G["k2 as loss (per token)"], G["per-position reverse KL on pi's prefixes"]),
        "F: k2 as loss == k1 in reward, own token only":
            torch.allclose(G["k2 as loss (per token)"], G["k1 in reward (own token only)"]),
        "G: k1 as loss has zero expected gradient":
            torch.allclose(G["k1 as loss (per token)"], torch.zeros_like(G["k1 as loss (per token)"]), atol=1e-12),
    }
    gaps = {}
    for alpha in (0.01, 0.1, 0.3, 1.0, 2.0):  # pi = pi_ref + alpha * direction (in logit space)
        Ga = gradients(ref_logits + alpha * direction, logq)
        gaps[alpha] = {
            "k3 as loss vs sequence forward KL": rel(Ga["k3 as loss (per token)"], Ga["seq forward KL"]),
            "k2 as loss vs sequence reverse KL": rel(Ga["k2 as loss (per token)"], Ga["seq reverse KL"]),
        }
        checks[f"A, C, D, E, F, G at alpha {alpha}"] = all((
            torch.allclose(Ga["k1 as loss (per token)"], torch.zeros_like(Ga["k1 as loss (per token)"]), atol=1e-12),
            torch.allclose(Ga["k3 as loss (per token)"], Ga["per-position forward KL on pi's prefixes"]),
            torch.allclose(Ga["k2 as loss (per token)"], Ga["per-position reverse KL on pi's prefixes"]),
            torch.allclose(Ga["k2 as loss (per token)"], Ga["k1 in reward (own token only)"]),
            torch.allclose(Ga["k1 in reward (reward-to-go)"], Ga["seq reverse KL"]),
            torch.allclose(Ga["k2 as loss (per token)"] + Ga["future-token part"], Ga["seq reverse KL"]),
        ))
    for name, ok in checks.items():
        print(f"{name:<60} {ok}")
    print("relative gradient gap as pi moves away from pi_ref:")
    for alpha, gap in gaps.items():
        print(f"  alpha {alpha:<5}" + "  ".join(f"{k}: {v:.4f}" for k, v in gap.items()))
    Path("results").mkdir(exist_ok=True)
    json.dump({"checks": checks, "gaps": gaps}, open("results/sequence.json", "w"))
