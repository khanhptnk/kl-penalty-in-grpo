"""Which fixed point does each KL implementation reach? A toy RL run on a tabular autoregressive policy.

Same tabular policy as kl_sequence.py (V = 3, T = 3, 27 sequences). Reward 1 for the sequence (0, 0, 0), else 0.
Training: REINFORCE with a batch-mean baseline, 64 sampled sequences per step, plain SGD, starting from the reference.
The KL term is one of: k1 in the reward (reward-to-go), k2 as a per-token loss, k3 as a per-token loss (GRPO),
or none. Entropy and KLs are computed exactly over all 27 sequences. The regularized objective
E[R] - beta * KL(P_pi || P_ref) has the closed-form optimum P* proportional to P_ref * exp(R / beta), shown for reference.

Run: uv run entropy_rl.py        (CPU, a few minutes; writes results/entropy.json)
"""

import json
from pathlib import Path

import torch

from kl_sequence import PIDX, PREFIXES, SEQS, TOK, V, reward_to_go

torch.set_default_dtype(torch.float64)
N, STEPS, LR, SEEDS = 64, 1500, 0.5, 10
BETAS = (0.05, 0.1, 0.2, 0.5, 1.0)
KINDS = ("k1 in reward", "k2 as loss", "k3 as loss")
R = torch.tensor([1.0 if y == (0, 0, 0) else 0.0 for y in SEQS])
REF = torch.randn(len(PREFIXES), V, generator=torch.Generator().manual_seed(0))
LQ = REF.log_softmax(-1)[PIDX, TOK]  # (27, T)


def metrics(theta: torch.Tensor) -> dict:
    logP = theta.log_softmax(-1)[PIDX, TOK].sum(-1)
    P, logQ = logP.exp(), LQ.sum(-1)
    return {
        "entropy": -(P * logP).sum().item(),
        "P(target)": P[0].item(),
        "KL(P || P_ref)": (P * (logP - logQ)).sum().item(),
        "KL(P_ref || P)": (logQ.exp() * (logQ - logP)).sum().item(),
    }


def train(kind: str, beta: float, seed: int, record: bool = False) -> tuple[dict, list[float]]:
    g = torch.Generator().manual_seed(seed)
    theta, curve = REF.clone(), []
    for step in range(STEPS):
        if record and step % 25 == 0:
            curve.append(metrics(theta)["entropy"])
        t = theta.clone().requires_grad_(True)
        LP = t.log_softmax(-1)[PIDX, TOK]
        idx = torch.multinomial(LP.sum(-1).exp().detach(), N, replacement=True, generator=g)
        lp, lq, r = LP[idx], LQ[idx], R[idx]
        adv = (r - r.mean())[:, None]
        if kind == "k1 in reward":
            loss = -((adv - beta * reward_to_go((lp - lq).detach())) * lp).sum(1).mean()
        else:
            loss = -(adv * lp).sum(1).mean()
            log_r = lq - lp
            if kind == "k3 as loss":
                loss = loss + beta * (log_r.exp() - 1 - log_r).sum(1).mean()
            elif kind == "k2 as loss":
                loss = loss + beta * (0.5 * log_r**2).sum(1).mean()
        theta = theta - LR * torch.autograd.grad(loss, t)[0]
    return metrics(theta), curve


if __name__ == "__main__":
    out = {"reference": metrics(REF), "optimum": {}, "sweep": {k: {} for k in KINDS}, "curves": {}}
    for beta in BETAS:
        Pstar = (LQ.sum(-1) + R / beta).softmax(0)
        out["optimum"][beta] = {"entropy": -(Pstar * Pstar.log()).sum().item(), "P(target)": Pstar[0].item()}
        line = f"beta {beta:<4}  optimum entropy {out['optimum'][beta]['entropy']:.3f}"
        for kind in KINDS:
            runs = [train(kind, beta, s)[0] for s in range(SEEDS)]
            out["sweep"][kind][beta] = runs
            e = torch.tensor([r["entropy"] for r in runs])
            line += f" | {kind} {e.mean():.3f} ± {e.std():.3f}"
        print(line, flush=True)
    for kind in (*KINDS, "no KL"):
        out["curves"][kind] = [train(kind, 0.2, s, record=True)[1] for s in range(SEEDS)]
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/entropy.json", "w"))
