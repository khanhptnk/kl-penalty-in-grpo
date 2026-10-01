"""KL estimators at a single position: exact gradient identities, and recovery of a dropped mode.

1. For a categorical policy pi and a fixed reference pi_ref, compute the expected gradient of each way of using a KL
   estimator, exactly (sum over the vocabulary, no sampling), and compare with the true gradients of the reverse KL
   KL(pi || pi_ref) and the forward KL KL(pi_ref || pi).
2. Take a policy that has nearly dropped a mode of the reference. Compute the exact mean and variance of the k3-as-loss
   and k2-as-loss gradient on that mode, then train with sampled gradients and track whether the mode comes back.

Run: uv run kl_single_position.py        (CPU, ~10 s; writes results/single_position.json)
"""

import json
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)


def gradient_identities(V: int = 6, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(V, generator=g).requires_grad_(True)
    logq = torch.log_softmax(torch.randn(V, generator=g), 0)  # pi_ref, fixed
    logp = torch.log_softmax(theta, 0)
    p, q = logp.exp(), logq.exp()
    w = p.detach()  # a ~ pi: the sampling distribution is not differentiated

    def grad(x):
        return torch.autograd.grad(x, theta, retain_graph=True)[0]

    reverse = grad((p * (logp - logq)).sum())
    forward = grad((q * (logq - logp)).sum())
    log_r = logq - logp  # log r, r = pi_ref / pi
    got = {
        "k1 in reward": grad((w * (logp - logq).detach() * logp).sum()),
        "k1 as loss": grad((w * -log_r).sum()),
        "k2 as loss": grad((w * 0.5 * log_r**2).sum()),
        "k3 as loss": grad((w * (log_r.exp() - 1 - log_r)).sum()),
    }
    zero = torch.zeros(V)
    return {
        name: {
            "matches reverse KL": torch.allclose(gv, reverse),
            "matches forward KL": torch.allclose(gv, forward),
            "is zero": torch.allclose(gv, zero, atol=1e-12),
        }
        for name, gv in got.items()
    }


# ---- a mode the policy nearly dropped: token J has pi_ref(J) = 0.2 but pi(J) ~ 2e-4 ----
V_DROP, J = 10, 0
REF = torch.full((V_DROP,), 0.8 / (V_DROP - 1))
REF[J] = 0.2
LOGQ = REF.log()
THETA0 = LOGQ.clone()
THETA0[J] = -9.0


def per_sample_loss(kind: str, log_r: torch.Tensor) -> torch.Tensor:
    if kind == "k3 as loss":
        return log_r.exp() - 1 - log_r
    if kind == "k2 as loss":
        return 0.5 * log_r**2
    raise ValueError(kind)


def dropped_mode_statistics() -> dict:
    """Exact mean and variance of the per-sample gradient w.r.t. the dropped mode's logit."""
    p = THETA0.softmax(0)
    out = {"pi(J)": p[J].item(), "pi_ref(J)": REF[J].item()}
    for kind in ("k3 as loss", "k2 as loss"):
        rows = []
        for a in range(V_DROP):  # the gradient when token a is the sampled one
            t = THETA0.clone().requires_grad_(True)
            log_r = LOGQ[a] - t.log_softmax(0)[a]
            rows.append(torch.autograd.grad(per_sample_loss(kind, log_r), t)[0][J])
        gJ = torch.stack(rows)
        mean = (p * gJ).sum()
        var = (p * (gJ - mean) ** 2).sum()
        out[kind] = {
            "expected gradient on J": mean.item(),
            "share of it from sampling J": (p[J] * gJ[J] / mean).item(),
            "noise / signal, 64-token batch": ((var / 64).sqrt() / mean.abs()).item(),
            "noise / signal, 1024-token batch": ((var / 1024).sqrt() / mean.abs()).item(),
        }
    return out


def recover(kind: str, seed: int, steps: int = 300, n: int = 64, lr: float = 0.5) -> list[float]:
    g = torch.Generator().manual_seed(seed)
    theta = THETA0.clone()
    traj = [theta.softmax(0)[J].item()]
    for _ in range(steps):
        t = theta.clone().requires_grad_(True)
        logp = t.log_softmax(0)
        if kind == "forward KL (exact)":
            loss = -(REF * logp).sum()
        else:
            a = torch.multinomial(logp.exp().detach(), n, replacement=True, generator=g)
            loss = per_sample_loss(kind, LOGQ[a] - logp[a]).mean()
        theta = theta - lr * torch.autograd.grad(loss, t)[0]
        traj.append(theta.softmax(0)[J].item())
    return traj


if __name__ == "__main__":
    results = {"identities": gradient_identities(), "dropped mode": dropped_mode_statistics()}
    results["recovery"] = {
        kind: [recover(kind, s) for s in range(20)] for kind in ("forward KL (exact)", "k3 as loss", "k2 as loss")
    }
    for name, r in results["identities"].items():
        print(f"{name:>13}: " + ", ".join(k for k, v in r.items() if v))
    print(json.dumps(results["dropped mode"], indent=2))
    for kind, trajs in results["recovery"].items():
        recovered = [next((i for i, x in enumerate(t) if x > 0.01), None) for t in trajs]  # first step past 1%
        hits = sorted(i for i in recovered if i is not None)
        print(f"{kind:>20}: {len(hits)}/{len(trajs)} seeds recover within 300 steps; first step past 1%: {hits}")
    Path("results").mkdir(exist_ok=True)
    json.dump(results, open("results/single_position.json", "w"))
