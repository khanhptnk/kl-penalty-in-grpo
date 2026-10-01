"""Figures for the post, each rendered twice (light and dark theme) as SVG, from the JSON files in results/.

Run: uv run plots.py        (after the three experiment scripts; writes figures/*.svg)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f in Path("/usr/share/fonts/truetype/ubuntu").glob("UbuntuMono*.ttf"):  # the blog's font, if installed
    font_manager.fontManager.addfont(str(f))
FONT = "Ubuntu Mono" if any(f.name == "Ubuntu Mono" for f in font_manager.fontManager.ttflist) else "DejaVu Sans Mono"

THEMES = {  # validated palette steps (dataviz reference palette), site surfaces #faf8f8 / #161618
    "light": dict(surface="#faf8f8", ink="#2b2b2b", ink2="#52514e", grid="#e5e5e5", ref="#6b6a66",
                  k3="#2a78d6", k1="#eb6834", k2="#1baf7a"),
    "dark": dict(surface="#161618", ink="#ebebec", ink2="#c3c2b7", grid="#393639", ref="#a3a29a",
                 k3="#3987e5", k1="#d95926", k2="#199e70"),
}
LW = 1.6  # ~2px


def style(ax, c):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(c["grid"])
    ax.tick_params(colors=c["ink2"], labelsize=10, length=0, pad=6)
    ax.grid(True, color=c["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(c["ink2"])
    ax.yaxis.label.set_color(c["ink2"])


def label(ax, x, y, text, color, c, **kw):
    """Direct label: text in ink (not series color), with a small colored dash for identity."""
    ax.annotate(text, (x, y), color=c["ink"], fontsize=10, va="center", **kw)


def new_fig(c, w=7.6, h=3.6, ncols=1):
    plt.rcParams.update({"font.family": FONT, "svg.fonttype": "path", "svg.hashsalt": "kl"})
    fig, axes = plt.subplots(1, ncols, figsize=(w, h))
    fig.patch.set_alpha(0)
    return fig, np.atleast_1d(axes)


def legend(ax, c, **kw):
    kw.setdefault("fontsize", 10)
    leg = ax.legend(frameon=False, labelcolor=c["ink"], handlelength=1.8, **kw)
    return leg


# ---------------------------------------------------------------- figure 1: gradient coefficient
def fig_coefficient(c, out):
    fig, (ax,) = new_fig(c, h=3.8)
    style(ax, c)
    ratio = np.logspace(np.log10(1 / 20), np.log10(20), 400)          # pi / pi_ref
    x = np.log(ratio)
    ax.plot(ratio, x, color=c["k1"], lw=LW, label="reverse KL: $k_1$ in reward, $k_2$ as loss")
    ax.plot(ratio, 1 - np.exp(-x), color=c["k3"], lw=LW, label="$k_3$ as loss")
    ax.axhline(1, color=c["ref"], lw=0.9, ls=(0, (4, 3)))
    ax.annotate("cap = 1", (1 / 19, 1.25), color=c["ink2"], fontsize=9)
    ax.axvline(1, color=c["grid"], lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks([1 / 16, 1 / 4, 1, 4, 16], ["1/16", "1/4", "1", "4", "16"])
    ax.set_ylim(-8, 4)
    ax.set_xlabel(r"$\pi(a)\,/\,\pi_{\mathrm{ref}}(a)$ at the sampled token")
    ax.set_ylabel(r"coefficient $c$ in $\nabla \mathrm{loss} = c\,\nabla\log\pi(a)$")
    ax.annotate("under-weighted:\n$k_3$ pushes up hard\n(exponential)", (0.24, -7.6), color=c["ink2"], fontsize=9.5)
    ax.annotate("over-weighted:\n$k_3$ pushes down gently\n(capped at 1)", (3.2, -2.2), color=c["ink2"], fontsize=9.5)
    label(ax, 20, np.log(20), "  reverse KL", c["k1"], c)
    label(ax, 20, 0.35, "  $k_3$", c["k3"], c)
    legend(ax, c, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, format="svg", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2: where training settles, vs beta
def fig_entropy(c, data, out):
    fig, (a1, a2) = new_fig(c, w=7.4, h=3.5, ncols=2)
    for ax in (a1, a2):
        style(ax, c)
    series = [("k1 in reward", "$k_1$ in reward", c["k1"]), ("k2 as loss", "$k_2$ as loss", c["k2"]),
              ("k3 as loss", "$k_3$ as loss", c["k3"])]
    betas = sorted(float(b) for b in data["optimum"])
    for ax, key, ylabel, title in ((a1, "entropy", "entropy (nats)", "(a) entropy of the policy"),
                                   (a2, "P(target)", "probability", "(b) P(rewarded sequence)")):
        ax.plot(betas, [data["optimum"][str(b)][key] for b in betas], color=c["ink2"], lw=1.2, ls=(0, (1, 2)),
                marker="o", ms=4, label="optimum")
        for name, lab, col in series:
            runs = [[r[key] for r in data["sweep"][name][str(b)]] for b in betas]
            ax.errorbar(betas, [np.mean(r) for r in runs], yerr=[np.std(r) for r in runs], color=col, lw=LW,
                        marker="o", ms=5, capsize=0, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(betas, [f"{b:g}" for b in betas])
        ax.minorticks_off()
        ax.set_xlabel("KL coefficient β")
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=c["ink"], fontsize=11, loc="left")
    legend(a1, c, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, format="svg", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3: dropped-mode recovery
def fig_recovery(c, data, out):
    fig, (ax,) = new_fig(c, h=3.6)
    style(ax, c)
    rec = {k: v[:5] for k, v in data["recovery"].items()}  # 5 seeds each keep the figure readable
    for i, tr in enumerate(rec["k2 as loss"]):
        ax.plot(tr, color=c["k2"], lw=1.2, alpha=0.9, label="$k_2$ as loss (reverse KL)" if i == 0 else None)
    for i, tr in enumerate(rec["k3 as loss"]):
        ax.plot(tr, color=c["k3"], lw=1.2, alpha=0.9, label="$k_3$ as loss, sampled" if i == 0 else None)
    ax.plot(rec["forward KL (exact)"][0], color=c["ref"], lw=1.2, ls=(0, (4, 3)), label="forward KL, exact gradient")
    ax.axhline(0.2, color=c["ref"], lw=0.9, ls=(0, (1, 2)))
    ax.annotate(r"$\pi_{\mathrm{ref}}(J) = 0.2$", (158, 0.17), color=c["ink2"], fontsize=9, ha="left", va="top")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlabel("training step (64 sampled tokens each)")
    ax.set_ylabel(r"$\pi(J)$, a mode the policy dropped")
    legend(ax, c, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, format="svg", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    single = json.load(open("results/single_position.json"))
    entropy = json.load(open("results/entropy.json"))
    out = Path("figures")
    out.mkdir(exist_ok=True)
    for theme, c in THEMES.items():
        fig_coefficient(c, out / f"kl-coefficient-{theme}.svg")
        fig_entropy(c, entropy, out / f"kl-entropy-{theme}.svg")
        fig_recovery(c, single, out / f"kl-recovery-{theme}.svg")
    print("wrote", sorted(p.name for p in out.glob("*.svg")))
