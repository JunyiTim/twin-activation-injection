"""Dose-response figures with the paper's condition names and the original study's reverse-coded scoring.
Usage: python make_figures.py   (writes dose_response_gratitude.png, dose_response_control.png)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import paired_reanalysis as pr

DOSE = "#D9622B"
ACTIVE = "#2A6F97"
BASE = "#6B6B6B"
LEVELS = [("Low", 1.8), ("Moderate", 2.5), ("High", 3.2), ("Extreme", 3.9)]
TITLES = {
    "gratitude": ["Manipulation check", "Proximal: Elevation", "Distal: Empathic Concern"],
    "control": ["Manipulation check", "Proximal: Fatigue", "Distal: Desire for Predictability"],
}


def mean_sem(w, m, c):
    x = w[(m, c)].dropna()
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def draw(domain, path, out):
    w = pr.wide(path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, m, title in zip(axes, pr.MEASURES, TITLES[domain]):
        xs = [b for _, b in LEVELS]
        ms, es = zip(*[mean_sem(w, m, f"dose_{l}") for l, _ in LEVELS])
        ax.errorbar(xs, ms, yerr=es, fmt="o-", color=DOSE, lw=2, ms=7, capsize=3, zorder=3)
        am, ae = mean_sem(w, m, "active_Moderate")
        ax.errorbar([2.5], [am], yerr=[ae], fmt="s", color=ACTIVE, ms=8, capsize=3, zorder=3)
        bm, be = mean_sem(w, m, "noinj")
        ax.axhline(bm, color=BASE, ls="--", lw=1.5, zorder=1)
        ax.axhspan(bm - be, bm + be, color=BASE, alpha=0.10, zorder=0)
        for cond, col in (("real_prime", DOSE), ("real_control", ACTIVE)):
            cm, ce = mean_sem(w, m, cond)
            ax.axhline(cm, color=col, ls=":", lw=1.5, alpha=0.8, zorder=1)
            ax.axhspan(cm - ce, cm + ce, color=col, alpha=0.12, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{l}\n($\\beta$={b})" for l, b in LEVELS], fontsize=8)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("mean score")
    handles = [
        Line2D([0], [0], color=DOSE, marker="o", lw=2, label="Dose Injection (mean $\\pm$ SEM)"),
        Line2D([0], [0], color=ACTIVE, marker="s", lw=0, ms=8, label="Active-Control Injection (Moderate, single point)"),
        Line2D([0], [0], color=BASE, ls="--", lw=1.5, label="No-Injection baseline ($\\pm$ SEM)"),
        Line2D([0], [0], color=DOSE, ls=":", lw=1.5, label="Real Prime ($\\pm$ SEM)"),
        Line2D([0], [0], color=ACTIVE, ls=":", lw=1.5, label="Real Control ($\\pm$ SEM)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    draw("gratitude", pr.GRAT, "dose_response_gratitude.png")
    draw("control", pr.CTRL, "dose_response_control.png")
