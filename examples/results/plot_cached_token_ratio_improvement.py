#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce the cached-token-ratio improvement plot.

The data is reconstructed from the provided screenshot, so it is approximate.
Replace the arrays below with the original measured values if available.

Output:
  cached_token_ratio_improvement.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "pdf"
    / "cached_token_ratio_improvement.pdf"
)


def main(output_pdf: str | Path = DEFAULT_OUTPUT):
    # Approximate data reconstructed from the figure.
    cached_ratio = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

    throughput = np.array([125, 112, 108, 88, 66, 50, 38, 30, 20])
    tpot = np.array([-170, -125, -70, -18, -18, -10, -6, -2, 3])
    ttft = np.array([65, 58, 48, 34, 22, 13, 8, 4, 0])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    purple = "#6B4C9A"
    orange = "#D95F02"
    teal = "#1B9E77"

    fig, ax = plt.subplots(figsize=(3, 2), dpi=120)

    x = np.arange(cached_ratio.size)
    bar_width = 0.24
    ax.bar(
        x - bar_width,
        throughput,
        width=bar_width,
        color=teal,
        alpha=0.86,
        label="Throughput",
    )
    ax.bar(
        x,
        tpot,
        width=bar_width,
        color=purple,
        alpha=0.82,
        label="TPOT",
    )
    ax.bar(
        x + bar_width,
        ttft,
        width=bar_width,
        color=orange,
        alpha=0.82,
        label="TTFT",
    )

    ax.set_xlabel("Cached token ratio (%)", labelpad=1)
    ax.set_ylabel("Improvement (%)", labelpad=1)

    ax.set_xlim(-0.75, cached_ratio.size - 0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(cached_ratio)
    ax.set_ylim(-210, 210)
    ax.set_yticks([-200, -100, 0, 100, 200])

    ax.grid(True, axis="y", linestyle="-.", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", pad=1)

    ax.legend(
        loc="lower right",
        ncol=1,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.35,
        borderaxespad=0.2,
    )

    output_pdf = Path(output_pdf).expanduser()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Saved to {output_pdf}")


if __name__ == "__main__":
    main()
