#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce the grouped-bar improvement plot.

The values are reconstructed from the provided screenshot:
  Throughput: 13.4, 18.2
  TPOT:        6.4, 25.0
  TTFT:      -13.9, -11.4

Output:
  k1_k2_improvement_bar.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "pdf" / "0.2_200_hfusion.pdf"


def main(output_pdf: str | Path = DEFAULT_OUTPUT):
    metrics = ["Throughput", "TPOT", "TTFT"]
    x = np.arange(len(metrics))

    labels = ["0.5:0.5", "0.75:0.25"]
    data = {
        "0.5:0.5": np.array([13.4, 6.4, -13.9]),
        "0.75:0.25": np.array([18.2, 25.0, -11.4]),
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    colors = {
        "0.5:0.5": "#1B9E77",
        "0.75:0.25": "#D95F02",
    }

    fig, ax = plt.subplots(figsize=(3, 2), dpi=120)

    y0 = data[labels[0]]
    y1 = data[labels[1]]
    for idx in x:
        ax.plot(
            [idx, idx],
            [y0[idx], y1[idx]],
            color="#8f8f8f",
            linewidth=1.1,
            zorder=2,
        )

    marker_styles = {
        "0.5:0.5": "o",
        "0.75:0.25": "s",
    }
    label_offsets = {
        "0.5:0.5": -0.08,
        "0.75:0.25": 0.08,
    }
    label_alignments = {
        "0.5:0.5": "right",
        "0.75:0.25": "left",
    }
    for label in labels:
        y = data[label]
        ax.scatter(
            x,
            y,
            label=label,
            color=colors[label],
            marker=marker_styles[label],
            s=28,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )

        for x_pos, val in zip(x, y):
            if val >= 0:
                y_pos = val + 2.2
                va = "bottom"
            else:
                y_pos = val - 2.2
                va = "top"

            ax.text(
                x_pos + label_offsets[label],
                y_pos,
                f"{abs(val):.0f}",
                ha=label_alignments[label],
                va=va,
                color=colors[label],
                zorder=5,
            )

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)

    ax.set_ylabel("Improvement (%)", labelpad=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_xlim(-0.35, len(metrics) - 0.65)

    ax.set_ylim(-40, 75)
    ax.set_yticks([-40, -20, 0, 20, 40, 60])

    ax.grid(
        True,
        axis="y",
        linestyle="-.",
        linewidth=0.5,
        alpha=0.5,
        zorder=0,
    )
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="K1:K2",
        loc="upper right",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#dddddd",
        framealpha=0.9,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.2,
        handletextpad=0.35,
    )

    fig.tight_layout(pad=0.25)
    output_pdf = Path(output_pdf).expanduser()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved to {output_pdf}")


if __name__ == "__main__":
    main()
