#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot e2e performance on L-Eval.

The values are taken from the L-Eval figure:
- Norm. Throughput and Norm. TTFT are the normalized labels shown above bars.
- SLO Attainment uses the actual percentages shown above bars.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "pdf" / "e2e_paper_assistant.pdf"


def add_grouped_bars(
    ax,
    values,
    value_labels,
    ylabel,
    ylim,
    yticks,
    show_rotation=True,
):
    models = ["Llama2-13B", "Llama2-70B"]
    methods = ["Recompute", "Swap", "HCache", "eLLM"]

    x = np.arange(len(models))
    width = 0.17
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    facecolors = {
        "Recompute": "#D9EEF2",
        "Swap": "#9FD0DC",
        "HCache": "#4E9AB3",
        "eLLM": "#1F5F8B",
    }
    hatches = {
        "Recompute": "///",
        "Swap": "\\\\\\",
        "HCache": "xx",
        "eLLM": "",
    }
    edgecolors = {
        "Recompute": "#222222",
        "Swap": "#222222",
        "HCache": "#222222",
        "eLLM": "#222222",
    }

    for j, method in enumerate(methods):
        bars = ax.bar(
            x + offsets[j],
            values[:, j],
            width=width,
            label=method,
            color=facecolors[method],
            edgecolor=edgecolors[method],
            linewidth=0.8,
            hatch=hatches[method],
            zorder=3,
        )

        for i, bar in enumerate(bars):
            val = value_labels[i][j]
            label = f"{val:.2f}" if ylabel != "SLO Attainment (%)" else f"{val:.1f}"
            y = bar.get_height()
            dy = (ylim[1] - ylim[0]) * 0.025
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + dy,
                label,
                ha="center",
                va="bottom",
                rotation=82 if show_rotation else 0,
                fontsize=7.0,
                color="black",
                zorder=5,
            )

    ax.set_ylabel(ylabel, fontsize=8.5, labelpad=2)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7.5)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.tick_params(axis="y", labelsize=7.5, width=0.8, length=3)
    ax.tick_params(axis="x", width=0.8, length=0)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    return ax


def main(output_pdf: str | Path = DEFAULT_OUTPUT):
    # L-Eval normalized throughput labels:
    # Llama2-13B: Recompute, Swap, HCache, eLLM = 1.11, 1.00, 1.88, 3.03
    # Llama2-70B: Recompute, Swap, HCache, eLLM = 1.28, 1.00, 1.56, 2.38
    throughput_labels = np.array([
        [1.11, 1.00, 1.88, 3.03],
        [1.28, 1.00, 1.56, 2.38],
    ], dtype=float)

    # L-Eval normalized TTFT labels:
    # Llama2-13B: Recompute, Swap, HCache, eLLM = 1.76, 1.71, 1.37, 1.00
    # Llama2-70B: Recompute, Swap, HCache, eLLM = 1.79, 1.62, 1.18, 1.00
    ttft_labels = np.array([
        [1.76, 1.71, 1.37, 1.00],
        [1.79, 1.62, 1.18, 1.00],
    ], dtype=float)

    # L-Eval SLO attainment values from the figure.
    slo = np.array([
        [87.2, 89.9, 92.7, 96.6],
        [90.0, 90.1, 94.0, 97.4],
    ], dtype=float)
    slo_labels = slo.copy()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.8,
        "hatch.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(5.65, 1.65))

    add_grouped_bars(
        axes[0],
        throughput_labels,
        throughput_labels,
        "Norm. Throughput",
        (0, 3.2),
        [0, 1, 2, 3],
    )
    add_grouped_bars(
        axes[1],
        ttft_labels,
        ttft_labels,
        "Norm. TTFT",
        (0, 3.0),
        [0, 1, 2, 3],
    )
    add_grouped_bars(
        axes[2],
        slo,
        slo_labels,
        "SLO Attainment (%)",
        (60, 105),
        [60, 80, 100],
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.06),
        ncol=4,
        frameon=False,
        columnspacing=0.7,
        handlelength=1.1,
        handletextpad=0.25,
        borderaxespad=0.0,
        fontsize=6.8,
    )

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.24, top=0.86, wspace=0.32)

    out = Path(output_pdf).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
