#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce the e2e performance figure from the screenshot.

The numbers printed above bars are taken from the figure:
- Throughput / TTFT panels use the normalized labels shown in the original figure.
- SLO panel uses the actual SLO-attainment percentages shown in the original figure.

Because the screenshot does not expose the exact raw bar heights for Throughput
and Mean TTFT, the absolute values below are reconstructed to visually match the
original plot while preserving the printed ratios. Replace `throughput` and
`ttft` with raw measurements if available.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "pdf" / "e2e_sharegpt.pdf"


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
            # Offset label relative to axis range for consistent compact spacing.
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
    # Approximate raw values reconstructed from the screenshot and printed ratios.
    # Throughput labels: [1.00, 1.01, 1.38, 2.64] and [1.00, 1.00, 1.25, 2.00].
    throughput = np.array([
        [300, 303, 414, 792],
        [430, 430, 537.5, 860],
    ], dtype=float)
    throughput_labels = np.array([
        [1.00, 1.01, 1.38, 2.64],
        [1.00, 1.00, 1.25, 2.00],
    ], dtype=float)

    # TTFT labels: [2.63, 2.59, 1.70, 1.00] and [1.62, 1.61, 1.21, 1.00].
    ttft = np.array([
        [736.4, 725.2, 476.0, 280.0],
        [469.8, 466.9, 350.9, 290.0],
    ], dtype=float)
    ttft_labels = np.array([
        [2.63, 2.59, 1.70, 1.00],
        [1.62, 1.61, 1.21, 1.00],
    ], dtype=float)

    # Exact values visible in the screenshot.
    slo = np.array([
        [86.5, 88.0, 92.8, 97.3],
        [85.7, 87.1, 92.9, 98.6],
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
        (0, 3.0),
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

    # Single shared legend across the top, matching the original compact style.
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
