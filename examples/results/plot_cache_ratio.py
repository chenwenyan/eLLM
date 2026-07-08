#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Fig. (b): Effect of cache ratio.

This script reconstructs the original-style plot from the shown figure:
- left y-axis: KV cache usage (%), blue/purple line + translucent fill
- right y-axis: # of running requests, olive line
- vertical event line and red star annotation at cache-ratio change
- output: PDF

The data below is synthetic but shaped to match the original figure visually.
Replace `t`, `kv_usage`, and `running_reqs` with the original traces if available.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


GPU_USAGE_COLOR = "#6B4C9A"
RUNNING_COLOR = "#D95F02"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "pdf" / "effect_of_cache_ratio.pdf"


def make_trace(seed: int = 7):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 541, 2)  # seconds

    kv_usage = np.zeros_like(t, dtype=float)
    running_reqs = np.zeros_like(t, dtype=float)

    # KV cache usage: quickly reaches ~100%, remains high after reducing cache
    # ratio, then drains near the end.
    for i, x in enumerate(t):
        if x < 12:
            kv_usage[i] = 0.0
        elif x < 35:
            kv_usage[i] = 100 * (x - 12) / (35 - 12)
        elif x < 235:
            kv_usage[i] = 98 + 1.2 * np.sin(x / 28.0)
        elif x < 480:
            kv_usage[i] = 96 + 3.6 * np.sin(x / 28.0) + 2.4 * np.sin(x / 7.5)
        elif x < 515:
            kv_usage[i] = 91 - 2.2 * (x - 480)
        else:
            kv_usage[i] = max(0, 14 - 0.9 * (x - 515))

    kv_usage += rng.normal(0, 1.3, size=t.size)
    kv_usage = np.clip(kv_usage, 0, 100)

    # Running requests: lower before the ratio change, then increases after 50% cache ratio.
    for i, x in enumerate(t):
        if x < 15:
            running_reqs[i] = 0
        elif x < 55:
            running_reqs[i] = 80 + 25 * np.sin(x / 7.0) + rng.normal(0, 7)
        elif x < 235:
            running_reqs[i] = 85 + 8 * np.sin(x / 16.0) + rng.normal(0, 5)
        elif x < 305:
            running_reqs[i] = 190 + 25 * np.sin(x / 13.0) + rng.normal(0, 7)
        elif x < 485:
            running_reqs[i] = 210 + 18 * np.sin(x / 23.0) + rng.normal(0, 6)
        elif x < 520:
            running_reqs[i] = max(0, 210 - 6 * (x - 485)) + rng.normal(0, 5)
        else:
            running_reqs[i] = max(0, 25 - 3.0 * (x - 520)) + rng.normal(0, 2)

    running_reqs = np.clip(running_reqs, 0, 240)
    return t, kv_usage, running_reqs


def plot(output_pdf: str | Path = DEFAULT_OUTPUT):
    t, kv_usage, running_reqs = make_trace()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    event_color = "red"

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(3, 2),
        dpi=120,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )

    ax1.step(
        t,
        kv_usage,
        where="post",
        color=GPU_USAGE_COLOR,
        linewidth=1.2,
        zorder=3,
    )
    ax1.fill_between(
        t,
        kv_usage,
        0,
        step="post",
        color=GPU_USAGE_COLOR,
        alpha=0.18,
        zorder=1,
    )
    ax1.set_ylim(-3, 105)
    ax1.set_yticks([0, 50, 100])
    ax1.set_ylabel("KV usage (%)", color=GPU_USAGE_COLOR, labelpad=1, fontsize=9)
    ax1.tick_params(axis="y", labelcolor=GPU_USAGE_COLOR, pad=1)
    ax1.tick_params(axis="x", labelbottom=False, pad=1)

    ax2.step(
        t,
        running_reqs,
        where="post",
        color=RUNNING_COLOR,
        linewidth=1.2,
        zorder=3,
    )
    ax2.fill_between(
        t,
        running_reqs,
        0,
        step="post",
        color=RUNNING_COLOR,
        alpha=0.16,
        zorder=1,
    )
    ax2.set_ylim(0, 250)
    ax2.set_yticks([0, 100, 200])
    ax2.set_ylabel("Running reqs", color=RUNNING_COLOR, labelpad=1, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=RUNNING_COLOR, pad=1)
    ax2.tick_params(axis="x", pad=1)
    ax2.set_xlabel("Time (s)", labelpad=1)
    ax2.set_xticks([0, 200, 400])

    # Event annotation.
    event_t = 235
    event_y = 43
    for ax in (ax1, ax2):
        ax.axvline(event_t, color="black", linewidth=0.8, alpha=0.75, zorder=2)
    ax1.scatter([event_t], [event_y], marker="*", s=45, color=event_color, zorder=6)
    ax1.annotate(
        "Cache ratio: 50%",
        xy=(event_t, event_y),
        xytext=(35, 16),
        textcoords="data",
        fontsize=6.5,
        color="black",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.9,
            color="black",
            shrinkA=2,
            shrinkB=3,
            mutation_scale=9,
        ),
        ha="left",
        va="bottom",
        zorder=7,
    )

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="-.", linewidth=0.5, alpha=0.5)

    output_pdf = Path(output_pdf).expanduser()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Saved to {output_pdf}")


if __name__ == "__main__":
    plot()
