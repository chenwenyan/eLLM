from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

DEFAULT_CSV = (
    "~/workspace/vllm/examples/imbalance/"
    "50.0qps-Llama-2-13b-chat-hf-170819-fcfs.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "pdf"
    / "imbalance_gpu_usage_requests.pdf"
)
REQUIRED_COLUMNS = ("GPU KV cache usage", "Running")
GPU_USAGE_COLOR = "#6B4C9A"
RUNNING_COLOR = "#D95F02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GPU KV cache usage and running requests over time."
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"Input CSV path. Defaults to {DEFAULT_CSV!r}.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output PDF path. Defaults to {str(DEFAULT_OUTPUT)!r}.",
    )
    return parser.parse_args()


def read_series(csv_path: str | Path) -> tuple[list[int], list[float], list[float]]:
    csv_path = Path(csv_path).expanduser()

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing required CSV column(s): {missing}")

        gpu_usage: list[float] = []
        running: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                gpu_usage.append(float(row["GPU KV cache usage"]))
                running.append(float(row["Running"]))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in {csv_path} at row {row_number}"
                ) from exc

    if not gpu_usage:
        raise ValueError(f"No data rows found in {csv_path}")

    x = list(range(len(gpu_usage)))
    return x, gpu_usage, running


def plot_distribution(csv_path: str | Path, output_path: str | Path) -> None:
    output_path = Path(output_path).expanduser()

    x, gpu_usage, running = read_series(csv_path)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(3, 2),
        dpi=120,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )

    ax1.step(
        x,
        gpu_usage,
        where="post",
        color=GPU_USAGE_COLOR,
        linewidth=1.2,
    )
    ax2.step(
        x,
        running,
        where="post",
        color=RUNNING_COLOR,
        linewidth=1.2,
    )

    ax1.fill_between(
        x,
        gpu_usage,
        step="post",
        color=GPU_USAGE_COLOR,
        alpha=0.18,
    )
    ax2.fill_between(
        x,
        running,
        step="post",
        color=RUNNING_COLOR,
        alpha=0.16,
    )

    ax1.set_ylabel("KV usage (%)", color=GPU_USAGE_COLOR, labelpad=1, fontsize=9)
    ax2.set_ylabel("Running reqs", color=RUNNING_COLOR, labelpad=1, fontsize=9)
    ax1.tick_params(axis="y", labelcolor=GPU_USAGE_COLOR, pad=1)
    ax2.tick_params(axis="y", labelcolor=RUNNING_COLOR, pad=1)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="-.", linewidth=0.5, alpha=0.5)
        ax.tick_params(axis="x", pad=1)

    ax2.set_xlabel("Time (s)", labelpad=1)
    ax1.tick_params(axis="x", labelbottom=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.98)
    plt.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_distribution(args.csv, args.output)


if __name__ == "__main__":
    main()
