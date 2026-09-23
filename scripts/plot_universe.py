"""Chart in-sample vs walk-forward Sharpe for every pair in universe_summary.csv.

python scripts/plot_universe.py --csv results/universe_summary.csv --out docs/universe_is_vs_oos.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

IS_COLOR = "#9aa5b1"
OOS_COLOR = "#1a4480"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/universe_summary.csv")
    ap.add_argument("--out", default="docs/universe_is_vs_oos.png")
    args = ap.parse_args()

    d = pd.read_csv(args.csv).sort_values("oos_sharpe")
    labels = d["name"].str.slice(0, 22)
    y = range(len(d))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [1.5, 1]})

    # Left: dumbbell — each pair's in-sample Sharpe and where it ended up out of sample.
    for i, (a, b) in enumerate(zip(d["is_sharpe"], d["oos_sharpe"])):
        ax1.plot([a, b], [i, i], color="#d0d5db", lw=2, zorder=1)
    ax1.scatter(d["is_sharpe"], y, color=IS_COLOR, s=36, label="In-sample optimum", zorder=2)
    ax1.scatter(d["oos_sharpe"], y, color=OOS_COLOR, s=36, label="Walk-forward (out of sample)", zorder=3)
    ax1.axvline(0, color="#555", lw=0.8)
    ax1.set_yticks(list(y), labels, fontsize=8)
    ax1.set_xlabel("Sharpe ratio (after costs)")
    ax1.set_title("Per pair: in-sample vs out-of-sample Sharpe")
    ax1.legend(loc="lower right", fontsize=8, frameon=False)
    ax1.grid(axis="x", color="#eee")

    # Right: scatter — does the in-sample number predict the out-of-sample one?
    ax2.scatter(d["is_sharpe"], d["oos_sharpe"], color=OOS_COLOR, s=30)
    lo = min(d["is_sharpe"].min(), d["oos_sharpe"].min()) - 0.1
    hi = max(d["is_sharpe"].max(), d["oos_sharpe"].max()) + 0.1
    ax2.plot([lo, hi], [lo, hi], ls="--", color="#999", lw=1, label="no decay (y = x)")
    ax2.axhline(0, color="#555", lw=0.8)
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel("In-sample Sharpe")
    ax2.set_ylabel("Walk-forward Sharpe")
    n_below = int((d["oos_sharpe"] < d["is_sharpe"]).sum())
    ax2.set_title(f"{n_below} of {len(d)} pairs fall below the line")
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.grid(color="#eee")

    fig.suptitle("HK–ADR premium mean reversion, daily closes, 2019 → 2026", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
