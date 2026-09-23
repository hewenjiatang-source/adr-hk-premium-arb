"""Figures: premium with bands, equity curves, parameter heatmaps."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .strategy import ThresholdRule  # noqa: E402


def plot_premium(premium: pd.Series, rule: ThresholdRule, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(premium.index, premium * 100, lw=0.8, color="#4a4a4a", label="premium")
    for level, label, style in [
        (rule.mean, "mean", "-"),
        (rule.sell_entry, f"entry (+{rule.entry_k}σ)", "--"),
        (rule.sell_exit, f"exit ({rule.exit_k:+}σ)", ":"),
    ]:
        ax.axhline(level * 100, ls=style, lw=1, color="#1a4480", label=label)
    ax.set_ylabel("ADR premium to HK (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_equity(curves: dict[str, pd.Series], title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, eq in curves.items():
        ax.plot(eq.index, eq, lw=1.2, label=label)
    ax.axhline(1.0, color="grey", lw=0.6)
    ax.set_ylabel("equity (start = 1)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_heatmap(grid: pd.DataFrame, value: str, title: str, path: str) -> None:
    """Heatmap of `value` over (entry_k, exit_k), like the original research notebook."""
    mat = grid.pivot(index="exit_k", columns="entry_k", values=value).sort_index()
    fig, ax = plt.subplots(figsize=(12, 10))
    data = mat.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="viridis", aspect="auto", origin="upper")
    ax.set_xticks(range(mat.shape[1]), [f"{c:.1f}" for c in mat.columns])
    ax.set_yticks(range(mat.shape[0]), [f"{r:.1f}" for r in mat.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white")
    ax.set_xlabel("entry threshold (σ above mean)")
    ax.set_ylabel("exit threshold (σ from mean)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
