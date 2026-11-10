"""Headline statistics from universe_summary.csv, with and without ratio-flagged pairs.

Pairs are backtested independently, so dropping the pairs whose ADR ratio
check fails (`ratio_check != "ok"`) is a pure filter on the existing results.

python scripts/summarize_universe.py --csv results/universe_summary.csv
"""
from __future__ import annotations

import argparse

import pandas as pd


def headline(d: pd.DataFrame) -> dict:
    gap = d["is_sharpe"] - d["oos_sharpe"]
    return {
        "pairs": len(d),
        "median IS Sharpe": d["is_sharpe"].median(),
        "median OOS Sharpe": d["oos_sharpe"].median(),
        "IS Sharpe > 0": int((d["is_sharpe"] > 0).sum()),
        "OOS Sharpe > 0": int((d["oos_sharpe"] > 0).sum()),
        "IS Sharpe > 0.3": int((d["is_sharpe"] > 0.3).sum()),
        "OOS Sharpe > 0.3": int((d["oos_sharpe"] > 0.3).sum()),
        "median OOS annual return": d["oos_annual_return"].median(),
        "median OOS max drawdown": d["oos_max_drawdown"].median(),
        "OOS worse than IS": int((gap > 0).sum()),
        "median Sharpe decay": gap.median(),
        "rank corr IS vs OOS": d["is_sharpe"].rank().corr(d["oos_sharpe"].rank()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/universe_summary.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    flagged = d["ratio_check"] != "ok"
    print("ratio-flagged pairs:", ", ".join(d.loc[flagged, "name"]) or "none")
    table = pd.DataFrame({
        "all pairs": headline(d),
        "excl. ratio-flagged": headline(d[~flagged]),
    })
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.to_string())


if __name__ == "__main__":
    main()
