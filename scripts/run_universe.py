"""Run in-sample vs. walk-forward for every pair in config/pairs.csv.

python scripts/run_universe.py --start 2019-01-01 --out results

Pairs that fail to download (delisted ADRs, changed tickers) are skipped and
listed at the end. Output: results/universe_summary.csv, one row per pair.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adr_hk_arb import CostModel, align_sessions, fetch_yahoo, in_sample_optimum, summarize, walk_forward  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(Path(__file__).resolve().parents[1] / "config" / "pairs.csv"))
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-days", type=int, default=756, help="skip pairs with less history")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs, dtype={"hk_ticker": str})
    costs = CostModel()
    rows, skipped = [], []
    for p in pairs.itertuples():
        tag = f"{p.hk_ticker}/{p.adr_ticker}"
        try:
            panel = align_sessions(**fetch_yahoo(p.hk_ticker, p.adr_ticker, args.start, args.end))
        except Exception as exc:  # noqa: BLE001 - keep going through the universe
            skipped.append((tag, f"download failed: {exc}"))
            continue
        if len(panel) < args.min_days:
            skipped.append((tag, f"only {len(panel)} aligned days"))
            continue
        best, _ = in_sample_optimum(panel, p.adr_per_share, costs)
        wf = summarize(walk_forward(panel, p.adr_per_share, costs=costs).oos)
        rows.append({
            "pair": tag, "name": p.name, "days": len(panel),
            "is_entry_k": None if best is None else best["entry_k"],
            "is_exit_k": None if best is None else best["exit_k"],
            "is_sharpe": None if best is None else best["sharpe"],
            "is_annual_return": None if best is None else best["annual_return"],
            "oos_sharpe": wf["sharpe"], "oos_annual_return": wf["annual_return"],
            "oos_max_drawdown": wf["max_drawdown"], "oos_trades": wf["n_trades"],
        })
        print(f"{tag:<18} IS Sharpe {rows[-1]['is_sharpe'] or float('nan'):5.2f} -> OOS {wf['sharpe']:5.2f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)
    table.to_csv(out / "universe_summary.csv", index=False)
    print(f"\n{len(rows)} pairs -> {out / 'universe_summary.csv'}")
    if skipped:
        print("skipped:")
        for tag, why in skipped:
            print(f"  {tag}: {why}")


if __name__ == "__main__":
    main()
