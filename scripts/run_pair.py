"""Research one HK/ADR pair: in-sample grid vs. walk-forward out-of-sample.

Examples
--------
python scripts/run_pair.py --hk 9988.HK --adr BABA --ratio 0.125 --start 2019-01-01
python scripts/run_pair.py --synthetic                       # offline demo
python scripts/run_pair.py --csv data/my_pair.csv --ratio 0.5  # vendor data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adr_hk_arb import (  # noqa: E402
    CostModel, ThresholdRule, align_sessions, backtest, fetch_yahoo, in_sample_optimum,
    load_csv, make_synthetic_pair, summarize, walk_forward,
)
from adr_hk_arb.plots import plot_equity, plot_heatmap, plot_premium  # noqa: E402
from adr_hk_arb.strategy import compute_premium  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hk", help="HK ticker, e.g. 0700.HK (needs --adr)")
    src.add_argument("--csv", help="long CSV with columns date,series,close")
    src.add_argument("--synthetic", action="store_true", help="synthetic demo data")
    ap.add_argument("--adr", help="ADR ticker, e.g. TCEHY")
    ap.add_argument("--ratio", type=float, default=0.5, help="ADRs per HK share")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--train-days", type=int, default=504)
    ap.add_argument("--test-days", type=int, default=126)
    ap.add_argument("--max-hold-days", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.synthetic:
        label, series = "synthetic", make_synthetic_pair(adr_per_share=args.ratio)
    elif args.csv:
        label, series = Path(args.csv).stem, load_csv(args.csv)
    else:
        if not args.adr:
            ap.error("--hk requires --adr")
        label, series = f"{args.hk}_{args.adr}", fetch_yahoo(args.hk, args.adr, args.start, args.end)

    panel = align_sessions(**series)
    costs = CostModel()
    out = Path(args.out) / label
    out.mkdir(parents=True, exist_ok=True)
    print(f"{label}: {len(panel)} aligned days {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"costs: {costs.describe()}")

    # 1) Naive in-sample optimum (what a single-pass grid search reports).
    best, grid = in_sample_optimum(panel, args.ratio, costs, max_hold_days=args.max_hold_days)
    grid.to_csv(out / "in_sample_grid.csv", index=False)
    plot_heatmap(grid, "sharpe", f"{label}: in-sample Sharpe by threshold", str(out / "heatmap_sharpe.png"))
    plot_heatmap(grid.assign(annual_return_pct=grid["annual_return"] * 100), "annual_return_pct",
                 f"{label}: in-sample annual return (%)", str(out / "heatmap_return.png"))

    prem = compute_premium(panel, args.ratio)
    curves = {}
    report = {"pair": label, "days": len(panel), "costs": costs.describe()}
    if best is not None:
        rule = ThresholdRule(prem.mean(), prem.std(), best["entry_k"], best["exit_k"], args.max_hold_days)
        ins = backtest(panel, args.ratio, rule, costs)
        curves["in-sample optimum (look-ahead)"] = ins.daily["equity"]
        report["in_sample"] = {"entry_k": rule.entry_k, "exit_k": rule.exit_k, **summarize(ins)}
        plot_premium(prem, rule, f"{label}: premium and in-sample bands", str(out / "premium.png"))
        ins.trades.to_csv(out / "in_sample_trades.csv", index=False)

    # 2) Walk-forward: thresholds fitted on the past only.
    wf = walk_forward(panel, args.ratio, args.train_days, args.test_days, costs,
                      max_hold_days=args.max_hold_days)
    wf.folds.to_csv(out / "walk_forward_folds.csv", index=False)
    wf.oos.trades.to_csv(out / "walk_forward_trades.csv", index=False)
    report["walk_forward"] = summarize(wf.oos)
    curves["walk-forward out-of-sample"] = wf.oos.daily["equity"]
    plot_equity(curves, f"{label}: equity after costs", str(out / "equity.png"))

    (out / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    for k in ("in_sample", "walk_forward"):
        if k in report:
            s = report[k]
            print(f"{k:>13}: Sharpe {s['sharpe']:.2f} | annual {s['annual_return']:.1%} | "
                  f"maxDD {s['max_drawdown']:.1%} | trades {s['n_trades']}")
    print(f"outputs -> {out}/")


if __name__ == "__main__":
    main()
