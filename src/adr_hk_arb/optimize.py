"""Parameter grid search and walk-forward (out-of-sample) evaluation.

Why walk-forward: choosing (entry_k, exit_k) as the best cell of a grid and then
reporting that cell's performance on the same data is an in-sample result. It
overstates what the rule would have earned live, because (a) the best of ~400
cells is partly luck and (b) the premium's mean/std used to set the bands were
computed with future data. `walk_forward` removes both: thresholds and the
mean/std are fitted on a trailing training window and then applied, unchanged,
to the next test window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel
from .metrics import summarize
from .strategy import BacktestResult, ThresholdRule, backtest, compute_premium

DEFAULT_ENTRY_GRID = np.round(np.linspace(0.1, 2.0, 20), 2)
DEFAULT_EXIT_GRID = np.round(np.linspace(-2.0, 2.0, 21), 2)


def grid_search(
    panel: pd.DataFrame,
    adr_per_share: float,
    mean: float,
    std: float,
    entry_grid=DEFAULT_ENTRY_GRID,
    exit_grid=DEFAULT_EXIT_GRID,
    costs: CostModel = CostModel(),
    max_hold_days: int | None = None,
    full_stats: bool = False,
) -> pd.DataFrame:
    """Backtest every (entry_k, exit_k) with entry_k > exit_k. One row per cell."""
    rows = []
    for e in entry_grid:
        for x in exit_grid:
            if e <= x:
                continue
            rule = ThresholdRule(mean, std, float(e), float(x), max_hold_days)
            stats = summarize(backtest(panel, adr_per_share, rule, costs), full=full_stats)
            rows.append({"entry_k": float(e), "exit_k": float(x), **stats})
    return pd.DataFrame(rows)


def pick_best(grid: pd.DataFrame, objective: str = "sharpe", min_trades: int = 5) -> pd.Series | None:
    eligible = grid[(grid["n_trades"] >= min_trades) & grid[objective].notna()]
    if eligible.empty:
        return None
    return eligible.loc[eligible[objective].idxmax()]


def in_sample_optimum(
    panel: pd.DataFrame,
    adr_per_share: float,
    costs: CostModel = CostModel(),
    objective: str = "sharpe",
    min_trades: int = 5,
    **grid_kwargs,
) -> tuple[pd.Series | None, pd.DataFrame]:
    """The naive approach: full-sample mean/std, best cell on the full sample.

    Kept on purpose as a baseline, to show how much of its performance survives
    out of sample.
    """
    prem = compute_premium(panel, adr_per_share)
    grid = grid_search(panel, adr_per_share, prem.mean(), prem.std(), costs=costs, **grid_kwargs)
    return pick_best(grid, objective, min_trades), grid


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame        # chosen parameters and test stats per fold
    oos: BacktestResult        # stitched out-of-sample daily results


def walk_forward(
    panel: pd.DataFrame,
    adr_per_share: float,
    train_days: int = 504,
    test_days: int = 126,
    costs: CostModel = CostModel(),
    objective: str = "sharpe",
    min_trades: int = 5,
    **grid_kwargs,
) -> WalkForwardResult:
    """Rolling train/test: fit on `train_days` rows, trade the next `test_days` rows.

    Parameters for a test window are chosen at the HK close of its first day.
    A training row's P&L depends on its ADR fill (`exec_date`), which can fall
    on or after that day, e.g. when the US market is closed on the last
    training date. Such rows are dropped from the training window, so the
    fitted parameters only use prices that were known when the test starts.
    """
    n = len(panel)
    if n < train_days + test_days:
        raise ValueError(f"need at least {train_days + test_days} rows, got {n}")

    fold_rows, dailies, trades = [], [], []
    start = 0
    while start + train_days < n:
        train = panel.iloc[start : start + train_days]
        test = panel.iloc[start + train_days : start + train_days + test_days]
        n_unsettled = int((train["exec_date"] >= test.index[0]).sum())
        train = train[train["exec_date"] < test.index[0]]
        prem = compute_premium(train, adr_per_share)
        mean, std = float(prem.mean()), float(prem.std())
        grid = grid_search(train, adr_per_share, mean, std, costs=costs, **grid_kwargs)
        best = pick_best(grid, objective, min_trades)

        if best is None:  # nothing tradable in training: stay flat
            rule = None
            daily = pd.DataFrame(
                {"premium": compute_premium(test, adr_per_share), "position": 0,
                 "trade_id": np.nan, "pnl": 0.0, "ret": 0.0},
                index=test.index,
            )
            test_trades = pd.DataFrame()
        else:
            rule = ThresholdRule(mean, std, best["entry_k"], best["exit_k"],
                                 grid_kwargs.get("max_hold_days"))
            res = backtest(test, adr_per_share, rule, costs, close_at_end=True)
            daily, test_trades = res.daily.copy(), res.trades

        fold = len(fold_rows)
        daily["trade_id"] = daily["trade_id"] + fold * 100_000  # keep ids unique across folds
        dailies.append(daily)
        trades.append(test_trades)
        fold_rows.append({
            "fold": fold,
            "train_start": train.index[0].date(),
            "train_end": train.index[-1].date(),
            "train_rows_dropped": n_unsettled,  # ADR fill not yet known at test start
            "test_start": test.index[0].date(),
            "test_end": test.index[-1].date(),
            "premium_mean": mean,
            "premium_std": std,
            "entry_k": None if best is None else best["entry_k"],
            "exit_k": None if best is None else best["exit_k"],
            "train_sharpe": None if best is None else best["sharpe"],
            "test_return": float((1 + daily["ret"]).prod() - 1),
            "test_trades": int(len(test_trades)),
        })
        start += test_days

    daily = pd.concat(dailies)
    daily["equity"] = (1 + daily["ret"]).cumprod()
    all_trades = pd.concat([t for t in trades if len(t)], ignore_index=True) if any(len(t) for t in trades) \
        else pd.DataFrame(columns=["entry_date", "exit_date", "side", "calendar_days", "trade_return", "closed"])
    return WalkForwardResult(folds=pd.DataFrame(fold_rows), oos=BacktestResult(daily=daily, trades=all_trades))
