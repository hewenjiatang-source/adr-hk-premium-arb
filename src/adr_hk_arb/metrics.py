"""Performance statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import BacktestResult

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0 or np.isnan(sd):
        return float("nan")
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def annual_return(equity: pd.Series) -> float:
    n = len(equity)
    if n < 2:
        return float("nan")
    return float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1.0)


def period_win_rate(daily: pd.DataFrame, freq: str) -> float:
    """Share of calendar periods with positive return, among periods with any exposure."""
    active = daily["trade_id"].notna()
    grouped = pd.DataFrame({"ret": daily["ret"], "active": active}).resample(freq)
    period_ret = grouped["ret"].apply(lambda r: (1 + r).prod() - 1)
    had_exposure = grouped["active"].any()
    period_ret = period_ret[had_exposure]
    if period_ret.empty:
        return float("nan")
    return float((period_ret > 0).mean())


def _max_streak(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def summarize(result: BacktestResult, full: bool = True) -> dict:
    """Performance summary. full=False skips the slower calendar win-rate stats (used in grid search)."""
    d, t = result.daily, result.trades
    years = max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    tr = t["trade_return"].to_numpy() if len(t) else np.array([])
    out = {
        "start": d.index[0].date(),
        "end": d.index[-1].date(),
        "sharpe": sharpe_ratio(d["ret"]),
        "annual_return": annual_return(d["equity"]),
        "total_return": float(d["equity"].iloc[-1] - 1.0),
        "max_drawdown": max_drawdown(d["equity"]),
        "n_trades": int(len(t)),
        "trades_per_year": len(t) / years,
        "avg_trade_return": float(tr.mean()) if len(tr) else float("nan"),
        "hit_rate": float((tr > 0).mean()) if len(tr) else float("nan"),
        "max_trade_gain": float(tr.max()) if len(tr) else float("nan"),
        "max_trade_loss": float(tr.min()) if len(tr) else float("nan"),
        "max_consecutive_wins": _max_streak(tr > 0),
        "max_consecutive_losses": _max_streak(tr <= 0),
        "max_hold_days": int(t["calendar_days"].max()) if len(t) else 0,
        "time_in_market": float(d["trade_id"].notna().mean()),
    }
    if full:
        out.update({
            "weekly_win_rate": period_win_rate(d, "W"),
            "monthly_win_rate": period_win_rate(d, "ME"),
            "yearly_win_rate": period_win_rate(d, "YE"),
        })
    return out
