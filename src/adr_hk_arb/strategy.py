"""Premium signal and a threshold mean-reversion backtest.

Conventions
-----------
premium_t = (ADR_close_prev_US * adr_per_share * USDHKD) / HK_close_t - 1

`adr_per_share` is the number of ADRs that represent one HK share
(e.g. BABA: 1 ADR = 8 shares -> 0.125). A positive premium means the ADR is
rich relative to Hong Kong.

Position +1 ("sell premium"): long HK shares, short an equal HKD notional of
ADRs. Position -1 ("buy premium"): the mirror trade — disabled by default
because it requires shorting the HK line, which is often expensive or
unavailable.

The hedge is static within a trade (share counts fixed at entry), and P&L is
measured against the notional of one leg. Equity compounds across trades.

Execution timing is approximated, not simulated: the HK leg fills at the same
HK close the signal is computed from, the ADR leg at the next US close, and
both are booked on the HK date row. Pending ADR orders and the overnight
one-legged position are not modelled as separate state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel


def compute_premium(panel: pd.DataFrame, adr_per_share: float) -> pd.Series:
    """Signal premium using only data known at the HK close."""
    hk_equiv = panel["adr_sig"] * adr_per_share * panel["fx_sig"]
    return (hk_equiv / panel["hk"] - 1.0).rename("premium")


@dataclass(frozen=True)
class ThresholdRule:
    """Enter when premium > mean + entry_k*std, exit when premium <= mean + exit_k*std.

    `mean` and `std` must come from data *before* the evaluation period
    (see `walkforward.py`); passing full-sample statistics is look-ahead.
    """
    mean: float
    std: float
    entry_k: float
    exit_k: float
    max_hold_days: int | None = None  # calendar days; None = no time stop
    allow_buy_premium: bool = False

    def __post_init__(self):
        if self.entry_k <= self.exit_k:
            raise ValueError("entry_k must be greater than exit_k")
        if self.std <= 0:
            raise ValueError("std must be positive")

    @property
    def sell_entry(self) -> float:
        return self.mean + self.entry_k * self.std

    @property
    def sell_exit(self) -> float:
        return self.mean + self.exit_k * self.std

    @property
    def buy_entry(self) -> float:
        return self.mean - self.entry_k * self.std

    @property
    def buy_exit(self) -> float:
        return self.mean - self.exit_k * self.std


def generate_positions(premium: pd.Series, rule: ThresholdRule) -> pd.Series:
    """Sequential state machine (hysteresis between entry and exit bands).

    The position decided at row t is assumed filled at row t's prices (the same
    HK close and the following US close) and earns P&L from row t+1 onwards. A position is never
    flipped from +1 to -1 in a single row: it exits first and may re-enter on
    a later row.
    """
    p = premium.to_numpy()
    dates = premium.index
    pos = np.zeros(len(p), dtype=np.int8)
    state, entry_date = 0, None
    for i, x in enumerate(p):
        if np.isnan(x):
            pos[i] = state
            continue
        timed_out = (
            state != 0
            and rule.max_hold_days is not None
            and (dates[i] - entry_date).days >= rule.max_hold_days
        )
        if state == 1 and (x <= rule.sell_exit or timed_out):
            state = 0
        elif state == -1 and (x >= rule.buy_exit or timed_out):
            state = 0
        elif state == 0:
            if x > rule.sell_entry:
                state, entry_date = 1, dates[i]
            elif rule.allow_buy_premium and x < rule.buy_entry:
                state, entry_date = -1, dates[i]
        pos[i] = state
    return pd.Series(pos, index=dates, name="position")


@dataclass
class BacktestResult:
    daily: pd.DataFrame   # premium, position, pnl, ret, equity per row
    trades: pd.DataFrame  # one row per trade

    @property
    def returns(self) -> pd.Series:
        return self.daily["ret"]


def backtest(
    panel: pd.DataFrame,
    adr_per_share: float,
    rule: ThresholdRule,
    costs: CostModel = CostModel(),
    close_at_end: bool = False,
) -> BacktestResult:
    """Run the rule over `panel`.

    close_at_end=True forces any open position flat on the last row (paying the
    exit cost), so that separate test windows in a walk-forward don't leave
    un-costed open trades.
    """
    premium = compute_premium(panel, adr_per_share)
    pos = generate_positions(premium, rule)
    if close_at_end and len(pos):
        pos.iloc[-1] = 0

    hk = panel["hk"].to_numpy()
    adr_hkd = (panel["adr_exec"] * panel["fx_exec"]).to_numpy()
    cur = pos.to_numpy().astype(float)
    prev = np.concatenate([[0.0], cur[:-1]])

    entering = (prev == 0) & (cur != 0)
    exiting = (prev != 0) & (cur == 0)
    active = (prev != 0) | (cur != 0)

    trade_id = np.cumsum(entering).astype(float)
    trade_id[~active] = np.nan

    # Entry reference prices, carried forward within each trade.
    hk_ref = pd.Series(np.where(entering, hk, np.nan)).ffill().to_numpy()
    adr_ref = pd.Series(np.where(entering, adr_hkd, np.nan)).ffill().to_numpy()

    d_hk = np.diff(hk, prepend=hk[0])
    d_adr = np.diff(adr_hkd, prepend=adr_hkd[0])
    days = np.diff(panel.index.to_numpy()).astype("timedelta64[D]").astype(float)
    days = np.concatenate([[0.0], days])

    with np.errstate(invalid="ignore", divide="ignore"):
        price_pnl = np.where(prev != 0, prev * (d_hk / hk_ref - d_adr / adr_ref), 0.0)
    trading_cost = costs.per_side * (entering.astype(float) + exiting.astype(float))
    carry = np.where(prev != 0, costs.daily_carry * days, 0.0)
    pnl = price_pnl - trading_cost - carry  # fraction of entry notional

    daily = pd.DataFrame(
        {
            "premium": premium.to_numpy(),
            "position": cur.astype(int),
            "trade_id": trade_id,
            "pnl": pnl,
        },
        index=panel.index,
    )
    # Convert entry-notional P&L to returns on current equity so trades compound.
    cum_before = daily.groupby("trade_id")["pnl"].cumsum() - daily["pnl"]
    base = (1.0 + cum_before.fillna(0.0)).clip(lower=1e-9)
    daily["ret"] = np.where(daily["trade_id"].notna(), daily["pnl"] / base, 0.0)
    # A day can at most wipe out the account; without this floor a runaway
    # losing trade (e.g. a mis-specified ADR ratio) makes equity negative.
    daily["ret"] = daily["ret"].clip(lower=-1.0)
    daily["equity"] = (1.0 + daily["ret"]).cumprod()

    trades = _trade_table(daily, exiting)
    return BacktestResult(daily=daily, trades=trades)


def _trade_table(daily: pd.DataFrame, exiting: np.ndarray) -> pd.DataFrame:
    rows = daily[daily["trade_id"].notna()].copy()
    if rows.empty:
        return pd.DataFrame(
            columns=["entry_date", "exit_date", "side", "calendar_days", "trade_return", "closed"]
        )
    rows["is_exit"] = exiting[daily["trade_id"].notna().to_numpy()]
    g = rows.groupby("trade_id")
    out = pd.DataFrame(
        {
            "entry_date": g.apply(lambda x: x.index[0], include_groups=False),
            "exit_date": g.apply(lambda x: x.index[-1], include_groups=False),
            "side": g["position"].first().map({1: "sell_premium", -1: "buy_premium"}),
            "trade_return": g["pnl"].sum(),
            "closed": g["is_exit"].any(),
        }
    )
    out["calendar_days"] = (out["exit_date"] - out["entry_date"]).dt.days
    out = out[["entry_date", "exit_date", "side", "calendar_days", "trade_return", "closed"]]
    return out.reset_index(drop=True)
