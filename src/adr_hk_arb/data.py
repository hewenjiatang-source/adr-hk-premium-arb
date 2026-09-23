"""Price loading and HK/US session alignment.

The core difficulty of an HK-vs-ADR backtest is that the two markets never
trade at the same time. Hong Kong closes at 16:00 HKT; the US regular session
runs roughly 21:30-04:00 HKT (22:30-05:00 in US winter time). This module builds a
panel indexed by *Hong Kong trading date* in which every column is something
that is actually knowable (or tradable) at that point in time:

    hk        HK close on date d                          (known at 16:00 HKT, d)
    adr_sig   last US close on a US date strictly < d     (known at 16:00 HKT, d)
    fx_sig    last USD/HKD fix on a date strictly < d     (known at 16:00 HKT, d)
    adr_exec  first US close on a US date >= d            (the next US session,
                                                           when the ADR leg fills)
    fx_exec   USD/HKD on the same date as adr_exec

The signal only uses `hk`, `adr_sig` and `fx_sig`. The P&L uses `hk` and
`adr_exec`, so the overnight legging risk between buying HK and selling the ADR
is included in the results instead of being assumed away.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

YAHOO_FX_TICKER = "HKD=X"  # HKD per 1 USD


def _close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Pull one ticker's close out of a yfinance download (single or multi-index)."""
    if isinstance(df.columns, pd.MultiIndex):
        s = df["Close"][ticker]
    else:
        s = df["Close"]
    s = s.dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.astype(float)


def fetch_yahoo(hk_ticker: str, adr_ticker: str, start: str, end: str | None = None) -> dict[str, pd.Series]:
    """Download split-adjusted daily closes for one pair plus USD/HKD from Yahoo Finance.

    Closes are split-adjusted but NOT dividend-adjusted (auto_adjust=False): the
    premium should compare traded prices, and dividends are handled as an
    explicit limitation (see README).
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pip install yfinance to download data, or use load_csv()") from exc

    raw = yf.download(
        [hk_ticker, adr_ticker, YAHOO_FX_TICKER],
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    return {
        "hk": _close_series(raw, hk_ticker),
        "adr": _close_series(raw, adr_ticker),
        "fx": _close_series(raw, YAHOO_FX_TICKER),
    }


def load_csv(path: str | Path) -> dict[str, pd.Series]:
    """Load a long-format CSV with columns: date, series (hk|adr|fx), close.

    Use this for data from Bloomberg / Refinitiv exports. Each series keeps its
    own trading calendar; alignment happens in `align_sessions`.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    out = {}
    for name in ("hk", "adr", "fx"):
        s = df.loc[df["series"] == name].set_index("date")["close"].sort_index()
        if s.empty:
            raise ValueError(f"{path}: no rows for series '{name}'")
        out[name] = s.astype(float)
    return out


def align_sessions(hk: pd.Series, adr: pd.Series, fx: pd.Series) -> pd.DataFrame:
    """Align HK, US and FX series onto HK trading dates without look-ahead.

    See module docstring for column definitions. Rows at the edges where any
    column is unavailable are dropped.
    """
    hk = hk.sort_index().dropna()
    adr = adr.sort_index().dropna()
    fx = fx.sort_index().dropna()

    panel = pd.DataFrame({"hk": hk})
    panel.index.name = "date"
    left = panel.reset_index()

    def asof(series: pd.Series, name: str, direction: str, allow_exact: bool) -> pd.Series:
        right = series.rename(name).rename_axis("src_date").reset_index()
        merged = pd.merge_asof(
            left[["date"]],
            right,
            left_on="date",
            right_on="src_date",
            direction=direction,
            allow_exact_matches=allow_exact,
        )
        return merged.set_index("date")

    sig = asof(adr, "adr_sig", "backward", allow_exact=False)
    fxs = asof(fx, "fx_sig", "backward", allow_exact=False)
    exe = asof(adr, "adr_exec", "forward", allow_exact=True)

    panel["adr_sig"] = sig["adr_sig"]
    panel["fx_sig"] = fxs["fx_sig"]
    panel["adr_exec"] = exe["adr_exec"]
    panel["exec_date"] = exe["src_date"]
    # FX on the day the ADR leg fills (last fix on or before that date).
    fx_exec = pd.merge_asof(
        exe[["src_date"]].rename(columns={"src_date": "d"}).reset_index().sort_values("d"),
        fx.rename("fx_exec").rename_axis("d").reset_index(),
        on="d",
        direction="backward",
    ).set_index("date")["fx_exec"]
    panel["fx_exec"] = fx_exec.reindex(panel.index)

    # Guard against stale quotes: drop rows where the "previous US close" is
    # more than 5 calendar days old (e.g. long holidays, halted ADRs).
    staleness = (panel.index - sig["src_date"]).dt.days
    panel = panel[staleness.le(5).values]
    return panel.dropna()


def make_synthetic_pair(
    n_days: int = 1000,
    adr_per_share: float = 0.5,
    premium_mean: float = 0.0,
    premium_vol: float = 0.01,
    half_life_days: float = 3.0,
    seed: int = 7,
) -> dict[str, pd.Series]:
    """Synthetic HK/ADR/FX closes with a mean-reverting premium.

    Used for tests and for the demo figures; it is NOT a claim about any real
    pair. The ADR close on date t is priced off the HK close of t+1 (the next HK
    session overlaps the information the US session trades on), which gives the
    US series a realistic one-session offset.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days + 1)
    hk = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.018, n_days + 1)))
    phi = 0.5 ** (1.0 / half_life_days)
    eps = rng.normal(0, premium_vol * np.sqrt(1 - phi**2), n_days + 1)
    prem = np.empty(n_days + 1)
    prem[0] = premium_mean
    for i in range(1, n_days + 1):
        prem[i] = premium_mean + phi * (prem[i - 1] - premium_mean) + eps[i]
    fx = 7.8 + rng.normal(0, 0.002, n_days + 1)
    # ADR (USD) on US date t reflects HK value of the next HK session.
    hk_next = np.append(hk[1:], hk[-1])
    adr = hk_next * (1 + prem) / (adr_per_share * fx)
    return {
        "hk": pd.Series(hk, index=dates),
        "adr": pd.Series(adr, index=dates),
        "fx": pd.Series(fx, index=dates),
    }
