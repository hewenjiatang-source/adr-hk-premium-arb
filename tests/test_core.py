"""Core correctness tests. Run with `pytest -q`."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adr_hk_arb import (  # noqa: E402
    CostModel, ThresholdRule, align_sessions, backtest, compute_premium,
    generate_positions, make_synthetic_pair, walk_forward,
)
from adr_hk_arb.metrics import max_drawdown  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return align_sessions(**make_synthetic_pair(n_days=800, adr_per_share=0.5, seed=3))


def test_premium_uses_ratio_and_fx():
    d = pd.to_datetime(["2024-01-02", "2024-01-03"])
    hk = pd.Series([100.0, 100.0], index=d)
    adr = pd.Series([25.8, 25.8], index=d)          # 1 ADR = 2 shares -> ratio 0.5
    fx = pd.Series([7.8, 7.8], index=d)
    panel = align_sessions(hk, adr, fx)
    prem = compute_premium(panel, adr_per_share=0.5)
    # HK-equivalent = 25.8 * 0.5 * 7.8 = 100.62 -> +0.62%
    assert prem.iloc[0] == pytest.approx(0.0062)


def test_alignment_has_no_lookahead():
    """The signal columns must come from US/FX dates strictly before the HK date."""
    s = make_synthetic_pair(n_days=50)
    panel = align_sessions(**s)
    adr = s["adr"]
    for d, row in panel.iterrows():
        prior = adr[adr.index < d]
        assert row["adr_sig"] == prior.iloc[-1]
        assert row["exec_date"] >= d


def test_positions_do_not_depend_on_future(panel):
    prem = compute_premium(panel, 0.5)
    rule = ThresholdRule(prem.mean(), prem.std(), 1.0, 0.0)
    full = generate_positions(prem, rule)
    cut = 400
    perturbed = prem.copy()
    perturbed.iloc[cut:] = np.random.default_rng(0).normal(0, 0.05, len(prem) - cut)
    part = generate_positions(perturbed, rule)
    pd.testing.assert_series_equal(full.iloc[:cut], part.iloc[:cut])


def test_hysteresis_rule():
    idx = pd.bdate_range("2024-01-01", periods=6)
    prem = pd.Series([0.0, 0.03, 0.02, 0.011, 0.009, 0.03], index=idx)
    rule = ThresholdRule(mean=0.0, std=0.01, entry_k=2.5, exit_k=1.0)
    pos = generate_positions(prem, rule).tolist()
    # enter above 2.5%, hold while above 1%, exit at 0.9%, re-enter at 3%
    assert pos == [0, 1, 1, 1, 0, 1]


def test_costs_only_reduce_returns(panel):
    prem = compute_premium(panel, 0.5)
    rule = ThresholdRule(prem.mean(), prem.std(), 1.0, 0.0)
    free = backtest(panel, 0.5, rule, CostModel.zero())
    paid = backtest(panel, 0.5, rule, CostModel())
    assert len(free.trades) == len(paid.trades) > 0
    assert paid.daily["equity"].iloc[-1] < free.daily["equity"].iloc[-1]
    diff = free.trades["trade_return"] - paid.trades["trade_return"]
    assert (diff >= 2 * CostModel().per_side - 1e-12).all()


def test_trade_return_matches_static_hedge(panel):
    """With zero costs, a closed trade's return equals the static-hedge formula."""
    prem = compute_premium(panel, 0.5)
    rule = ThresholdRule(prem.mean(), prem.std(), 1.0, 0.0)
    res = backtest(panel, 0.5, rule, CostModel.zero())
    t = res.trades[res.trades["closed"]].iloc[0]
    a, b = panel.loc[t["entry_date"]], panel.loc[t["exit_date"]]
    expected = (b["hk"] / a["hk"] - 1) - (b["adr_exec"] * b["fx_exec"] / (a["adr_exec"] * a["fx_exec"]) - 1)
    assert t["trade_return"] == pytest.approx(expected, rel=1e-9)


def test_invalid_rule_rejected():
    with pytest.raises(ValueError):
        ThresholdRule(0.0, 0.01, entry_k=0.5, exit_k=0.5)


def test_walk_forward_uses_only_past_statistics(panel):
    wf = walk_forward(panel, 0.5, train_days=300, test_days=100)
    assert len(wf.folds) >= 4
    for f in wf.folds.itertuples():
        train = panel.loc[str(f.train_start):str(f.train_end)]
        assert f.premium_mean == pytest.approx(compute_premium(train, 0.5).mean())
        assert f.train_end < f.test_start
    assert wf.oos.daily.index.is_monotonic_increasing


def test_empty_series_gives_clear_error():
    s = make_synthetic_pair(n_days=20)
    with pytest.raises(ValueError):
        align_sessions(s["hk"], s["adr"].iloc[:0], s["fx"])


def test_equity_never_negative_with_wrong_ratio(panel):
    """A badly wrong ADR ratio creates huge hedge P&L; equity must stay >= 0."""
    prem = compute_premium(panel, 5.0)
    rule = ThresholdRule(prem.mean(), prem.std(), 0.2, -0.2)
    res = backtest(panel, 5.0, rule)
    assert (res.daily["equity"] >= 0).all()


def test_walk_forward_training_excludes_unsettled_adr_fills():
    """US closed on the last training day: its ADR fill lands on the first test day.

    That row's P&L is not known when the test window starts, so it must not be
    used to choose parameters. Scrambling the US close on the test start date
    must leave the fold-0 parameters unchanged.
    """
    train_days, test_days = 300, 100
    s = make_synthetic_pair(n_days=800, adr_per_share=0.5, seed=3)
    dates = align_sessions(**s).index
    us_holiday, test_start = dates[train_days - 1], dates[train_days]
    adr = s["adr"].drop(us_holiday)
    panel = align_sessions(s["hk"], adr, s["fx"])
    assert panel.loc[us_holiday, "exec_date"] == test_start  # HK open, US closed

    wf = walk_forward(panel, 0.5, train_days=train_days, test_days=test_days)
    f0 = wf.folds.iloc[0]
    assert f0["test_start"] == test_start.date()
    assert f0["train_end"] == dates[train_days - 2].date()
    assert f0["train_rows_dropped"] == 1

    shocked = adr.copy()
    shocked.loc[test_start] *= 1.5
    wf2 = walk_forward(align_sessions(s["hk"], shocked, s["fx"]), 0.5,
                       train_days=train_days, test_days=test_days)
    cols = ["premium_mean", "premium_std", "entry_k", "exit_k", "train_sharpe"]
    pd.testing.assert_series_equal(wf.folds.iloc[0][cols], wf2.folds.iloc[0][cols])


def test_walk_forward_training_fills_precede_test_start(panel):
    wf = walk_forward(panel, 0.5, train_days=300, test_days=100)
    for f in wf.folds.itertuples():
        train = panel.loc[str(f.train_start):str(f.train_end)]
        assert (train["exec_date"] < pd.Timestamp(f.test_start)).all()


def test_max_drawdown_counts_initial_capital():
    idx = pd.bdate_range("2024-01-01", periods=2)
    assert max_drawdown(pd.Series([0.90, 0.95], index=idx)) == pytest.approx(-0.10)
    assert max_drawdown(pd.Series([1.10, 0.99], index=idx)) == pytest.approx(-0.10)
    assert max_drawdown(pd.Series([1.00, 1.05], index=idx)) == 0.0


def test_first_day_cost_shows_in_drawdown():
    """Entering on the first row pays costs immediately; that loss is a drawdown."""
    idx = pd.bdate_range("2024-01-01", periods=3)
    panel = pd.DataFrame({"hk": [100.0] * 3, "adr_sig": [30.0] * 3, "fx_sig": [7.8] * 3,
                          "adr_exec": [30.0] * 3, "fx_exec": [7.8] * 3, "exec_date": idx}, index=idx)
    rule = ThresholdRule(mean=0.0, std=0.1, entry_k=0.5, exit_k=-10.0)  # premium +17% -> enter day 1
    res = backtest(panel, 0.5, rule, CostModel())
    assert res.daily["position"].iloc[0] == 1
    assert res.daily["equity"].iloc[0] == pytest.approx(1 - CostModel().per_side)
    assert max_drawdown(res.daily["equity"]) <= -CostModel().per_side
