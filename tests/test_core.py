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
