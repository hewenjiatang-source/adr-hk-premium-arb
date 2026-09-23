"""Hong Kong vs. US ADR premium mean-reversion research toolkit."""
from .costs import CostModel
from .data import align_sessions, fetch_yahoo, load_csv, make_synthetic_pair
from .metrics import summarize
from .optimize import grid_search, in_sample_optimum, walk_forward
from .strategy import ThresholdRule, backtest, compute_premium, generate_positions

__all__ = [
    "CostModel", "ThresholdRule", "align_sessions", "backtest", "compute_premium",
    "fetch_yahoo", "generate_positions", "grid_search", "in_sample_optimum",
    "load_csv", "make_synthetic_pair", "summarize", "walk_forward",
]
