"""Transaction and financing cost assumptions.

All numbers are public fee schedules or illustrative assumptions — change them to
match your own broker. They are kept in one place so the effect of each cost
component on the strategy can be tested separately (see `CostModel.zero()`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    # --- Hong Kong leg, per side, as a fraction of notional -----------------
    # Stamp duty was 0.13% until the Nov 2023 cut and 0.10% after. Check the current
    # HKEX / IRD schedule before using these.
    hk_stamp_duty: float = 0.10 / 100
    hk_sfc_levy: float = 0.0027 / 100
    hk_afrc_levy: float = 0.00015 / 100
    hk_trading_fee: float = 0.00565 / 100
    hk_commission: float = 1.0 * BPS

    # --- US ADR leg, per side ------------------------------------------------
    us_commission: float = 1.0 * BPS
    us_sec_fee: float = 0.0  # tiny; set if you want to be exact (sells only)

    # --- Execution slippage, per side, per leg --------------------------------
    slippage: float = 2.0 * BPS

    # --- Financing, annualised, charged per calendar day held ----------------
    # Cost to fund the long HK position (spread over the cash rate) and
    # cost to borrow the ADR you are short. Illustrative defaults only.
    long_funding_rate: float = 50 * BPS
    short_borrow_rate: float = 30 * BPS

    @property
    def hk_per_side(self) -> float:
        return (
            self.hk_stamp_duty + self.hk_sfc_levy + self.hk_afrc_levy
            + self.hk_trading_fee + self.hk_commission + self.slippage
        )

    @property
    def us_per_side(self) -> float:
        return self.us_commission + self.us_sec_fee + self.slippage

    @property
    def per_side(self) -> float:
        """Cost of opening OR closing one hedged unit (both legs)."""
        return self.hk_per_side + self.us_per_side

    @property
    def daily_carry(self) -> float:
        """Financing cost per calendar day for one hedged unit."""
        return (self.long_funding_rate + self.short_borrow_rate) / 365.0

    @classmethod
    def zero(cls) -> "CostModel":
        return cls(**{f: 0.0 for f in cls.__dataclass_fields__})

    def with_(self, **kwargs) -> "CostModel":
        return replace(self, **kwargs)

    def describe(self) -> str:
        return (
            f"HK {self.hk_per_side / BPS:.2f} bps/side, US {self.us_per_side / BPS:.2f} bps/side, "
            f"round trip {2 * self.per_side / BPS:.2f} bps, "
            f"carry {(self.long_funding_rate + self.short_borrow_rate) / BPS:.0f} bps/yr"
        )
