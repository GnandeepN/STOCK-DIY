from __future__ import annotations

"""
capital.py — simple capital management and position sizing helpers.

Sizing modes:
  - value: fixed notional per trade (existing behavior)
  - capital: equal-weight capital slice = total_capital / max_concurrent
  - atr-risk: risk-based sizing using SL distance and risk% of capital
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CapitalRules:
    total_capital: float = 100000.0
    risk_per_trade_pct: float = 1.0  # percent of capital
    max_concurrent: int = 10
    max_notional_per_trade: Optional[float] = None  # hard cap
    max_pct_cap_per_instrument: Optional[float] = None  # e.g., 10 (% of capital)


def cap_slice(r: CapitalRules) -> float:
    """Equal-weight slice of capital for one position."""
    m = max(1, int(r.max_concurrent))
    return r.total_capital / m


def size_value(price: float, notional: float) -> int:
    if price <= 0 or notional <= 0:
        return 0
    return max(1, int(notional // price))


def size_capital(price: float, rules: CapitalRules) -> int:
    notional = cap_slice(rules)
    if rules.max_notional_per_trade:
        notional = min(notional, float(rules.max_notional_per_trade))
    return size_value(price, notional)


def size_atr_risk(price: float, sl: Optional[float], rules: CapitalRules, side: str) -> int:
    """ATR risk sizing: qty = risk_amount / stop_distance.
    BUY: stop_distance = price - sl; SELL: stop_distance = sl - price.
    """
    if price <= 0 or sl is None:
        return 0
    risk_amount = max(0.0, rules.total_capital * (rules.risk_per_trade_pct / 100.0))
    stop_dist = (price - sl) if side.upper() == "BUY" else (sl - price)
    if stop_dist <= 0:
        return 0
    qty = int(risk_amount // stop_dist)
    if qty <= 0:
        return 0
    # Optional notional cap
    if rules.max_notional_per_trade:
        qty = min(qty, int(float(rules.max_notional_per_trade) // price) or 1)
    # Percent of capital clamp
    if rules.max_pct_cap_per_instrument and rules.max_pct_cap_per_instrument > 0:
        cap = rules.total_capital * (rules.max_pct_cap_per_instrument / 100.0)
        qty = min(qty, int(cap // price) or 1)
    # Do not cap by equal-weight slice here; ATR-risk sizing should reflect risk budget.
    # Any portfolio-level caps are enforced upstream (planner/execution).
    return max(1, qty)


def size_kelly(price: float, sl: Optional[float], tp: Optional[float], prob_up: Optional[float], side: str, rules: CapitalRules, max_frac: float = 0.1) -> int:
    """Kelly sizing using edge from prob and payoff ratio.
    Approx payoff ratio R = (tp - price) / (price - sl) for BUY (and inverse for SELL).
    Fraction f* = p - (1-p)/R (bounded to [0, max_frac]).
    """
    if price <= 0 or sl is None or tp is None or prob_up is None:
        return 0
    if side.upper() == "BUY":
        risk = max(1e-6, price - sl)
        reward = max(1e-6, tp - price)
        p = float(prob_up)
    else:
        risk = max(1e-6, sl - price)
        reward = max(1e-6, price - tp)
        p = 1 - float(prob_up)
    R = reward / risk if risk > 0 else 0
    if R <= 0:
        return 0
    f = p - (1 - p) / R
    # Confidence adjustment: scale by (prob_up - 0.5), floor at 0
    conf = max(0.0, (prob_up or 0.5) - 0.5)
    f = f * (1.0 + conf)
    f = max(0.0, min(float(max_frac), float(f)))
    notional = rules.total_capital * f
    if rules.max_notional_per_trade:
        notional = min(notional, float(rules.max_notional_per_trade))
    if rules.max_pct_cap_per_instrument and rules.max_pct_cap_per_instrument > 0:
        notional = min(notional, rules.total_capital * (rules.max_pct_cap_per_instrument / 100.0))
    return size_value(price, notional)


def size_vol_target(price: float, atr: Optional[float], rules: CapitalRules, target_vol_pct: float = 20.0) -> int:
    """Volatility targeting: allocate so that ATR/price ≈ target_vol (% annualized proxy simplified).
    notional ~ capital * (target_vol / realized_vol). We approximate realized_vol ~ ATR/price.
    """
    if price <= 0 or atr is None or atr <= 0:
        return 0
    realized = atr / price * 100  # percent
    if realized <= 0:
        return 0
    scale = target_vol_pct / realized
    notional = rules.total_capital * min(1.0, max(0.0, scale))
    if rules.max_notional_per_trade:
        notional = min(notional, float(rules.max_notional_per_trade))
    if rules.max_pct_cap_per_instrument and rules.max_pct_cap_per_instrument > 0:
        notional = min(notional, rules.total_capital * (rules.max_pct_cap_per_instrument / 100.0))
    return size_value(price, notional)
