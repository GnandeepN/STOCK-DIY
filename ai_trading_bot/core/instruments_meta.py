from __future__ import annotations

"""
Optional metadata loader for shortability, price bands, and freeze quantities.

If reports/instruments_meta.csv exists with columns:
  ticker, shortable, band_low, band_high, freeze_qty, tick_size
we'll use it to preflight shorts and limit-price guards.
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
META = BASE / "reports" / "instruments_meta.csv"

_CACHE = None


def load_meta():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if META.exists():
        try:
            df = pd.read_csv(META)
            _CACHE = {str(r.ticker).strip().upper(): r._asdict() for r in df.itertuples(index=False)}
            return _CACHE
        except Exception:
            _CACHE = {}
            return _CACHE
    _CACHE = {}
    return _CACHE


def preflight_short(ticker: str, qty: int, price: float) -> tuple[bool, str]:
    """Return (ok, reason). If metadata not available, allow.
    """
    meta = load_meta().get(str(ticker).upper())
    if not meta:
        return True, "no_meta"
    if str(meta.get("shortable", True)).lower() in ("false", "0", "no"): 
        return False, "not_shortable"
    fz = meta.get("freeze_qty")
    if fz and int(fz) > 0 and int(qty) > int(fz):
        return False, "freeze_exceeded"
    lo = meta.get("band_low"); hi = meta.get("band_high")
    if lo is not None and hi is not None and price:
        try:
            if not (float(lo) <= float(price) <= float(hi)):
                return False, "price_outside_band"
        except Exception:
            pass
    return True, "ok"

