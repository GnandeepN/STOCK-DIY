from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
INSTR_CSV = BASE_DIR / "reports" / "kite_instruments.csv"


def resolve_symbol(nse_ticker: str) -> Tuple[str, str]:
    """Resolve our ticker like 'INFY.NS' to (exchange, tradingsymbol) for Zerodha.
    If a local instruments CSV exists (from Zerodha dump), prefer that mapping.
    Fallback: assume 'INFY.NS' -> ('NSE', 'INFY').
    """
    tsym = nse_ticker.replace(".NS", "").replace(" ", "").upper()
    if INSTR_CSV.exists():
        try:
            with INSTR_CSV.open() as f:
                r = csv.DictReader(f)
                for row in r:
                    if row.get("exchange") == "NSE" and row.get("tradingsymbol", "").upper() == tsym:
                        return "NSE", tsym
        except Exception:
            pass
    return "NSE", tsym


def to_yahoo_ticker(exchange: str, tradingsymbol: str) -> str:
    """Map broker exchange + tradingsymbol to a Yahoo Finance ticker.

    - NSE -> ".NS" suffix
    - BSE -> ".BO" suffix
    - Fallback: return tradingsymbol as-is if exchange is unknown
    """
    exch = str(exchange or "").upper()
    tsym = str(tradingsymbol or "").strip().upper()
    if not tsym:
        return tsym
    if exch == "NSE":
        return f"{tsym}.NS"
    if exch == "BSE":
        return f"{tsym}.BO"
    return tsym
