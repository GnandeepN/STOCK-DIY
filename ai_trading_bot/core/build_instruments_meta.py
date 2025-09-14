from __future__ import annotations

"""
Normalize Zerodha instruments.csv into reports/instruments_meta.csv

Usage:
  python build_instruments_meta.py --src /path/to/instruments.csv

Output columns: ticker, shortable, band_low, band_high, freeze_qty, tick_size
Notes:
 - If band columns are not available in the source, they are left blank.
 - Treats NSE equities as shortable=True by default; adjust manually if needed.
"""

import argparse
import pandas as pd
from pathlib import Path

from ai_trading_bot.core.instruments import resolve_symbol

BASE = Path(__file__).resolve().parent
OUT = BASE / "reports" / "instruments_meta.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to Zerodha instruments.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    # Expected columns vary; try to map
    # Columns we might use: exchange, tradingsymbol, tick_size, price_band_low, price_band_high, freeze_qty
    wanted = {}
    for c in df.columns:
        cl = c.strip().lower()
        wanted[cl] = c

    rows = []
    for r in df.itertuples(index=False):
        def get(col):
            name = wanted.get(col)
            return getattr(r, name) if name else None

        exch = str(get("exchange") or "").upper()
        tsym = str(get("tradingsymbol") or "").upper()
        if exch not in ("NSE", "BSE"):
            continue
        ticker = f"{tsym}.NS" if exch == "NSE" else f"{tsym}.BO"
        shortable = True if exch == "NSE" else False  # conservative default
        band_low = get("price_band_low")
        band_high = get("price_band_high")
        freeze_qty = get("freeze_qty")
        tick = get("tick_size")
        rows.append(dict(
            ticker=ticker, shortable=shortable, band_low=band_low, band_high=band_high,
            freeze_qty=freeze_qty, tick_size=tick,
        ))

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Saved → {OUT} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

