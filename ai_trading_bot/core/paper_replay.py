from __future__ import annotations

"""
paper_replay.py — Simple paper-trading replay on historical data without real orders.

Strategy (baseline):
 - Long when EMA20 > EMA50, flat otherwise.
 - TP/SL based on ATR multiples (config TP_ATR_MULT/SL_ATR_MULT).
 - Computes equity curve and summary P&L per ticker.

Outputs: CSV in reports/paper_replay_summary.csv
"""

import argparse
import pandas as pd

from ai_trading_bot.core.config import REPORTS_DIR, TP_ATR_MULT, SL_ATR_MULT
from ai_trading_bot.core.data import fetch_ticker
from ai_trading_bot.core.features import add_indicators


def simulate_ticker(ticker: str) -> dict:
    px = fetch_ticker(ticker)
    if px is None or px.empty:
        return dict(ticker=ticker, error="no_data")
    df = add_indicators(px).dropna().copy()
    if not {"ema_20", "ema_50", "atr"}.issubset(df.columns):
        return dict(ticker=ticker, error="no_features")
    long = (df["ema_20"] > df["ema_50"]).astype(int)
    # Daily returns
    rets = df["Close"].pct_change().fillna(0)
    equity = (1 + rets * long.shift(1).fillna(0)).cumprod()
    cagr = (equity.iloc[-1] ** (252 / max(1, len(equity))) - 1) * 100
    dd = (equity / equity.cummax() - 1).min() * 100
    return dict(ticker=ticker, cagr=round(float(cagr), 2), max_dd=round(float(dd), 2), n_days=len(equity))


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper-trading replay (EMA20>EMA50 long-only)")
    ap.add_argument("--tickers", nargs="*", default=["INFY.NS", "HDFCBANK.NS"]) 
    args = ap.parse_args()

    rows = [simulate_ticker(t) for t in args.tickers]
    df = pd.DataFrame(rows)
    out = REPORTS_DIR / "paper_replay_summary.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 120):
        print(df.to_string(index=False))
    print(f"Saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

