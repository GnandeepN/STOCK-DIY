from __future__ import annotations

"""
sync_holdings.py — Fetch Zerodha holdings, map to Yahoo tickers, and optionally
download/update their market data so your cache stays current.

Usage examples:
  # Just fetch and save watchlist CSV
  python sync_holdings.py

  # Fetch, save, and download/update cached price data
  python sync_holdings.py --download

Outputs:
  reports/holdings_tickers.csv with columns:
    ticker, exchange, tradingsymbol, qty, average_price, last_price, pnl
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

from instruments import to_yahoo_ticker
from data import update_all

try:
    from brokers import ZerodhaBroker
except Exception:
    ZerodhaBroker = None


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"


def _load_env_file():
    try:
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k and v and os.getenv(k) is None:
                    os.environ[k.strip()] = v.strip()
    except Exception:
        pass


def fetch_holdings() -> pd.DataFrame:
    if ZerodhaBroker is None:
        raise RuntimeError("kiteconnect not installed. Run `pip install kiteconnect`.")
    b = ZerodhaBroker()
    b.connect()
    rows = b.get_holdings() or []
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "exchange", "tradingsymbol", "qty", "average_price", "last_price", "pnl"
        ])
    df = pd.DataFrame(rows)
    # Normalize columns we care about
    need = ["exchange", "tradingsymbol", "quantity", "average_price", "last_price", "pnl"]
    for c in need:
        if c not in df.columns:
            df[c] = None
    df.rename(columns={"quantity": "qty"}, inplace=True)
    df["ticker"] = df.apply(lambda r: to_yahoo_ticker(r.get("exchange"), r.get("tradingsymbol")), axis=1)
    cols = ["ticker", "exchange", "tradingsymbol", "qty", "average_price", "last_price", "pnl"]
    out = df[cols].copy()
    # Drop rows with no mapped ticker
    out = out[out["ticker"].astype(str).str.len() > 0]
    return out


def main() -> int:
    _load_env_file()
    ap = argparse.ArgumentParser(description="Sync Zerodha holdings to watchlist and optionally download price data")
    ap.add_argument("--download", action="store_true", help="Download/update cached data for holding tickers")
    args = ap.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = fetch_holdings()
    except Exception as e:
        print(f"ERROR: {e}")
        return 2

    out_csv = REPORTS_DIR / "holdings_tickers.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved holdings watchlist → {out_csv} ({len(df)} rows)")

    if args.download and not df.empty:
        tickers = sorted(df["ticker"].unique().tolist())
        print(f"Downloading/Updating data for {len(tickers)} holding tickers…")
        update_all(tickers, force=False)
        print("Done updating holdings data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

