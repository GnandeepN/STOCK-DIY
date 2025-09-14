from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
import sqlite3
import pandas as pd

from ai_trading_bot.core.instruments import to_yahoo_ticker
from ai_trading_bot.core.config import REPORTS_DIR

try:
    from ai_trading_bot.brokers import ZerodhaBroker
except Exception:
    ZerodhaBroker = None


SNAP_DIR = REPORTS_DIR / "portfolio_snapshots"
DB_PATH = REPORTS_DIR / "portfolio.db"


def snapshot() -> pd.DataFrame:
    if ZerodhaBroker is None:
        raise RuntimeError("kiteconnect not installed. Run `pip install kiteconnect`.")
    b = ZerodhaBroker(); b.connect()
    holds = b.get_holdings() or []
    poss = b.get_positions() or {}
    pos_day = poss.get("day", []) + poss.get("net", [])
    rows = []
    for h in holds:
        rows.append(dict(
            kind="holding",
            exchange=h.get("exchange"), tradingsymbol=h.get("tradingsymbol"),
            ticker=to_yahoo_ticker(h.get("exchange"), h.get("tradingsymbol")),
            qty=h.get("quantity"), avg_price=h.get("average_price"), last_price=h.get("last_price"),
            pnl=h.get("pnl"),
        ))
    for p in pos_day:
        rows.append(dict(
            kind="position",
            exchange=p.get("exchange"), tradingsymbol=p.get("tradingsymbol"),
            ticker=to_yahoo_ticker(p.get("exchange"), p.get("tradingsymbol")),
            qty=p.get("quantity"), avg_price=p.get("average_price"), last_price=p.get("last_price"),
            pnl=p.get("pnl"),
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["value"] = (pd.to_numeric(df["qty"], errors="coerce").fillna(0) * pd.to_numeric(df["last_price"], errors="coerce").fillna(0))
    return df


def save_csv_sqlite(df: pd.DataFrame) -> Path:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    datestr = datetime.now().strftime("%Y-%m-%d")
    csv_path = SNAP_DIR / f"snapshot_{datestr}.csv"
    df.to_csv(csv_path, index=False)

    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date TEXT,
            kind TEXT,
            exchange TEXT,
            tradingsymbol TEXT,
            ticker TEXT,
            qty INTEGER,
            avg_price REAL,
            last_price REAL,
            value REAL,
            pnl REAL
        )
        """
    )
    df2 = df.copy(); df2["date"] = datestr
    cols = ["date", "kind", "exchange", "tradingsymbol", "ticker", "qty", "avg_price", "last_price", "value", "pnl"]
    for _, r in df2[cols].iterrows():
        con.execute(
            f"INSERT INTO snapshots ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",
            [r[c] for c in cols]
        )
    con.commit(); con.close()
    return csv_path


def main() -> int:
    df = snapshot()
    if df.empty:
        print("No holdings/positions found.")
        return 0
    path = save_csv_sqlite(df)
    print(f"Saved snapshot → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

