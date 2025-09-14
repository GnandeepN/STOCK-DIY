from __future__ import annotations

"""
monitor_drift.py — Alerts when feature drift (PSI) or live hit-rate deviates.

Logic:
 - Reads reports/drift_summary.csv (written by backtest_walkforward.py) and alerts
   if any feature PSI exceeds a threshold (default 0.25).
 - Computes live hit-rate from trades table (last N closed trades) and compares
   to backtest walk-forward average winrate; alerts if deviation > delta.
"""

from pathlib import Path
import sqlite3
import pandas as pd
from ai_trading_bot.core.notify import send_message
from ai_trading_bot.core.config import BASE_DIR

REPORTS = BASE_DIR / "reports"
DB = REPORTS / "orders.db"


def alert(msg: str):
    try:
        send_message(msg)
    except Exception:
        pass
    print(msg)


def check_drift(psi_threshold: float = 0.25) -> int:
    path = REPORTS / "drift_summary.csv"
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    bad = df[df["psi"] >= psi_threshold]
    if not bad.empty:
        tops = bad.head(10).to_dict(orient="records")
        lines = [f"{r['ticker']}:{r['feature']} psi={r['psi']}" for r in tops]
        alert("⚠️ Feature drift detected:\n" + "\n".join(lines))
        return 1
    return 0


def check_hitrate(delta: float = 0.15, last_n: int = 30) -> int:
    if not DB.exists():
        return 0
    con = sqlite3.connect(DB)
    cur = con.execute(
        "SELECT side, realized_pnl FROM trades WHERE status='CLOSED' ORDER BY ts_close DESC LIMIT ?",
        (last_n,)
    )
    rows = cur.fetchall(); con.close()
    if not rows:
        return 0
    wins = sum(1 for _, pnl in rows if (pnl or 0) > 0)
    live_hr = wins / max(1, len(rows))
    # Load backtest winrate avg
    summ = REPORTS / "backtest_walkforward_summary.csv"
    if not summ.exists():
        return 0
    df = pd.read_csv(summ)
    bt_hr = (df["winrate"].dropna().mean() or 0) / 100.0
    if abs(live_hr - bt_hr) >= delta:
        alert(f"⚠️ Live hit-rate deviates: live={live_hr:.2f}, backtest={bt_hr:.2f}")
        return 1
    return 0


def main() -> int:
    x = check_drift() + check_hitrate()
    if x == 0:
        print("No drift/hit-rate alerts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

