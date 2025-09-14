from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple

DB_PATH = Path(__file__).resolve().parent / "reports" / "orders.db"


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _conn():
    return sqlite3.connect(DB_PATH)


def error_count_today() -> int:
    if not DB_PATH.exists():
        return 0
    con = _conn()
    cur = con.execute(
        "SELECT COUNT(1) FROM orders WHERE date(ts)=date('now','localtime') AND status='ERROR'"
    )
    n = int(cur.fetchone()[0])
    con.close()
    return n


def realized_pnl_today() -> float:
    if not DB_PATH.exists():
        return 0.0
    con = _conn()
    # Need trades table with exit prices
    cur = con.execute(
        "SELECT side, qty, entry_price, exit_price FROM trades WHERE status='CLOSED' AND date(ts_close)=date('now','localtime') AND exit_price IS NOT NULL"
    )
    total = 0.0
    for side, qty, entry, exitp in cur.fetchall():
        qty = int(qty or 0)
        entry = float(entry or 0)
        exitp = float(exitp or 0)
        pnl = (exitp - entry) * qty if str(side).upper() == 'BUY' else (entry - exitp) * qty
        total += pnl
    con.close()
    return total


def recent_error_count(minutes: int = 10) -> int:
    if not DB_PATH.exists():
        return 0
    con = _conn()
    cur = con.execute(
        "SELECT COUNT(1) FROM orders WHERE ts >= datetime('now','localtime', ?) AND status='ERROR'",
        (f"-{int(minutes)} minutes",)
    )
    n = int(cur.fetchone()[0]); con.close(); return n


def allow_trading(max_errors: int = 5, max_dd: float = -10000.0, max_recent_errors: int = 3, recent_window_min: int = 10) -> Tuple[bool, str]:
    """Circuit breaker: return (ok, reason). max_dd is negative rupees threshold for today's realized PnL.
    Example: max_errors=5, max_dd=-5000 → stop if >=5 errors today or PnL <= -5000.
    """
    errs = error_count_today()
    if errs >= max_errors:
        return False, f"Too many errors today: {errs} >= {max_errors}"
    pnl = realized_pnl_today()
    if pnl <= max_dd:
        return False, f"Intraday drawdown hit: PnL {pnl:.0f} <= {max_dd:.0f}"
    rerr = recent_error_count(minutes=recent_window_min)
    if rerr >= max_recent_errors:
        return False, f"Recent broker errors: {rerr} in last {recent_window_min}m"
    return True, "OK"
