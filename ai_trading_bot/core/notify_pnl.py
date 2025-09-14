from __future__ import annotations

from pathlib import Path
import sqlite3
from datetime import datetime
from ai_trading_bot.core.notify import send_message
from ai_trading_bot.core.config import BASE_DIR

DB_PATH = BASE_DIR / "reports" / "orders.db"


def realized_pnl_today() -> float:
    if not DB_PATH.exists():
        return 0.0
    con = sqlite3.connect(DB_PATH)
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


def main() -> int:
    pnl = realized_pnl_today()
    send_message(f"Intraday P&L snapshot: ₹{pnl:,.0f} (realized)")
    print(f"Sent intraday P&L: {pnl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

