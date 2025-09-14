from __future__ import annotations

"""
daily_summary.py — Sends a Telegram summary after market close (IST) with
opened trades, closed trades, invested amount, and realized P&L for the day.

Schedule this daily (e.g., 16:00 IST) via cron/launchd:

  python daily_summary.py
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import pytz

from ai_trading_bot.core.orders_db import DB_PATH
from ai_trading_bot.core.notify import send_message


def ist_today_datestr() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d")


def parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "").split(".")[0])
    except Exception:
        return None


def main() -> int:
    if not DB_PATH.exists():
        send_message("No trades database yet. No activity today.")
        return 0
    con = sqlite3.connect(DB_PATH)
    cur = con.execute("SELECT id, ts_open, ts_close, ticker, exchange, tradingsymbol, side, qty, entry_price, exit_price FROM trades")
    rows = cur.fetchall()
    con.close()

    today = ist_today_datestr()
    opened = []
    closed = []
    invested = 0.0
    realized = 0.0

    for r in rows:
        (_id, ts_open, ts_close, ticker, exch, tsym, side, qty, entry, exitp) = r
        qty = int(qty or 0)
        entry = float(entry or 0)
        exitp = None if exitp is None else float(exitp)

        d_open = parse_ts(ts_open)
        if d_open and d_open.strftime("%Y-%m-%d") == today:
            opened.append((ticker, side, qty, entry))
            invested += qty * entry

        if ts_close:
            d_close = parse_ts(ts_close)
            if d_close and d_close.strftime("%Y-%m-%d") == today and exitp is not None:
                closed.append((ticker, side, qty, entry, exitp))
                if str(side).upper() == "BUY":
                    realized += (exitp - entry) * qty
                else:
                    realized += (entry - exitp) * qty

    lines = []
    lines.append(f"📊 Daily Summary — {today} (IST)")
    if opened:
        lines.append("Opened:")
        for t, s, q, e in opened[:20]:
            lines.append(f" • {s} {t} x{q} @ {e}")
        if len(opened) > 20:
            lines.append(f" … and {len(opened)-20} more")
    else:
        lines.append("Opened: none")

    if closed:
        lines.append("Closed:")
        for t, s, q, e, x in closed[:20]:
            pnl = (x - e) * q if s == "BUY" else (e - x) * q
            sign = "+" if pnl >= 0 else "-"
            lines.append(f" • {t} x{q} {s}->@{x} PnL {sign}{abs(pnl):.2f}")
        if len(closed) > 20:
            lines.append(f" … and {len(closed)-20} more")
    else:
        lines.append("Closed: none")

    lines.append(f"Invested today (notional): ₹{invested:,.0f}")
    lines.append(f"Realized P&L today: ₹{realized:,.0f}")

    msg = "\n".join(lines)
    ok = send_message(msg)
    if not ok:
        print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

