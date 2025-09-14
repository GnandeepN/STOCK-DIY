from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from ai_trading_bot.core.config import BASE_DIR

DB_PATH = BASE_DIR / "reports" / "orders.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    # Orders audit log
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT,
            exchange TEXT,
            tradingsymbol TEXT,
            side TEXT,
            qty INTEGER,
            price REAL,
            order_type TEXT,
            product TEXT,
            status TEXT,
            broker_order_id TEXT,
            error TEXT,
            note TEXT
        )
        """
    )
    # Trades lifecycle
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_open TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT,
            exchange TEXT,
            tradingsymbol TEXT,
            side TEXT,          -- BUY (long) or SELL (short entry)
            qty INTEGER,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            product TEXT,
            status TEXT,        -- OPEN/CLOSED/CANCELLED/ERROR
            broker_entry_order_id TEXT,
            ts_close TIMESTAMP,
            exit_price REAL,
            exit_reason TEXT,   -- TP/SL/MANUAL/ERROR
            broker_exit_order_id TEXT,
            note TEXT
        )
        """
    )
    # Add new columns if missing
    for ddl in [
        "ALTER TABLE trades ADD COLUMN is_holding INTEGER DEFAULT 0",
        "ALTER TABLE trades ADD COLUMN atr REAL",
        "ALTER TABLE trades ADD COLUMN trail_sl REAL",
        "ALTER TABLE trades ADD COLUMN breakeven_done INTEGER DEFAULT 0",
        "ALTER TABLE trades ADD COLUMN tp1_done INTEGER DEFAULT 0",
        "ALTER TABLE trades ADD COLUMN stagnation_bars INTEGER DEFAULT 0",
        "ALTER TABLE trades ADD COLUMN realized_qty INTEGER DEFAULT 0",
        "ALTER TABLE trades ADD COLUMN realized_pnl REAL DEFAULT 0",
        "ALTER TABLE trades ADD COLUMN remaining_qty INTEGER",
    ]:
        try:
            con.execute(ddl)
        except Exception:
            pass
    return con


def log_order(row: Dict[str, Any]) -> None:
    con = _conn()
    cols = [
        "ticker", "exchange", "tradingsymbol", "side", "qty", "price",
        "order_type", "product", "status", "broker_order_id", "error", "note",
    ]
    vals = [row.get(c) for c in cols]
    con.execute(
        f"INSERT INTO orders ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",
        vals,
    )
    con.commit()
    con.close()


def create_trade(row: Dict[str, Any]) -> int:
    """Insert a new OPEN trade; returns trade id."""
    con = _conn()
    cols = [
        "ticker", "exchange", "tradingsymbol", "side", "qty", "entry_price",
        "stop_loss", "take_profit", "product", "status", "broker_entry_order_id", "note",
        "is_holding", "atr", "trail_sl", "breakeven_done", "tp1_done", "stagnation_bars", "realized_qty", "realized_pnl", "remaining_qty",
    ]
    vals = [row.get(c) for c in cols]
    cur = con.execute(
        f"INSERT INTO trades ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",
        vals,
    )
    trade_id = int(cur.lastrowid)
    con.commit()
    con.close()
    return trade_id


def close_trade(trade_id: int, exit_price: float, reason: str, broker_exit_order_id: Optional[str] = None) -> None:
    con = _conn()
    con.execute(
        """
        UPDATE trades
        SET status = 'CLOSED', ts_close = CURRENT_TIMESTAMP, exit_price = ?, exit_reason = ?, broker_exit_order_id = ?
        WHERE id = ?
        """,
        (exit_price, reason, broker_exit_order_id, trade_id),
    )
    con.commit()
    con.close()


def get_open_trades() -> List[Dict[str, Any]]:
    con = _conn()
    cur = con.execute("SELECT id, ticker, exchange, tradingsymbol, side, qty, entry_price, stop_loss, take_profit, product FROM trades WHERE status = 'OPEN'")
    rows = [
        {
            "id": r[0], "ticker": r[1], "exchange": r[2], "tradingsymbol": r[3], "side": r[4],
            "qty": r[5], "entry_price": r[6], "stop_loss": r[7], "take_profit": r[8], "product": r[9],
        }
        for r in cur.fetchall()
    ]
    con.close()
    return rows


def get_open_trades_full() -> List[Dict[str, Any]]:
    con = _conn()
    cur = con.execute(
        "SELECT id, ticker, exchange, tradingsymbol, side, qty, entry_price, stop_loss, take_profit, product, broker_entry_order_id, atr, trail_sl, breakeven_done, tp1_done, stagnation_bars, realized_qty, realized_pnl, COALESCE(remaining_qty, qty) FROM trades WHERE status='OPEN'"
    )
    rows = [
        {
            "id": r[0], "ticker": r[1], "exchange": r[2], "tradingsymbol": r[3], "side": r[4],
            "qty": r[5], "entry_price": r[6], "stop_loss": r[7], "take_profit": r[8], "product": r[9],
            "broker_entry_order_id": r[10], "atr": r[11], "trail_sl": r[12], "breakeven_done": r[13], "tp1_done": r[14],
            "stagnation_bars": r[15], "realized_qty": r[16], "realized_pnl": r[17], "remaining_qty": r[18],
        }
        for r in cur.fetchall()
    ]
    con.close()
    return rows


def update_trail_sl(trade_id: int, value: float):
    con = _conn(); con.execute("UPDATE trades SET trail_sl=? WHERE id=?", (value, trade_id)); con.commit(); con.close()


def mark_breakeven(trade_id: int):
    con = _conn(); con.execute("UPDATE trades SET breakeven_done=1 WHERE id=?", (trade_id,)); con.commit(); con.close()


def mark_tp1(trade_id: int):
    con = _conn(); con.execute("UPDATE trades SET tp1_done=1 WHERE id=?", (trade_id,)); con.commit(); con.close()


def inc_stagnation(trade_id: int):
    con = _conn(); con.execute("UPDATE trades SET stagnation_bars = COALESCE(stagnation_bars,0)+1 WHERE id=?", (trade_id,)); con.commit(); con.close()


def reset_stagnation(trade_id: int):
    con = _conn(); con.execute("UPDATE trades SET stagnation_bars = 0 WHERE id=?", (trade_id,)); con.commit(); con.close()


def apply_partial_exit(trade_id: int, qty: int, price: float, reason: str):
    con = _conn()
    cur = con.execute("SELECT side, qty, entry_price, COALESCE(remaining_qty, qty), realized_qty, realized_pnl FROM trades WHERE id=?", (trade_id,))
    row = cur.fetchone()
    if not row:
        con.close(); return
    side, qty0, entry, rem, realized_qty, realized_pnl = row
    side = str(side).upper()
    rem = int(rem or 0)
    qty = min(int(qty), max(0, rem))
    if qty <= 0:
        con.close(); return
    pnl_part = (price - entry) * qty if side == 'BUY' else (entry - price) * qty
    new_rem = rem - qty
    new_rq = int(realized_qty or 0) + qty
    new_rp = float(realized_pnl or 0.0) + pnl_part
    if new_rem <= 0:
        con.execute(
            "UPDATE trades SET status='CLOSED', ts_close=CURRENT_TIMESTAMP, exit_price=?, exit_reason=?, realized_qty=?, realized_pnl=?, remaining_qty=0 WHERE id=?",
            (price, reason, new_rq, new_rp, trade_id)
        )
    else:
        con.execute(
            "UPDATE trades SET realized_qty=?, realized_pnl=?, remaining_qty=? WHERE id=?",
            (new_rq, new_rp, new_rem, trade_id)
        )
    con.commit(); con.close()


def find_open_trade(exchange: str, tradingsymbol: str) -> Optional[Dict[str, Any]]:
    con = _conn()
    cur = con.execute(
        "SELECT id, side, qty FROM trades WHERE status = 'OPEN' AND exchange = ? AND tradingsymbol = ? ORDER BY id DESC LIMIT 1",
        (exchange, tradingsymbol),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {"id": row[0], "side": row[1], "qty": row[2]}


def cancel_trade(trade_id: int, note: str = None):
    con = _conn()
    con.execute("UPDATE trades SET status='CANCELLED', note = COALESCE(?, note) WHERE id = ?", (note, trade_id))
    con.commit()
    con.close()


def update_trade_note(trade_id: int, note: str):
    con = _conn()
    con.execute("UPDATE trades SET note = ? WHERE id = ?", (note, trade_id))
    con.commit()
    con.close()
