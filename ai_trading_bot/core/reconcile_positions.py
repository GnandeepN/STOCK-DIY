from __future__ import annotations

"""
Reconcile positions: ensure local DB has OPEN trades for live broker positions.

Creates missing OPEN trades using current position quantity and average price.
Does not auto-close or cancel existing trades — use reconcile_trades.py for that.
"""

import argparse
from typing import Dict, Tuple

from ai_trading_bot.core.orders_db import find_open_trade, create_trade
from ai_trading_bot.core.instruments import to_yahoo_ticker

try:
    from ai_trading_bot.brokers import ZerodhaBroker
except Exception:
    ZerodhaBroker = None


def fetch_positions(broker) -> Dict[Tuple[str, str], dict]:
    pos = broker.get_positions() or {}
    net = pos.get("net", [])
    out: Dict[Tuple[str, str], dict] = {}
    for p in net:
        exch = str(p.get("exchange") or "").upper()
        tsym = str(p.get("tradingsymbol") or "").upper()
        qty = int(p.get("quantity") or 0)
        if not exch or not tsym or qty == 0:
            continue
        out[(exch, tsym)] = p
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Create local OPEN trades for current broker positions if missing")
    ap.add_argument("--product", default="CNC", help="Default product for created trades when not available in positions")
    args = ap.parse_args()

    if ZerodhaBroker is None:
        print("kiteconnect not installed.")
        return 2
    b = ZerodhaBroker()
    try:
        b.connect()
    except Exception as e:
        print(f"Broker connect error: {e}")
        return 3

    pos_map = fetch_positions(b)
    created = 0
    for (exch, tsym), p in pos_map.items():
        # If there is already an OPEN trade, skip
        existing = find_open_trade(exch, tsym)
        if existing is not None:
            continue
        qty = abs(int(p.get("quantity") or 0))
        side = "BUY" if int(p.get("quantity") or 0) > 0 else "SELL"
        avg = float(p.get("average_price") or 0.0)
        prod = str(p.get("product") or args.product)
        ticker = to_yahoo_ticker(exch, tsym)
        create_trade(dict(
            ticker=ticker,
            exchange=exch,
            tradingsymbol=tsym,
            side=side,
            qty=qty,
            entry_price=avg,
            stop_loss=None,
            take_profit=None,
            product=prod,
            status="OPEN",
            broker_entry_order_id=None,
            note="reconcile:from_position",
            is_holding=1,
            atr=None, trail_sl=None, breakeven_done=0, tp1_done=0, stagnation_bars=0, realized_qty=0, realized_pnl=0.0, remaining_qty=qty,
        ))
        print(f"Created OPEN trade for {exch}:{tsym} {side} x{qty} @ {avg}")
        created += 1

    print(f"Reconcile positions complete. Trades created: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
