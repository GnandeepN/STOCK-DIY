from __future__ import annotations

import argparse
from typing import Dict
from ai_trading_bot.core.orders_db import get_open_trades_full, cancel_trade, update_trade_note

try:
    from ai_trading_bot.brokers import ZerodhaBroker
except Exception:
    ZerodhaBroker = None


def fetch_orders_map(broker) -> Dict[str, dict]:
    try:
        orders = broker.kite.orders()  # type: ignore[attr-defined]
        return {str(o.get("order_id")): o for o in orders}
    except Exception:
        return {}


def reconcile(dry: bool = False, broker=None) -> int:
    if broker is None:
        if ZerodhaBroker is None:
            print("kiteconnect not installed.")
            return 2
        broker = ZerodhaBroker()
        try:
            broker.connect()
        except Exception as e:
            print(f"Broker connect error: {e}")
            return 3

    omap = fetch_orders_map(broker)
    open_trades = get_open_trades_full()
    fixed = 0
    for tr in open_trades:
        oid = str(tr.get("broker_entry_order_id") or "")
        if not oid:
            # No broker order id; cannot reconcile
            continue
        od = omap.get(oid)
        if od is None:
            # If order not found at broker, likely cancelled/rejected; mark cancelled
            msg = f"Order {oid} not found at broker; marking trade {tr['id']} CANCELLED"
            print(msg)
            if not dry:
                cancel_trade(int(tr["id"]), note="reconcile:order_missing")
                fixed += 1
            continue
        status = str(od.get("status") or "").upper()
        if status in ("CANCELLED", "REJECTED"):
            msg = f"Order {oid} status={status}; marking trade {tr['id']} CANCELLED"
            print(msg)
            if not dry:
                cancel_trade(int(tr["id"]), note=f"reconcile:{status.lower()}")
                fixed += 1
        else:
            # COMPLETE, OPEN, or QUEUED: leave as OPEN
            pass

    print(f"Reconcile complete. Trades marked CANCELLED: {fixed}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconcile OPEN trades in DB with broker order statuses")
    ap.add_argument("--dry", action="store_true", help="Only report, do not modify DB")
    args = ap.parse_args()
    return reconcile(dry=bool(args.dry))


if __name__ == "__main__":
    raise SystemExit(main())
