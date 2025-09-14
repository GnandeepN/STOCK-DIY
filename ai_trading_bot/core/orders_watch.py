from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from ai_trading_bot.core.notify import send_message
from ai_trading_bot.core.config import BASE_DIR

try:
    from ai_trading_bot.brokers import ZerodhaBroker
except Exception:
    ZerodhaBroker = None

STATE = BASE_DIR / "reports" / "orders_notified.json"


def load_state() -> set[str]:
    try:
        if STATE.exists():
            return set(json.loads(STATE.read_text()).get("notified", []))
    except Exception:
        pass
    return set()


def save_state(s: set[str]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"notified": sorted(list(s))}, indent=2))


def main() -> int:
    if ZerodhaBroker is None:
        print("kiteconnect not installed.")
        return 2
    b = ZerodhaBroker(); b.connect()
    notified = load_state()
    try:
        orders = b.kite.orders()  # type: ignore[attr-defined]
    except Exception as e:
        print(f"orders() failed: {e}")
        return 3
    new = False
    for o in orders:
        oid = str(o.get("order_id"))
        status = str(o.get("status"))
        if status == "COMPLETE" and oid and oid not in notified:
            sym = f"{o.get('exchange')}:{o.get('tradingsymbol')}"
            side = o.get("transaction_type")
            qty = o.get("quantity")
            price = o.get("average_price") or o.get("price")
            send_message(f"Order filled ✅ {side} {sym} x{qty} @ {price}")
            notified.add(oid)
            new = True
    if new:
        save_state(notified)
    print("Checked orders; notifications sent:" + (" yes" if new else " no"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

