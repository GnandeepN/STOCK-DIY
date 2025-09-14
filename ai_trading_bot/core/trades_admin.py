from __future__ import annotations

import argparse
from typing import List
from orders_db import get_open_trades, cancel_trade


def list_open() -> int:
    rows = get_open_trades()
    if not rows:
        print("No OPEN trades.")
        return 0
    print("id  exchange  symbol        side  qty  entry   SL     TP    product")
    for r in rows:
        print(f"{r['id']:>2}  {r['exchange']:<8}  {r['tradingsymbol']:<12}  {r['side']:<4}  {r['qty']:<3}  "
              f"{(r['entry_price'] or 0):>7}  {str(r['stop_loss'] or ''):>6}  {str(r['take_profit'] or ''):>6}  {r.get('product','')}")
    return 0


def cancel_all() -> int:
    rows = get_open_trades()
    if not rows:
        print("No OPEN trades to cancel.")
        return 0
    for r in rows:
        cancel_trade(int(r['id']), note="admin-cancel")
    print(f"Cancelled {len(rows)} OPEN trades in DB (no broker action).")
    return 0


def cancel_one(trade_id: int) -> int:
    cancel_trade(int(trade_id), note="admin-cancel")
    print(f"Cancelled trade #{trade_id} in DB (no broker action).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Admin utility to manage local trades DB")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-open", help="List OPEN trades")
    sub.add_parser("cancel-all-open", help="Mark all OPEN trades as CANCELLED (DB only)")
    p_cancel = sub.add_parser("cancel", help="Cancel one trade by id (DB only)")
    p_cancel.add_argument("id", type=int)

    args = ap.parse_args()
    if args.cmd == "list-open":
        return list_open()
    if args.cmd == "cancel-all-open":
        return cancel_all()
    if args.cmd == "cancel":
        return cancel_one(args.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

