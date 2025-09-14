from __future__ import annotations

"""
monitor_trades.py — monitors open trades and exits at TP/SL using market orders.

This avoids broker-specific OCO/GTT by polling quotes and placing exits when
thresholds are breached. Run during market hours (or with --amo if you want to
allow AMO exits after close for CNC).

Usage:
  python monitor_trades.py --interval 15 --live

Notes:
  - For BUY (long): exit SELL when LTP >= TP or LTP <= SL
  - For SELL (short): exit BUY when LTP <= TP or LTP >= SL
  - Exits use MARKET orders with product matching the entry product if known.
  - Keeps it simple: one-and-done exit; cancels the remaining virtual leg.
"""

import argparse
import os
import time
from datetime import datetime, time as dtime
from pathlib import Path
import pytz

from ai_trading_bot.core.orders_db import (
    get_open_trades_full, log_order, update_trail_sl, mark_breakeven, mark_tp1,
    inc_stagnation, reset_stagnation, apply_partial_exit
)
from ai_trading_bot.core.notify import request_approval, send_message
from ai_trading_bot.core.circuit_breakers import allow_trading
from ai_trading_bot.core.config import (
    TRAIL_ATR_MULT, STAGNATION_R_MULT, STAGNATION_MAX_BARS,
    HOLDINGS_TRAIL_ATR_MULT, HOLDINGS_TP1_R, HOLDINGS_TIME_MAX_BARS,
    BASE_DIR,
)

try:
    from ai_trading_bot.brokers import ZerodhaBroker, OrderRequest
except Exception:
    ZerodhaBroker = None
    OrderRequest = None  # type: ignore


def _load_env_file():
    try:
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k and v and os.getenv(k) is None:
                    os.environ[k.strip()] = v.strip()
    except Exception:
        pass


_load_env_file()


def is_market_open() -> bool:
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    t = now.time()
    return (t >= dtime(9, 15)) and (t <= dtime(15, 30)) and (now.weekday() < 5)


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitor trades and exit at TP/SL")
    ap.add_argument("--interval", type=int, default=15, help="Polling interval in seconds")
    ap.add_argument("--live", action="store_true", help="Place live exit orders (default dry-run)")
    ap.add_argument("--buy-product", default=os.getenv("ORDER_BUY_PRODUCT", "CNC"))
    ap.add_argument("--sell-product", default=os.getenv("ORDER_SELL_PRODUCT", "MIS"))
    ap.add_argument("--amo", action="store_true", help="Allow AMO exits when market closed (CNC)")
    ap.add_argument("--approve-exits", action="store_true", help="Require Telegram/CLI approval before placing exit orders")
    ap.add_argument("--ignore-cb", action="store_true", help="Ignore circuit breakers (not recommended in live)")
    ap.add_argument("--watch-holdings", action="store_true", help="Create OPEN trades for CNC holdings before monitoring")
    args = ap.parse_args()

    broker = None
    if args.live:
        if ZerodhaBroker is None:
            print("kiteconnect not installed. Run `pip install kiteconnect` or use dry-run.")
            return 2
        broker = ZerodhaBroker()
        try:
            broker.connect()
        except Exception as e:
            print(f"Broker connect error: {e}")
            # Send login URL via Telegram if session is missing/expired
            try:
                from kiteconnect import KiteConnect
                from notify import send_message
                import os as _os
                if "login URL" in str(e) or "access_token" in str(e):
                    _api = _os.getenv("KITE_API_KEY", "")
                    if _api:
                        _url = KiteConnect(api_key=_api).login_url()
                        send_message("Zerodha token missing/expired. Please log in and reply with request_token:\n" + _url)
            except Exception:
                pass
            return 3
    # Circuit breaker check
    if not args.ignore_cb:
        ok, why = allow_trading(max_errors=int(os.getenv("CB_MAX_ERRORS", 5)), max_dd=float(os.getenv("CB_MAX_DD", -10000)))
        if not ok:
            print(f"Circuit breaker active: {why}")
            try:
                send_message(f"Circuit breaker active: {why}. Halting exits today.")
            except Exception:
                pass
            return 10

    # Optional: reconcile holdings into trades so guardian logic applies
    if args.watch_holdings and broker is not None:
        try:
            from reconcile_positions import main as _rp_main
            _ = _rp_main.__doc__
            # Call reconcile_positions via import to reuse connection
            from reconcile_positions import fetch_positions
            pos_map = fetch_positions(broker)
            from orders_db import find_open_trade, create_trade
            from instruments import to_yahoo_ticker
            for (exch, tsym), p in pos_map.items():
                if find_open_trade(exch, tsym) is not None:
                    continue
                qty = abs(int(p.get("quantity") or 0))
                if qty <= 0:
                    continue
                side = "BUY" if int(p.get("quantity") or 0) > 0 else "SELL"
                avg = float(p.get("average_price") or 0.0)
                ticker = to_yahoo_ticker(exch, tsym)
                create_trade(dict(ticker=ticker, exchange=exch, tradingsymbol=tsym, side=side, qty=qty,
                                  entry_price=avg, stop_loss=None, take_profit=None, product=p.get("product","CNC"),
                                  status="OPEN", broker_entry_order_id=None, note="watch_holding",
                                  is_holding=1, atr=None, trail_sl=None, breakeven_done=0, tp1_done=0, stagnation_bars=0,
                                  realized_qty=0, realized_pnl=0.0, remaining_qty=qty))
        except Exception:
            pass

    print("Monitoring open trades. Ctrl+C to stop.")
    while True:
        try:
            if not is_market_open() and not args.amo and args.live:
                print("Market closed; sleeping. Use --amo for after-market exits.")
                time.sleep(max(60, args.interval))
                continue

            open_trades = get_open_trades_full()
            if not open_trades:
                time.sleep(args.interval)
                continue

            for tr in open_trades:
                exch = tr["exchange"]; tsym = tr["tradingsymbol"]
                side = str(tr["side"]).upper()
                qty_total = int(tr.get("qty") or 0)
                rem_qty = int(tr.get("remaining_qty") or qty_total)
                if rem_qty <= 0:
                    continue
                sl = tr.get("stop_loss"); tp = tr.get("take_profit")
                atr = float(tr.get("atr") or 0.0)
                trail = tr.get("trail_sl")
                is_hold = bool(int(tr.get("is_holding") or 0) == 1)
                trail_mult = HOLDINGS_TRAIL_ATR_MULT if is_hold else TRAIL_ATR_MULT
                tp1_r = HOLDINGS_TP1_R if is_hold else 1.0
                stagnation_max = HOLDINGS_TIME_MAX_BARS if is_hold else STAGNATION_MAX_BARS

                # Get last price
                ltp = None
                if broker is not None:
                    try:
                        q = broker.quote(exch, tsym)
                        ltp = float(q.get("last_price") or 0)
                        if not ltp:
                            oi = q.get("ohlc", {})
                            ltp = float(oi.get("close", 0))
                    except Exception:
                        pass
                # If dry-run or quote failed, skip
                if not ltp:
                    continue

                exit_needed = False; reason = None
                product = args.buy_product if side == "BUY" else args.sell_product
                exit_side = "SELL" if side == "BUY" else "BUY"

                if side == "BUY":
                    if tp is not None and ltp >= float(tp):
                        exit_needed = True; reason = "TP"
                    if sl is not None and ltp <= float(sl):
                        exit_needed = True; reason = "SL"
                    # TP1 and trailing
                    if not tr.get("tp1_done"):
                        if sl is not None:
                            R = max(1e-6, float(tr["entry_price"]) - float(sl))
                        else:
                            R = max(1e-6, atr)
                        if ltp >= float(tr["entry_price"]) + tp1_r * R:
                            qty_half = max(1, rem_qty // 2)
                            status = "DRYRUN"; order_id=None; error=None
                            if broker is not None and args.live:
                                if args.approve_exits:
                                    summary = f"EXIT SELL {exch}:{tsym} Qty={qty_half} LTP≈{ltp} reason=TP1"
                                    ok = request_approval(title="Exit Approval", text=summary, timeout=180)
                                    if not ok:
                                        try:
                                            ans = input(f"Approve exit? [y/N] {summary} ").strip().lower()
                                            if not ans.startswith("y"):
                                                continue
                                        except Exception:
                                            continue
                                req = OrderRequest(symbol=f"{exch}:{tsym}", side="SELL", quantity=qty_half, price=None,
                                                   product=(tr.get("product") or product), order_type="MARKET", variety="regular", tag="ai_trading_bot")
                                res = broker.place_order(req); order_id=res.order_id; error=res.error; status = "PLACED" if res.ok else "ERROR"
                            log_order(dict(ticker=f"{exch}:{tsym}", exchange=exch, tradingsymbol=tsym, side="SELL", qty=qty_half,
                                           price=ltp, order_type="MARKET", product=product, status=status,
                                           broker_order_id=order_id, error=error, note="tp1"))
                            if not error:
                                apply_partial_exit(tr["id"], qty_half, ltp, reason="TP1")
                                mark_tp1(tr["id"]); update_trail_sl(tr["id"], float(tr["entry_price"])) ; mark_breakeven(tr["id"]) 
                            continue
                    if atr > 0:
                        new_trail = ltp - trail_mult * atr
                        if trail is None or new_trail > float(trail):
                            update_trail_sl(tr["id"], float(new_trail)); trail = new_trail
                    if trail is not None and ltp <= float(trail):
                        exit_needed = True; reason = "TRAIL"
                    if atr > 0:
                        R = max(1e-6, float(tr["entry_price"]) - float(sl or (float(tr["entry_price"]) - atr)))
                        if abs(ltp - float(tr["entry_price"])) <= STAGNATION_R_MULT * R:
                            inc_stagnation(tr["id"])
                            if int(tr.get("stagnation_bars") or 0) + 1 >= stagnation_max:
                                exit_needed = True; reason = "TIME"
                        else:
                            reset_stagnation(tr["id"]) 
                else:
                    if tp is not None and ltp <= float(tp):
                        exit_needed = True; reason = "TP"
                    if sl is not None and ltp >= float(sl):
                        exit_needed = True; reason = "SL"
                    if not tr.get("tp1_done"):
                        if sl is not None:
                            R = max(1e-6, float(sl) - float(tr["entry_price"]))
                        else:
                            R = max(1e-6, atr)
                        if ltp <= float(tr["entry_price"]) - tp1_r * R:
                            qty_half = max(1, rem_qty // 2)
                            status = "DRYRUN"; order_id=None; error=None
                            if broker is not None and args.live:
                                if args.approve_exits:
                                    summary = f"EXIT BUY {exch}:{tsym} Qty={qty_half} LTP≈{ltp} reason=TP1"
                                    ok = request_approval(title="Exit Approval", text=summary, timeout=180)
                                    if not ok:
                                        try:
                                            ans = input(f"Approve exit? [y/N] {summary} ").strip().lower()
                                            if not ans.startswith("y"):
                                                continue
                                        except Exception:
                                            continue
                                req = OrderRequest(symbol=f"{exch}:{tsym}", side="BUY", quantity=qty_half, price=None,
                                                   product=(tr.get("product") or product), order_type="MARKET", variety="regular", tag="ai_trading_bot")
                                res = broker.place_order(req); order_id=res.order_id; error=res.error; status = "PLACED" if res.ok else "ERROR"
                            log_order(dict(ticker=f"{exch}:{tsym}", exchange=exch, tradingsymbol=tsym, side="BUY", qty=qty_half,
                                           price=ltp, order_type="MARKET", product=product, status=status,
                                           broker_order_id=order_id, error=error, note="tp1"))
                            if not error:
                                apply_partial_exit(tr["id"], qty_half, ltp, reason="TP1")
                                mark_tp1(tr["id"]); update_trail_sl(tr["id"], float(tr["entry_price"])) ; mark_breakeven(tr["id"]) 
                            continue
                    if atr > 0:
                        new_trail = ltp + trail_mult * atr
                        if trail is None or new_trail < float(trail):
                            update_trail_sl(tr["id"], float(new_trail)); trail = new_trail
                    if trail is not None and ltp >= float(trail):
                        exit_needed = True; reason = "TRAIL"
                    if atr > 0:
                        R = max(1e-6, float(sl or (float(tr["entry_price"]) + atr)) - float(tr["entry_price"]))
                        if abs(ltp - float(tr["entry_price"])) <= STAGNATION_R_MULT * R:
                            inc_stagnation(tr["id"])
                            if int(tr.get("stagnation_bars") or 0) + 1 >= stagnation_max:
                                exit_needed = True; reason = "TIME"
                        else:
                            reset_stagnation(tr["id"]) 

                status = "DRYRUN"; order_id=None; error=None; note=f"exit-{(reason or 'EXIT').lower()}"

                if broker is not None and args.live:
                    if args.approve_exits:
                        summary = f"EXIT {exit_side} {exch}:{tsym} Qty={rem_qty} LTP≈{ltp} reason={reason}"
                        ok = request_approval(title="Exit Approval", text=summary, timeout=180)
                        if not ok:
                            try:
                                ans = input(f"Approve exit? [y/N] {summary} ").strip().lower()
                                if not ans.startswith("y"):
                                    try:
                                        send_message(f"Exit rejected ❌ {exch}:{tsym} qty={rem_qty} ({reason})")
                                    except Exception:
                                        pass
                                    continue
                            except Exception:
                                try:
                                    send_message(f"Exit rejected ❌ {exch}:{tsym} qty={rem_qty} ({reason})")
                                except Exception:
                                    pass
                                continue
                        else:
                            try:
                                send_message(f"Exit approved ✅ {exch}:{tsym} qty={rem_qty} ({reason}). Placing order…")
                            except Exception:
                                pass
                    try:
                        req = OrderRequest(
                            symbol=f"{exch}:{tsym}", side=exit_side, quantity=rem_qty, price=None,
                            product=(tr.get("product") or product), order_type="MARKET", variety="regular", tag="ai_trading_bot"
                        )
                        res = broker.place_order(req)
                        order_id = res.order_id
                        error = res.error
                        status = "PLACED" if res.ok else "ERROR"
                    except Exception as e:
                        error = str(e)
                        status = "ERROR"

                log_order(dict(
                    ticker=f"{exch}:{tsym}", exchange=exch, tradingsymbol=tsym, side=exit_side, qty=rem_qty,
                    price=ltp, order_type="MARKET", product=product, status=status,
                    broker_order_id=order_id, error=error, note=note,
                ))

                if error:
                    print(f"EXIT-ERROR: {exch}:{tsym} {exit_side} x{rem_qty} @~{ltp} err={error}")
                    try:
                        send_message(f"Exit error ⚠️ {exch}:{tsym} {exit_side} x{rem_qty} @≈{ltp} → {error}")
                    except Exception:
                        pass
                    continue

                # apply full remaining exit
                apply_partial_exit(tr["id"], rem_qty, ltp, reason=reason or "EXIT")
                print(f"EXIT: trade#{tr['id']} {exch}:{tsym} {exit_side} x{rem_qty} via {reason} @~{ltp}")
                try:
                    send_message(f"Trade closed ✅ {exch}:{tsym} via {reason} @≈{ltp}")
                except Exception:
                    pass

            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Stopping monitor.")
            return 0
        except Exception:
            # Avoid crashing the monitor loop on transient errors
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
