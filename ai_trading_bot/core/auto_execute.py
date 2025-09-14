from __future__ import annotations

"""
auto_execute.py — create trades from filtered signals and place entries.

Usage examples:
  # Dry-run (default): logs trades to DB but does not place live orders
  python auto_execute.py --signals reports/filtered_signals_ranked.csv

  # Live with CNC longs and MIS shorts, size ~₹10,000 per trade
  python auto_execute.py --signals reports/filtered_signals_ranked.csv --live \
         --buy-product CNC --sell-product MIS --value 10000

This script:
  - reads ranked signals with stop_loss/take_profit
  - resolves symbols to exchange/tradingsymbol
  - computes quantity by notional/price
  - places entry order (market)
  - creates an OPEN trade in DB with SL/TP
  - skips tickers where an OPEN trade already exists

Exits are handled by monitor_trades.py.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime, time
import pytz
import pandas as pd

from ai_trading_bot.core.instruments import resolve_symbol
from ai_trading_bot.core.config import TP_ATR_MULT, SL_ATR_MULT
from ai_trading_bot.core.notify import request_approval, send_message
from ai_trading_bot.core.capital import CapitalRules, size_value, size_capital, size_atr_risk, size_kelly, size_vol_target
from ai_trading_bot.core.orders_db import log_order, create_trade, find_open_trade
from ai_trading_bot.core.circuit_breakers import allow_trading
import sqlite3
from ai_trading_bot.core.config import MAX_DAILY_ENTRIES, COOLDOWN_MINUTES
import yfinance as yf
from ai_trading_bot.core.instruments_meta import preflight_short

try:
    from ai_trading_bot.brokers import ZerodhaBroker, OrderRequest
except Exception:
    ZerodhaBroker = None
    OrderRequest = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent


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


def load_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Signals file not found: {path}")
    df = pd.read_csv(path)
    if "signal" in df.columns:
        df = df[df["signal"].isin(["BUY", "SELL"])].copy()
    required = ["ticker", "close"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"signals file missing '{c}' column")
    # Ensure SL/TP columns exist
    for c in ["stop_loss", "take_profit"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def compute_qty(price: float, notional: float) -> int:
    if price and price > 0:
        return max(1, int(notional // float(price)))
    return 0


def within_market_hours(allow_amo: bool) -> tuple[bool, str]:
    ist = pytz.timezone("Asia/Kolkata")
    t = datetime.now(ist).time()
    mkt_open, mkt_close = time(9, 15), time(15, 30)
    is_open = (t >= mkt_open) and (t <= mkt_close)
    return (True, "regular") if is_open else ((allow_amo, "amo") if allow_amo else (False, "regular"))


def main() -> int:
    p = argparse.ArgumentParser(description="Create trades from signals and place entries")
    p.add_argument("--signals", default=str(BASE_DIR / "reports" / "filtered_signals_ranked.csv"))
    p.add_argument("--live", action="store_true", help="Send live orders (default dry-run)")
    p.add_argument("--buy-product", default=os.getenv("ORDER_BUY_PRODUCT", "CNC"), help="Product for BUY entries (CNC/MIS)")
    p.add_argument("--sell-product", default=os.getenv("ORDER_SELL_PRODUCT", "MIS"), help="Product for SELL entries (CNC/MIS)")
    p.add_argument("--value", type=float, default=float(os.getenv("ORDER_NOTIONAL", 10000)), help="Approx value per trade")
    p.add_argument("--amo", action="store_true", help="Use AMO entries if market is closed")
    p.add_argument("--no-approval", action="store_true", help="Do not request Telegram/CLI approval before placing live entries")
    p.add_argument("--paper", action="store_true", help="Use paper broker (no real orders) but create trades and logs like live")
    p.add_argument("--ignore-cb", action="store_true", help="Ignore circuit breakers (useful for dry-run)")
    p.add_argument("--only", help="Comma-separated list of tickers to include (e.g., 'UPL.NS,INFY.NS')", default=None)
    p.add_argument("--reconcile", action="store_true", help="Reconcile local OPEN trades with broker before placing new ones")
    p.add_argument("--plan", help="Execute a pre-approved plan CSV (orders_plan.csv)")
    p.add_argument("--buy-only", action="store_true", help="Ignore SELL/EXIT/SHORT; place only BUY entries")
    p.add_argument("--approve-plan", action="store_true", help="Ask a single Telegram approval for the whole plan before executing")
    p.add_argument("--sizing", choices=["value", "capital", "atr-risk", "kelly", "vol"], default=os.getenv("ORDER_SIZING", "value"), help="Position sizing mode")
    p.add_argument("--capital", type=float, default=float(os.getenv("CAPITAL_TOTAL", 100000)), help="Total trading capital for sizing")
    p.add_argument("--risk", type=float, default=float(os.getenv("RISK_PER_TRADE_PCT", 1.0)), help="Risk per trade in percent (atr-risk)")
    p.add_argument("--max-concurrent", type=int, default=int(os.getenv("MAX_CONCURRENT_TRADES", 10)), help="Max concurrent open trades")
    p.add_argument("--max-notional", type=float, default=float(os.getenv("MAX_NOTIONAL_PER_TRADE", 0) or 0), help="Hard cap per-trade notional (0=disabled)")
    p.add_argument("--max-pct-cap", type=float, default=float(os.getenv("MAX_PCT_CAP_PER_INSTRUMENT", 0) or 0), help="Max % of capital per instrument (0=disabled)")
    p.add_argument("--target-vol", type=float, default=float(os.getenv("TARGET_VOL_PCT", 20.0)), help="Target vol % for --sizing vol")
    p.add_argument("--max-kelly", type=float, default=float(os.getenv("MAX_KELLY_FRAC", 0.1)), help="Max Kelly fraction")
    args = p.parse_args()

    plan_df = None
    if args.plan:
        plan_df = pd.read_csv(Path(args.plan))
        if args.buy_only and not plan_df.empty:
            # Keep only BUY intents in plan mode
            if "intent" in plan_df.columns:
                plan_df = plan_df[plan_df["intent"].astype(str).str.upper() == "BUY"].copy()
    else:
        df = load_signals(Path(args.signals))
        if args.buy_only and not df.empty and "signal" in df.columns:
            df = df[df["signal"].astype(str).str.upper() == "BUY"].copy()
        if args.only:
            wanted = [t.strip() for t in args.only.split(",") if t.strip()]
            df = df[df["ticker"].isin(wanted)]
    # Emptiness check for selected mode
    if plan_df is not None:
        if plan_df.empty:
            print("Plan is empty. Nothing to execute.")
            return 0
    else:
        if df.empty:
            print("No actionable signals.")
            return 0

    broker = None
    if args.live or args.paper:
        if ZerodhaBroker is None:
            print("kiteconnect not installed. Run `pip install kiteconnect` or use --paper/dry-run.")
            if not args.paper:
                return 2
        if args.paper:
            try:
                from brokers import PaperBroker
                broker = PaperBroker()
            except Exception:
                broker = None
        else:
            broker = ZerodhaBroker()
        try:
            broker.connect()
        except Exception as e:
            print(f"Broker connect error: {e}")
            # Help the user by sending the login URL via Telegram when access_token is missing
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

    ok_hours, variety = within_market_hours(args.amo)
    if not ok_hours and (args.live and not args.paper):
        print("Market closed. Re-run with --amo to place AMO entries, or run dry-run.")
        return 4

    # Optional reconciliation of OPEN trades (e.g., when user cancelled in app)
    if args.reconcile and broker is not None and not args.paper:
        try:
            from reconcile_trades import reconcile as _reconcile
            _reconcile(dry=False, broker=broker)
        except Exception:
            print("Reconcile failed; continuing without reconciliation.")

    placed = 0
    # Enforce circuit breakers only for live/paper unless explicitly ignored
    if (args.live or args.paper) and (not args.ignore_cb):
        ok, why = allow_trading(max_errors=int(os.getenv("CB_MAX_ERRORS", 5)), max_dd=float(os.getenv("CB_MAX_DD", -10000)))
        if not ok:
            print(f"Circuit breaker active: {why}")
            try:
                from notify import send_message
                send_message(f"Circuit breaker active: {why}. Halting entries today.")
            except Exception:
                pass
            return 10
    # Capital rules
    rules = CapitalRules(
        total_capital=float(args.capital),
        risk_per_trade_pct=float(args.risk),
        max_concurrent=int(args.max_concurrent),
        max_notional_per_trade=(float(args.max_notional) if float(args.max_notional) > 0 else None),
        max_pct_cap_per_instrument=(float(args.max_pct_cap) if float(args.max_pct_cap) > 0 else None),
    )
    # Optional plan-level approval
    if plan_df is not None and args.approve_plan and (args.live or args.paper) and not args.no_approval:
        summary = f"Plan intents: " + ", ".join([f"{k}={v}" for k, v in plan_df["intent"].value_counts().to_dict().items()])
        if not request_approval("Execute Plan", summary, timeout=180):
            print("Plan rejected.")
            return 0

    # Iterator over either plan or signals
    rows_iter = plan_df.itertuples(index=False) if plan_df is not None else df.itertuples(index=False)
    for r in rows_iter:
        if plan_df is not None:
            # plan mode
            ticker = str(getattr(r, "ticker"))
            intent = str(getattr(r, "intent")).upper()
            product = str(getattr(r, "product") or "MIS").upper()
            qty = int(getattr(r, "qty") or 0)
            price = float(getattr(r, "price_hint") or 0)
            sl = None; tp = None; atr = None; side = "BUY" if intent == "BUY" else "SELL"
            reason_prod = getattr(r, "reason", None)
            if intent == "IGNORE" or qty <= 0:
                print(f"SKIP: {ticker} intent={intent}")
                continue
            # EXIT maps to SELL CNC with holding qty
            if intent == "EXIT":
                side = "SELL"; product = "CNC"
        else:
            # signals mode
            r = r._asdict()
            ticker = str(r["ticker"]).strip()
            side = str(r["signal"]).upper()
            price = float(r.get("close", 0) or 0)
            sl = r.get("stop_loss"); tp = r.get("take_profit"); atr = r.get("atr")
            sl = float(sl) if pd.notna(sl) else None
            tp = float(tp) if pd.notna(tp) else None
            atr = float(atr) if (atr is not None and pd.notna(atr)) else None
            # product recommendation (already implemented earlier)
            rec_prod = str(r.get("recommended_product") or "").upper() if "recommended_product" in r else ""
            reason_prod = r.get("recommended_reason") if "recommended_reason" in r else None
            if side == "BUY" and rec_prod in ("CNC", "MIS"):
                product = rec_prod
            else:
                product = args.buy_product if side == "BUY" else args.sell_product
            exch, tsym = resolve_symbol(ticker)
            # SELL semantics for signals mode: if SELL and no plan, treat as SHORT intent unless you have holding (handled by user policies when using plan_day.py)
            # sizing for signals mode
            if args.sizing == "value":
                qty = compute_qty(price, args.value)
            elif args.sizing == "capital":
                qty = size_capital(price, rules)
            elif args.sizing == "atr-risk":
                qty = size_atr_risk(price, sl, rules, side)
            elif args.sizing == "kelly":
                qty = size_kelly(price, sl, tp, r.get("proba_up"), side, rules, max_frac=float(args.max_kelly))
            else:
                qty = size_vol_target(price, atr, rules, target_vol_pct=float(args.target_vol))
            if qty <= 0:
                print(f"SKIP: {ticker} invalid qty for price {price}")
                continue
        # Common path continues below
        exch, tsym = resolve_symbol(ticker)

        # Sanity-fix SELL levels if inconsistent (older signals may have BUY-style SL/TP)
        if side == "SELL" and atr and atr > 0:
            if (tp is None or tp >= price) or (sl is None or sl <= price):
                tp = round(price - TP_ATR_MULT * atr, 2)
                sl = round(price + SL_ATR_MULT * atr, 2)

        # Skip if an OPEN trade already exists for this instrument
        existing = find_open_trade(exch, tsym)
        if existing is not None:
            print(f"SKIP: {ticker} has an open trade (id={existing['id']}).")
            continue

        # Enforce max concurrent positions and daily caps/cooldown (LIVE only)
        if args.live:
            # Lazy import to avoid circular
            from orders_db import get_open_trades
            if len(get_open_trades()) >= rules.max_concurrent:
                print(f"SKIP: max concurrent trades reached ({rules.max_concurrent}).")
                break
            # Daily cap on new entries
            try:
                con = sqlite3.connect(BASE_DIR / "reports" / "orders.db")
                cur = con.execute("SELECT COUNT(1) FROM trades WHERE date(ts_open)=date('now','localtime') AND status='OPEN'")
                cnt = int(cur.fetchone()[0]); con.close()
                if cnt >= int(os.getenv("MAX_DAILY_ENTRIES", MAX_DAILY_ENTRIES)):
                    print(f"SKIP: daily entry cap reached ({cnt}).")
                    break
            except Exception:
                pass
            # Cooldown after two consecutive losers
            try:
                con = sqlite3.connect(BASE_DIR / "reports" / "orders.db")
                cur = con.execute("SELECT ts_close, ( (exit_price - entry_price)*realized_qty ) as pnl FROM trades WHERE status='CLOSED' ORDER BY ts_close DESC LIMIT 2")
                rows = cur.fetchall(); con.close()
                if len(rows) == 2:
                    import datetime as _dt
                    two_loss = all((r[1] or 0) < 0 for r in rows)
                    if two_loss:
                        # Check last close time
                        def _parse(ts):
                            try:
                                return _dt.datetime.fromisoformat(str(ts).split('.')[0])
                            except Exception:
                                return None
                        last_ts = _parse(rows[0][0])
                        if last_ts is not None:
                            if (_dt.datetime.now() - last_ts).total_seconds() < int(os.getenv("COOLDOWN_MINUTES", COOLDOWN_MINUTES))*60:
                                print("SKIP: cooldown active after two consecutive losers.")
                                break
            except Exception:
                pass

        # Sizing
        if args.sizing == "value":
            qty = compute_qty(price, args.value)
        elif args.sizing == "capital":
            qty = size_capital(price, rules)
        elif args.sizing == "atr-risk":
            qty = size_atr_risk(price, sl, rules, side)
        elif args.sizing == "kelly":
            qty = size_kelly(price, sl, tp, r.get("proba_up"), side, rules, max_frac=float(args.max_kelly))
        else:  # vol targeting
            qty = size_vol_target(price, atr, rules, target_vol_pct=float(args.target_vol))
        if qty <= 0:
            print(f"SKIP: {ticker} invalid qty for price {price}")
            continue

        # Portfolio-level volatility scaling (average ATR/price over current df)
        try:
            if args.sizing in ("value", "capital", "atr-risk", "kelly"):
                atrs = df.get("atr")
                if atrs is not None and len(df) > 0 and price > 0:
                    avg_realized = float((df["atr"] / df["close"]).mean() * 100)
                    target = float(args.target_vol)
                    if avg_realized > 0:
                        scale = min(1.0, max(0.3, target / avg_realized))
                        qty = max(1, int(qty * scale))
        except Exception:
            pass

        # Index volatility scaling (NIFTY) — reduce size when index ATR/Close is high
        try:
            idx = yf.download(os.getenv("INDEX_SYMBOL", "^NSEI"), period="3mo", interval="1d", auto_adjust=False, progress=False, threads=False)
            if not idx.empty:
                import numpy as _np
                hi = idx["High"]; lo = idx["Low"]; cl = idx["Close"]; prev = cl.shift(1)
                tr = _np.maximum(hi - lo, _np.maximum((hi - prev).abs(), (lo - prev).abs()))
                atr_series = tr.ewm(alpha=1/14, adjust=False).mean()
                # Extract pure Python floats (no pandas/numpy scalar warnings)
                arr_atr = _np.ravel(atr_series.to_numpy(dtype=float))
                arr_cl = _np.ravel(cl.to_numpy(dtype=float))
                if arr_cl.size > 0 and arr_atr.size > 0:
                    atr_val = arr_atr[-1].item()
                    close_val = arr_cl[-1].item()
                else:
                    atr_val = 0.0
                    close_val = 0.0
                if close_val > 0.0:
                    vol_pct = (atr_val / close_val) * 100.0
                    target_vol = float(os.getenv("INDEX_TARGET_VOL_PCT", 20.0))
                    scale = min(1.0, max(0.3, target_vol / vol_pct)) if vol_pct > 0.0 else 1.0
                    qty = max(1, int(qty * scale))
        except Exception:
            pass

        # Recommended product from signals (BUY only); SELL stays MIS by default in India
        rec_prod = str(r.get("recommended_product") or "").upper() if "recommended_product" in r else ""
        reason_prod = r.get("recommended_reason") if "recommended_reason" in r else None
        if side == "BUY" and rec_prod in ("CNC", "MIS"):
            product = rec_prod
        else:
            product = args.buy_product if side == "BUY" else args.sell_product
        status = "DRYRUN"
        order_id = None
        error = None
        note_reason = (f";decide={reason_prod}" if reason_prod else "")
        if side == "SELL" and not note_reason:
            note_reason = ";decide=short_mis"
        note = f"dry-run/{variety}/{product}{note_reason}"

        if args.live or args.paper:
            # Approval step (Telegram). If unavailable, fall back to CLI prompt unless --no-approval.
            approved = True
            if not args.no_approval:
                # Enhanced Telegram message with product + reason
                header = f"{side} {ticker} x{qty} @≈{price} [{product}]"
                if side == "BUY":
                    reason_text = reason_prod or ""
                else:
                    reason_text = "Short entries forced intraday"
                reason_line = f"Reason: {reason_text}" if reason_text else ""
                summary = header if not reason_line else f"{header}\n{reason_line}"
                approved = request_approval(title="Order Approval", text=summary, timeout=180)
                if not approved:
                    # As a fallback when Telegram not configured or timeout, prompt on CLI
                    try:
                        ans = input(f"Approve? [y/N] {summary} ").strip().lower()
                        approved = ans.startswith("y")
                    except Exception:
                        approved = False

            if not approved:
                log_order(dict(
                    ticker=ticker, exchange=exch, tradingsymbol=tsym, side=side, qty=qty,
                    price=price, order_type="MARKET", product=product, status="REJECTED",
                    broker_order_id=None, error=None, note="rejected-by-user",
                ))
                print(f"REJECTED: {ticker} {side} x{qty} @~{price} [{exch}:{tsym}]")
                try:
                    send_message(f"Thanks. Rejected ❌ {side} {ticker} x{qty} @≈{price}. Not placing order.")
                except Exception:
                    pass
                continue

            # Approved — notify before placing
            try:
                send_message(f"Thanks. Approved ✅ {side} {ticker} x{qty} @≈{price} [{product}]. Placing order…")
            except Exception:
                pass

            # Preflight for shorts (metadata-based)
            if side == "SELL" and str(product).upper() == "MIS":
                ok_pf, why_pf = preflight_short(ticker, qty, price)
                if not ok_pf:
                    log_order(dict(
                        ticker=ticker, exchange=exch, tradingsymbol=tsym, side=side, qty=qty,
                        price=price, order_type="MARKET", product=product, status="REJECTED",
                        broker_order_id=None, error=None, note=f"preflight:{why_pf}{note_reason}",
                    ))
                    print(f"REJECTED (preflight): {ticker} {side} x{qty} {why_pf}")
                    try:
                        send_message(f"Rejected (preflight) ❌ {side} {ticker} x{qty} @≈{price} [{product}] → {why_pf}")
                    except Exception:
                        pass
                    continue
            req = OrderRequest(
                symbol=f"{exch}:{tsym}", side=side, quantity=qty, price=None,
                product=product, order_type="MARKET", variety=variety, tag="ai_trading_bot"
            )
            res = broker.place_order(req)
            order_id = res.order_id
            error = res.error
            status = "PLACED" if res.ok else "ERROR"
            note = f"live/{variety}/{product}{note_reason}"

        log_order(dict(
            ticker=ticker, exchange=exch, tradingsymbol=tsym, side=side, qty=qty,
            price=price, order_type="MARKET", product=product, status=status,
            broker_order_id=order_id, error=error, note=note,
        ))

        if error:
            print(f"{status}: {ticker} {side} x{qty} @~{price} [{exch}:{tsym}] error={error}")
            try:
                send_message(f"Order error ⚠️ {side} {ticker} x{qty} @≈{price} [{product}] → {error}")
            except Exception:
                pass
            continue

        if (args.live or args.paper) and status != "ERROR":
            trade_id = create_trade(dict(
                ticker=ticker, exchange=exch, tradingsymbol=tsym, side=side, qty=qty,
                entry_price=price, stop_loss=sl, take_profit=tp, product=product, status="OPEN",
                broker_entry_order_id=order_id, note=note,
                atr=atr, trail_sl=None, breakeven_done=0, tp1_done=0, stagnation_bars=0, realized_qty=0, realized_pnl=0.0, remaining_qty=qty,
            ))
            print(f"OPEN: trade#{trade_id} {ticker} {side} x{qty} @~{price} SL={sl} TP={tp}")
            placed += 1
            try:
                send_message(f"Order placed ✅ {side} {ticker} x{qty} @≈{price} [{product}] id={order_id}")
            except Exception:
                pass

    print(f"Done. Trades opened: {placed}. Monitor with monitor_trades.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
