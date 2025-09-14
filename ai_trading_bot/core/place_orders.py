from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime, time
import pytz
import pandas as pd

from ai_trading_bot.core.instruments import resolve_symbol
from ai_trading_bot.core.orders_db import log_order
from ai_trading_bot.core.config import BASE_DIR

try:
    from ai_trading_bot.brokers import ZerodhaBroker, OrderRequest
except Exception:
    ZerodhaBroker = None
    OrderRequest = None  # type: ignore

# Load .env at startup for ORDER_* and KITE_* if present
def _load_env_file():
    try:
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip()
                if k and v and os.getenv(k) is None:
                    os.environ[k] = v
    except Exception:
        pass

_load_env_file()


def load_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Signals file not found: {path}")
    df = pd.read_csv(path)
    # Keep BUY/SELL only and drop obvious NAs
    if "signal" in df.columns:
        df = df[df["signal"].isin(["BUY", "SELL"])].copy()
    if "ticker" not in df.columns:
        raise ValueError("signals file missing 'ticker' column")
    return df


def compute_qty(price: float, default_value: float) -> int:
    if price <= 0:
        return 0
    qty = int(default_value // price)
    return max(qty, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Route signals to Zerodha (dry-run by default)")
    parser.add_argument("--signals", default=str(BASE_DIR / "reports" / "filtered_signals_ranked.csv"))
    parser.add_argument("--live", action="store_true", help="Send live orders (default dry-run)")
    parser.add_argument("--product", default=os.getenv("ORDER_PRODUCT", "CNC"), help="[DEPRECATED] global product (use --buy-product/--sell-product)")
    parser.add_argument("--buy-product", default=os.getenv("ORDER_BUY_PRODUCT", "CNC"), help="Product for BUY orders (CNC/MIS)")
    parser.add_argument("--sell-product", default=os.getenv("ORDER_SELL_PRODUCT", "MIS"), help="Product for SELL orders (CNC/MIS)")
    parser.add_argument("--value", type=float, default=float(os.getenv("ORDER_NOTIONAL", 10000)), help="Approx value per trade")
    parser.add_argument("--amo", action="store_true", help="Use AMO orders (after-market) if market is closed")
    args = parser.parse_args()

    df = load_signals(Path(args.signals))
    if df.empty:
        print("No actionable signals.")
        return 0

    # Connect broker only if live
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
            return 3

    # Market hours check (NSE): 09:15–15:30 IST
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist).time()
    mkt_open, mkt_close = time(9, 15), time(15, 30)
    market_open = (now_ist >= mkt_open) and (now_ist <= mkt_close)

    placed = 0
    for _, r in df.iterrows():
        ticker = str(r["ticker"]).strip()
        side = str(r["signal"]).upper()
        price = float(r.get("close", 0)) or 0.0
        exch, tsym = resolve_symbol(ticker)
        qty = compute_qty(price, args.value)
        if qty <= 0:
            continue

        note = "dry-run"
        status = "DRYRUN"
        order_id = None
        error = None

        if args.live:
            # Choose product per side (SELL defaults to MIS to allow shorting intraday)
            product = args.buy_product if side == "BUY" else args.sell_product
            variety = "regular"
            if args.amo or not market_open:
                variety = "amo"
            req = OrderRequest(symbol=f"{exch}:{tsym}", side=side, quantity=qty, price=None,
                               product=product, order_type="MARKET", variety=variety, tag="ai_trading_bot")
            res = broker.place_order(req)
            order_id = res.order_id
            error = res.error
            status = "PLACED" if res.ok else "ERROR"
            note = f"live/{variety}/{product}"

        log_order(dict(
            ticker=ticker, exchange=exch, tradingsymbol=tsym, side=side, qty=qty,
            price=price, order_type="MARKET", product=args.product, status=status,
            broker_order_id=order_id, error=error, note=note,
        ))

        if error:
            print(f"{status}: {ticker} {side} x{qty} @~{price} [{exch}:{tsym}] error={error}")
        else:
            print(f"{status}: {ticker} {side} x{qty} @~{price} [{exch}:{tsym}] {order_id or ''}")
        placed += 1

    print(f"Done. Orders processed: {placed}. See reports/orders.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
