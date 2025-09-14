from __future__ import annotations

import argparse
import os
import pandas as pd
from pathlib import Path
from typing import Dict

from ai_trading_bot.core.instruments import to_yahoo_ticker
from ai_trading_bot.core.notify import request_approval, send_message
from ai_trading_bot.core.portfolio_policy import decide_action
from ai_trading_bot.core.config import (
    REPORTS_DIR, MAX_PORTFOLIO_NOTIONAL, MAX_SINGLE_NAME_PCT, MAX_SECTOR_PCT, MAX_INDUSTRY_PCT,
    BASE_DIR
)
import yfinance as yf

try:
    from ai_trading_bot.brokers import ZerodhaBroker
except Exception:
    ZerodhaBroker = None


def load_holdings_fallback() -> pd.DataFrame:
    fp = REPORTS_DIR / "holdings_tickers.csv"
    if fp.exists():
        return pd.read_csv(fp)
    return pd.DataFrame(columns=["ticker", "qty", "average_price"])


def fetch_holdings_df() -> pd.DataFrame:
    if ZerodhaBroker is None:
        return load_holdings_fallback()
    b = ZerodhaBroker()
    try:
        b.connect()
        rows = b.get_holdings() or []
        df = pd.DataFrame(rows)
        if df.empty:
            return load_holdings_fallback()
        df["ticker"] = df.apply(lambda r: to_yahoo_ticker(r.get("exchange"), r.get("tradingsymbol")), axis=1)
        return df.rename(columns={"quantity": "qty", "average_price": "average_price"})[
            ["ticker", "qty", "average_price"]
        ]
    except Exception:
        return load_holdings_fallback()


def summarize_plan(df: pd.DataFrame) -> str:
    c = df["intent"].value_counts().to_dict()
    parts = [f"{k}={v}" for k, v in c.items()]
    return "Plan: " + ", ".join(parts)


def _sector_info(tickers: list[str]) -> dict:
    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            out[t] = dict(
                sector=info.get("sector", "Unknown"),
                industry=info.get("industry", "Unknown"),
            )
        except Exception:
            out[t] = dict(sector="Unknown", industry="Unknown")
    return out


def enforce_caps(plan: pd.DataFrame, holds: pd.DataFrame) -> pd.DataFrame:
    # Compute current portfolio exposure by ticker/sector/industry
    holds = holds.copy()
    if holds.empty:
        holds["ticker"] = []
    # Estimate holding notional by qty * avg price (fallback 0)
    holds["notional"] = (holds.get("qty", 0) * holds.get("average_price", 0)).fillna(0)
    total_now = float(holds["notional"].sum())

    # Sector/industry map for tickers
    tickers = plan["ticker"].unique().tolist()
    meta = _sector_info(tickers)

    # Precompute current sector/industry totals
    sec_now = {}
    ind_now = {}
    for _, r in holds.iterrows():
        t = str(r.get("ticker")); n = float(r.get("notional", 0))
        m = meta.get(t, {"sector": "Unknown", "industry": "Unknown"})
        sec_now[m["sector"]] = sec_now.get(m["sector"], 0.0) + n
        ind_now[m["industry"]] = ind_now.get(m["industry"], 0.0) + n

    total_limit = float(MAX_PORTFOLIO_NOTIONAL)
    name_pct = float(MAX_SINGLE_NAME_PCT) / 100.0
    sec_pct = float(MAX_SECTOR_PCT) / 100.0
    ind_pct = float(MAX_INDUSTRY_PCT) / 100.0

    # Function to cap qty for each row
    capped_rows = []
    for r in plan.itertuples(index=False):
        t = str(r.ticker)
        intent = str(r.intent).upper()
        price = float(getattr(r, "price_hint") or 0)
        qty = int(r.qty or 0)
        if qty <= 0 or price <= 0 or intent == "IGNORE":
            capped_rows.append(r._asdict())
            continue

        add_notional = qty * price
        # Portfolio notional cap
        remaining_total = max(0.0, total_limit - total_now)
        if remaining_total <= 0:
            qty = 0
        else:
            qty = min(qty, int(remaining_total // price) if price > 0 else 0)

        # Single-name cap
        # current name exposure
        curr_name = float(holds.loc[holds["ticker"] == t, "notional"].sum())
        max_name = total_limit * name_pct
        cap_qty_name = int(max(0, (max_name - curr_name)) // price) if price > 0 else 0
        qty = min(qty, cap_qty_name)

        # Sector/industry caps
        m = meta.get(t, {"sector": "Unknown", "industry": "Unknown"})
        curr_sec = float(sec_now.get(m["sector"], 0.0))
        max_sec = total_limit * sec_pct
        cap_qty_sec = int(max(0, (max_sec - curr_sec)) // price) if price > 0 else 0
        qty = min(qty, cap_qty_sec)

        curr_ind = float(ind_now.get(m["industry"], 0.0))
        max_ind = total_limit * ind_pct
        cap_qty_ind = int(max(0, (max_ind - curr_ind)) // price) if price > 0 else 0
        qty = min(qty, cap_qty_ind)

        # Apply
        new_row = r._asdict()
        new_row["qty"] = max(0, qty)
        # Update running tallies
        add_n = qty * price
        total_now += add_n
        sec_now[m["sector"]] = sec_now.get(m["sector"], 0.0) + add_n
        ind_now[m["industry"]] = ind_now.get(m["industry"], 0.0) + add_n
        if qty == 0:
            new_row["reason"] = (str(new_row.get("reason")) + ";cap_veto").strip(";")
        capped_rows.append(new_row)

    return pd.DataFrame(capped_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a day plan from signals + holdings")
    ap.add_argument("--signals", default=str(REPORTS_DIR / "filtered_signals_ranked.csv"))
    ap.add_argument("--out", default=str(REPORTS_DIR / "orders_plan.csv"))
    ap.add_argument("--approve", action="store_true", help="Ask plan approval via Telegram")
    args = ap.parse_args()

    sig = pd.read_csv(Path(args.signals))
    holds = fetch_holdings_df()
    hmap: Dict[str, Dict] = {str(r.ticker): {"quantity": int(r.qty or 0), "average_price": float(r.average_price or 0)} for r in holds.itertuples(index=False)}

    plan_rows = []
    for r in sig.itertuples(index=False):
        tkr = str(r.ticker)
        holding = hmap.get(tkr)
        policy = decide_action(r._asdict(), holding)
        intent = policy["action"]
        prod = policy["product"]
        reason = policy.get("reason")
        if intent == "IGNORE":
            qty = 0
        elif intent == "EXIT":
            qty = int(holding.get("quantity") if holding else 0)
        elif intent == "SHORT":
            qty = max(1, int((10000 // float(getattr(r, "close", 0) or 1))))
        else:  # BUY
            qty = max(1, int((10000 // float(getattr(r, "close", 0) or 1))))
        plan_rows.append(dict(
            ticker=tkr, intent=intent, product=prod, qty=qty, price_hint=getattr(r, "close", None), reason=reason
        ))

    plan = pd.DataFrame(plan_rows)
    # Enforce portfolio caps (resize/skip before approval)
    plan = enforce_caps(plan, holds)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out, index=False)

    msg = summarize_plan(plan)
    print(msg)
    print(f"Saved → {out}")
    if args.approve:
        ok = request_approval("Day Plan", msg + f"\nCSV: {out.name}", timeout=180)
        if not ok:
            try:
                send_message("Plan rejected ❌")
            except Exception:
                pass
            return 2
        try:
            send_message("Plan approved ✅")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
