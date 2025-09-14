from __future__ import annotations

"""
Modern Streamlit dashboard for ai-trading-bot

Run:
  streamlit run ui.py

This UI is read-only over your existing pipeline and DB. It does not change
backend logic (signals, training, execution). It focuses on a clean UX:
- Sidebar navigation (Home, Signals, Portfolio, Orders, P&L)
- Metrics and compact views
- Optional sparkline charts for tickers (last 20 bars via yfinance)
- Plan preview with simple Approve/Reject simulation
"""

import os
import sqlite3
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports"
LOGS = BASE / "logs"


# --------------------
# Utilities & caching
# --------------------
@st.cache_data(show_spinner=False, ttl=60)
def _load_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=30)
def _load_signals():
    filt = _load_csv(REPORTS / "filtered_signals_ranked.csv")
    top_b = _load_csv(REPORTS / "top_buys.csv")
    top_s = _load_csv(REPORTS / "top_sells.csv")
    return filt, top_b, top_s


@st.cache_data(show_spinner=False, ttl=60)
def _latest_snapshot() -> tuple[pd.DataFrame, Path | None]:
    snap_dir = REPORTS / "portfolio_snapshots"
    if not snap_dir.exists():
        return pd.DataFrame(), None
    files = sorted(snap_dir.glob("snapshot_*.csv"))
    if not files:
        return pd.DataFrame(), None
    latest = files[-1]
    try:
        df = pd.read_csv(latest)
        return df, latest
    except Exception:
        return pd.DataFrame(), latest


@st.cache_data(show_spinner=False, ttl=30)
def _load_db(query: str) -> pd.DataFrame:
    db = REPORTS / "orders.db"
    if not db.exists():
        return pd.DataFrame()
    con = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(query, con)
    except Exception:
        df = pd.DataFrame()
    finally:
        con.close()
    return df


@st.cache_data(show_spinner=False, ttl=120)
def _mini_chart(ticker: str) -> pd.DataFrame:
    # Tiny price frame for last 1mo, 1d bars (kept light)
    try:
        import yfinance as yf
        df = yf.download(ticker, period="1mo", interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # Handle MultiIndex columns (yfinance may return (ticker, field))
        if isinstance(df.columns, pd.MultiIndex):
            # Extract the 'Close' field across tickers
            try:
                close_df = df.xs("Close", axis=1, level=-1)
                if isinstance(close_df, pd.DataFrame):
                    series = close_df[ticker] if ticker in close_df.columns else close_df.iloc[:, 0]
                else:
                    series = close_df
                out = pd.DataFrame({"Date": df.index, "Close": series.values})
                return out.tail(20)
            except Exception:
                # Fallback: flatten and try again
                df.columns = df.columns.get_level_values(-1)
        # Standard single-index path
        out = df[["Close"]].tail(20).reset_index()
        # Ensure the first column is named 'Date'
        if out.columns[0] != "Date":
            out.rename(columns={out.columns[0]: "Date"}, inplace=True)
        return out[["Date", "Close"]]
    except Exception:
        pass
    return pd.DataFrame()


def _style_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    show = df.copy()
    cols = [c for c in ["ticker", "signal", "close", "take_profit", "stop_loss", "rr", "proba_up", "cagr", "winrate"] if c in show.columns]
    show = show[cols]
    return show


def _metric_card(label: str, value: str, delta: str | None = None):
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(label, value, delta)


# --------------------
# Layout & navigation
# --------------------
st.set_page_config(page_title="AI Trading Bot", page_icon="📈", layout="wide")

# Light CSS polish
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem;}
    .metric-label {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("📊 AI Trading Bot")
    section = st.radio("Navigate", ["Home", "Signals", "Portfolio", "Orders", "P&L"], index=0)
    st.divider()
    if st.button("🔄 Refresh", use_container_width=True):
        _load_signals.clear()
        _latest_snapshot.clear()
        _load_db.clear()
        _mini_chart.clear()
        st.toast("Refreshed")
    st.caption("v1 dashboard — logic unchanged")

# Header: date & capital
today = date.today().isoformat()
capital = os.getenv("CAPITAL_TOTAL", "100000")
h1, h2, h3 = st.columns([2, 1, 1])
with h1:
    st.subheader("Trading Dashboard")
with h2:
    st.metric("Date", today)
with h3:
    st.metric("Capital", f"₹{float(capital):,.0f}")

st.divider()


# --------------------
# Pages
# --------------------
if section == "Home":
    st.info("Welcome. Use the sidebar to browse Signals, Portfolio, Orders, and P&L.")
    st.write("Recent files:")
    col1, col2 = st.columns(2)
    with col1:
        filt, topb, tops = _load_signals()
        st.write("Filtered signals", filt.head(10))
    with col2:
        st.write("Plan CSV", _load_csv(REPORTS / "orders_plan.csv").head(10))

elif section == "Signals":
    st.subheader("Signals")
    filt, top_b, top_s = _load_signals()
    if filt.empty:
        st.warning("No signals found. Run signals_today.py and filter_signals.py.")
    else:
        buys = filt[filt.get("signal").astype(str).str.upper() == "BUY"]
        sells = filt[filt.get("signal").astype(str).str.upper() == "SELL"]
        st.write(f"Actionables: {len(filt)} (BUY={len(buys)}, SELL={len(sells)})")
        tb1, tb2 = st.tabs(["Top BUYs", "Top SELLs"])
        with tb1:
            show = _style_signals(top_b if not top_b.empty else buys.head(6))
            st.dataframe(show, use_container_width=True)
            for _, r in show.head(6).iterrows():
                tkr = str(r.get("ticker"))
                data = _mini_chart(tkr)
                if not data.empty:
                    st.line_chart(data.set_index("Date"), height=120)
        with tb2:
            show = _style_signals(top_s if not top_s.empty else sells.head(6))
            st.dataframe(show, use_container_width=True)

elif section == "Portfolio":
    st.subheader("Portfolio")
    df, latest = _latest_snapshot()
    if df.empty:
        st.info("No portfolio snapshots yet. Run portfolio_snapshot.py after market hours.")
    else:
        if "value" in df.columns:
            total_val = float(df["value"].fillna(0).sum())
            st.metric("Equity (est)", f"₹{total_val:,.0f}")
        st.dataframe(df, use_container_width=True, height=360)
        img = REPORTS / "equity_curve.png"
        if img.exists():
            st.image(str(img), caption="Equity Curve")

elif section == "Orders":
    st.subheader("Orders & Trades")
    with st.expander("Open Trades", expanded=True):
        q = (
            "SELECT id, ts_open, ticker, exchange||':'||tradingsymbol AS symbol, side, product, qty, remaining_qty, "
            "entry_price, stop_loss, take_profit, trail_sl, tp1_done, breakeven_done, note "
            "FROM trades WHERE status='OPEN' ORDER BY ts_open DESC"
        )
        st.dataframe(_load_db(q), use_container_width=True)
    with st.expander("Closed Trades", expanded=False):
        q = (
            "SELECT id, ts_open, ts_close, ticker, exchange||':'||tradingsymbol AS symbol, side, product, realized_qty, "
            "entry_price, exit_price, realized_pnl, exit_reason "
            "FROM trades WHERE status='CLOSED' ORDER BY ts_close DESC LIMIT 200"
        )
        st.dataframe(_load_db(q), use_container_width=True)

elif section == "P&L":
    st.subheader("P&L Dashboard")
    # Realized P&L today
    q = (
        "SELECT date(ts_close) as d, SUM(realized_pnl) as realized FROM trades "
        "WHERE status='CLOSED' GROUP BY date(ts_close) ORDER BY d"
    )
    df = _load_db(q)
    if df.empty:
        st.info("No realized trades yet.")
    else:
        st.bar_chart(df.set_index("d"), height=220)
    # Sector exposure (approx) from latest snapshot
    snap, _ = _latest_snapshot()
    if not snap.empty and "ticker" in snap.columns:
        # Best-effort sector mapping from signals cache if available
        sig, _, _ = _load_signals()
        sector_map = {}
        if not sig.empty and "sector" in sig.columns:
            g = sig[["ticker", "sector"]].dropna().drop_duplicates()
            sector_map = dict(zip(g["ticker"], g["sector"]))
        exp = []
        for _, r in snap.iterrows():
            t = str(r.get("ticker")); v = float(r.get("value") or 0)
            s = sector_map.get(t, "Unknown")
            exp.append((s, v))
        if exp:
            exp_df = pd.DataFrame(exp, columns=["sector", "value"]).groupby("sector").sum().sort_values("value", ascending=False)
            st.bar_chart(exp_df, height=220)

st.caption("© Trading dashboard — Streamlit UI layer only. Backend logic unchanged.")
