# filter_signals.py (Phase 5: Robust + Logging + Drop Dead Strategies)

import pandas as pd
import logging
from ai_trading_bot.core.config import (
    REPORTS_DIR, LOGS_DIR, MIN_CONFIDENCE, MIN_AVG_VOLUME, SECTOR_CAPS, INDUSTRY_CAPS, RR_MIN,
    RR_INTRADAY_MAX, RR_CNC_MIN, CONFIDENCE_CNC, REGIME_NIFTY_RSI_MIN, REGIME_NIFTY_RSI_STRONG, LATEST_ENTRY_MIS_AFTER
)
from datetime import datetime
import pytz
import os
from ai_trading_bot.core.logger import get_logger

# --- Tunables for surfacing BUYs ---
TOP_BUYS = 6
TOP_SELLS = 6
MIN_BUYS_IN_MAIN = 6

# --- Setup logger ---
LOG_FILE = LOGS_DIR / "filter_signals.log"
logger = get_logger("filter_signals", LOG_FILE)

# --- Configurable thresholds ---
MIN_CAGR = -1.0   # keep only CAGR > MIN_CAGR
MAX_DD = 20.0     # allow up to 20% drawdown
TOP_N = None      # set to int for limiting

try:
    # --- Load reports ---
    signals = pd.read_csv(REPORTS_DIR / "signals_today.csv")
    backtest = pd.read_csv(REPORTS_DIR / "backtest_walkforward_summary.csv")
except FileNotFoundError as e:
    logger.error(f"Missing required file: {e}")
    print(f"❌ ERROR: {e}")
    exit(1)

# --- Merge ---
merged = pd.merge(signals, backtest, on="ticker", how="inner")

# If 'signal' column missing (all rows errored), short-circuit gracefully
if "signal" not in merged.columns:
    logger.warning("No 'signal' column found after merge; likely all rows errored upstream.")
    merged.to_csv(REPORTS_DIR / "all_signals_ranked.csv", index=False)
    # Save empty filtered result and exit
    pd.DataFrame(columns=["ticker"]).to_csv(REPORTS_DIR / "filtered_signals_ranked.csv", index=False)
    print("⚠️ No actionable trades: signals data has no 'signal' column (errors upstream).")
    logger.info("No actionable trades: missing 'signal' column; saved empty filtered file.")
    exit(0)

# --- Keep only BUY/SELL (BUY-only toggle supported) ---
BUY_ONLY = bool(int(os.getenv("BUY_ONLY", "1")))  # default ON per user preference
if BUY_ONLY:
    actionable = merged[merged["signal"].isin(["BUY"])].copy()
else:
    actionable = merged[merged["signal"].isin(["BUY", "SELL"])].copy()

# --- Apply filters ---
filtered = actionable[
    (actionable["cagr"] > MIN_CAGR) &
    (actionable["max_dd"].abs() <= MAX_DD)
].copy()

# Confidence filter: BUY proba_up>=MIN_CONFIDENCE; SELL proba_up<=1-MIN_CONFIDENCE
if "proba_up" in filtered.columns and MIN_CONFIDENCE:
    keep_buy = (filtered["signal"] == "BUY") & (filtered["proba_up"] >= MIN_CONFIDENCE)
    keep_sell = (filtered["signal"] == "SELL") & (filtered["proba_up"] <= (1 - MIN_CONFIDENCE))
    filtered = filtered[keep_buy | keep_sell]

# Volume filter (uses avg_volume_20 from signals_today extras)
if MIN_AVG_VOLUME and "avg_volume_20" in filtered.columns:
    filtered = filtered[filtered["avg_volume_20"] >= MIN_AVG_VOLUME]

# Entry quality filters for BUY
if "signal" in filtered.columns:
    buy_mask = filtered["signal"] == "BUY"
    if "rr" in filtered.columns:
        filtered = filtered[~buy_mask | (filtered["rr"] >= RR_MIN)]
    # Multi-timeframe trend and regime
    for col in ["mtf_trend_ok", "regime_ok"]:
        if col in filtered.columns:
            filtered = filtered[~buy_mask | (filtered[col] == True)]
    # Breakout + volume OR pullback
    if set(["breakout_ok", "vol_spike", "pullback_ok"]).issubset(filtered.columns):
        filt = (filtered["breakout_ok"].fillna(False) & filtered["vol_spike"].fillna(False)) | (filtered["pullback_ok"].fillna(False))
        filtered = filtered[~buy_mask | filt]

# Ensure required price columns exist even if empty
for col in ["close", "take_profit", "stop_loss"]:
    if col not in filtered.columns:
        filtered[col] = pd.Series(dtype=float)

# --- Drop dead strategies ---
filtered = filtered[~((filtered["cagr"] == 0.0) & (filtered["winrate"] == 0.0))]
if "n_trades" in filtered.columns:
    filtered = filtered[filtered["n_trades"] > 0]

# --- Add profit potential ---
if {"take_profit", "close"}.issubset(filtered.columns) and len(filtered) >= 0:
    # safe compute; handles empty frames
    filtered["profit_pct"] = (
        (filtered["take_profit"] - filtered["close"]) / filtered["close"] * 100
    ).round(2)

# --- Rank ---
filtered = filtered.sort_values(by=["cagr", "winrate"], ascending=[False, False])
if TOP_N:
    filtered = filtered.head(TOP_N)

# --- Sector caps (limit number per sector)
if SECTOR_CAPS and "sector" in filtered.columns:
    rows = []
    counts = {}
    for _, row in filtered.iterrows():
        sec = str(row.get("sector") or "").strip()
        cap = SECTOR_CAPS.get(sec)
        if cap is None:
            rows.append(row)
            continue
        c = counts.get(sec, 0)
        if c < cap:
            rows.append(row)
            counts[sec] = c + 1
    filtered = pd.DataFrame(rows)

# --- Industry caps (limit number per industry)
if INDUSTRY_CAPS and "industry" in filtered.columns:
    rows = []
    counts = {}
    for _, row in filtered.iterrows():
        ind = str(row.get("industry") or "").strip()
        cap = INDUSTRY_CAPS.get(ind)
        if cap is None:
            rows.append(row)
            continue
        c = counts.get(ind, 0)
        if c < cap:
            rows.append(row)
            counts[ind] = c + 1
    filtered = pd.DataFrame(rows)

relax_log = None

# Helper: append relaxed BUYs to reach MIN_BUYS_IN_MAIN while respecting caps
def _append_relaxed_buys(base: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    global relax_log
    if "signal" not in base.columns:
        return current
    buys_now = current[current["signal"] == "BUY"]["ticker"].unique().tolist()
    cands = base[(base["signal"] == "BUY")].copy()
    # Enforce trend and regime only
    for col in ["trend_ok", "mtf_trend_ok", "regime_ok"]:
        if col in cands.columns:
            cands = cands[cands[col] == True]
    # Relax RR and confidence
    if "rr" in cands.columns:
        cands = cands[cands["rr"] >= 1.2]
    if "proba_up" in cands.columns:
        cands = cands[cands["proba_up"] >= 0.52]
    # Exclude already present
    cands = cands[~cands["ticker"].isin(buys_now)].copy()
    # Rank by cagr, winrate, rr if present
    by = ["cagr", "winrate"] + (["rr"] if "rr" in cands.columns else [])
    cands = cands.sort_values(by=by, ascending=[False] * len(by))
    # Apply caps while appending
    rows = []
    # Start counts from current filtered
    sec_counts = {}
    ind_counts = {}
    if "sector" in current.columns and SECTOR_CAPS:
        for _, row in current.iterrows():
            sec = str(row.get("sector") or "").strip()
            if sec:
                sec_counts[sec] = sec_counts.get(sec, 0) + 1
    if "industry" in current.columns and INDUSTRY_CAPS:
        for _, row in current.iterrows():
            ind = str(row.get("industry") or "").strip()
            if ind:
                ind_counts[ind] = ind_counts.get(ind, 0) + 1
    need = max(0, MIN_BUYS_IN_MAIN - len(buys_now))
    for _, row in cands.iterrows():
        if need <= 0:
            break
        # Sector cap check
        if SECTOR_CAPS and "sector" in cands.columns:
            sec = str(row.get("sector") or "").strip()
            cap = SECTOR_CAPS.get(sec)
            if cap is not None and sec_counts.get(sec, 0) >= cap:
                continue
        # Industry cap check
        if INDUSTRY_CAPS and "industry" in cands.columns:
            ind = str(row.get("industry") or "").strip()
            capi = INDUSTRY_CAPS.get(ind)
            if capi is not None and ind_counts.get(ind, 0) >= capi:
                continue
        rows.append(row)
        if "sector" in cands.columns:
            _sec_val = str(row.get("sector") or "").strip()
            if _sec_val:
                sec_counts[_sec_val] = sec_counts.get(_sec_val, 0) + 1
        if "industry" in cands.columns:
            _ind_val = str(row.get("industry") or "").strip()
            if _ind_val:
                ind_counts[_ind_val] = ind_counts.get(_ind_val, 0) + 1
        need -= 1
    if rows:
        relax_log = f"⚠️ Relaxed filters to surface at least {TOP_BUYS} BUY signals."
        return pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
    return current

# --- Recommend product (CNC vs MIS) for BUYs (idempotent if already set) ---
def _decide_product_row(row) -> tuple:
    # Defaults
    prod = row.get("recommended_product")
    reason = row.get("recommended_reason")
    if pd.notna(prod):
        return prod, reason
    if row.get("signal") != "BUY":
        return None, None
    rr = row.get("rr"); proba = row.get("proba_up"); n_rsi = row.get("nifty_rsi"); trend_ok = bool(row.get("trend_ok"))
    breakout = bool(row.get("breakout_ok")); vol_spike = bool(row.get("vol_spike")); pullback = bool(row.get("pullback_ok"))
    ist = pytz.timezone("Asia/Kolkata"); now_t = datetime.now(ist).strftime("%H:%M"); is_late = now_t >= LATEST_ENTRY_MIS_AFTER
    product = "CNC"; reasons = []
    try:
        if rr is not None and float(rr) < RR_INTRADAY_MAX:
            product = "MIS"; reasons.append(f"RR {rr}< {RR_INTRADAY_MAX}")
        elif rr is not None and float(rr) >= RR_CNC_MIN:
            reasons.append(f"RR {rr}≥ {RR_CNC_MIN}")
        if proba is not None and float(proba) >= CONFIDENCE_CNC:
            reasons.append(f"conf {proba}≥{CONFIDENCE_CNC}")
        else:
            if product != "MIS":
                product = "MIS"; reasons.append("low confidence")
        if n_rsi is not None and float(n_rsi) < REGIME_NIFTY_RSI_MIN:
            product = "MIS"; reasons.append("weak regime")
        elif n_rsi is not None and float(n_rsi) >= REGIME_NIFTY_RSI_STRONG and trend_ok:
            reasons.append("strong regime+trend")
        if breakout and vol_spike:
            reasons.append("breakout+vol → CNC")
        if pullback:
            if product != "MIS":
                product = "MIS"; reasons.append("pullback → MIS")
        if is_late:
            product = "MIS"; reasons.append(f"after {LATEST_ENTRY_MIS_AFTER}")
    except Exception:
        pass
    return product, ", ".join(reasons) if reasons else None

if "signal" in filtered.columns:
    if "recommended_product" not in filtered.columns:
        filtered["recommended_product"] = pd.NA
    if "recommended_reason" not in filtered.columns:
        filtered["recommended_reason"] = pd.NA
    mask = filtered["signal"] == "BUY"
    if mask.any():
        rp = filtered[mask].apply(lambda r: _decide_product_row(r), axis=1)
        filtered.loc[mask, "recommended_product"] = [x[0] for x in rp]
        filtered.loc[mask, "recommended_reason"] = [x[1] for x in rp]

if BUY_ONLY:
    buys_in_filtered = len(filtered) if not filtered.empty else 0
    if buys_in_filtered < MIN_BUYS_IN_MAIN:
        filtered = _append_relaxed_buys(actionable, filtered)
else:
    buys_in_filtered = int((filtered["signal"] == "BUY").sum()) if not filtered.empty else 0

# Top lists from final filtered set
if not filtered.empty and "signal" in filtered.columns:
    t_buys = filtered[filtered["signal"] == "BUY"].copy()
    t_sells = filtered[filtered["signal"] == "SELL"].copy() if not BUY_ONLY else pd.DataFrame(columns=filtered.columns)
    if "rr" in t_buys.columns:
        t_buys = t_buys.sort_values(by=["cagr", "winrate", "rr"], ascending=[False, False, False])
    else:
        t_buys = t_buys.sort_values(by=["cagr", "winrate"], ascending=[False, False])
    t_sells = t_sells.sort_values(by=["cagr", "winrate"], ascending=[False, False])
    t_buys.head(TOP_BUYS).to_csv(REPORTS_DIR / "top_buys.csv", index=False)
    t_sells.head(TOP_SELLS).to_csv(REPORTS_DIR / "top_sells.csv", index=False)

# --- Save outputs ---
merged.to_csv(REPORTS_DIR / "all_signals_ranked.csv", index=False)
filtered.to_csv(REPORTS_DIR / "filtered_signals_ranked.csv", index=False)

n_buy = len(filtered) if BUY_ONLY else int((filtered["signal"] == "BUY").sum()) if "signal" in filtered.columns else 0
n_sell = 0 if BUY_ONLY else int((filtered["signal"] == "SELL").sum()) if "signal" in filtered.columns else 0
mode_tag = "BUY-only mode" if BUY_ONLY else "BUY/SELL"
msg = (f"✅ Ranked {len(filtered)} actionable signals ({n_buy} BUYs, {n_sell} SELLs) [{mode_tag}] "
       f"(CAGR>{MIN_CAGR}, MaxDD<={MAX_DD})")
print(msg)
logger.info(msg)
print(f"📄 Saved filtered: {REPORTS_DIR/'filtered_signals_ranked.csv'}")
print(f"📄 Saved all: {REPORTS_DIR/'all_signals_ranked.csv'}\n")
if relax_log:
    print(relax_log)
    logger.info(relax_log)

# --- Pretty print trades ---
if not filtered.empty:
    for _, row in filtered.iterrows():
        if row["signal"] == "BUY":
            print(f"📈 {row['ticker']}: BUY at {row['close']} → Target {row['take_profit']} "
                  f"(+{row['profit_pct']}%) | Stop {row['stop_loss']} | "
                  f"CAGR {row['cagr']}%, Winrate {row['winrate']}%")
        else:
            print(f"📉 {row['ticker']}: SELL at {row['close']} → Target {row['take_profit']} "
                  f"| Stop {row['stop_loss']} | CAGR {row['cagr']}%, Winrate {row['winrate']}%")
else:
    print("⚠️ No actionable trades found with current filters.")
    logger.info("No actionable trades found with current filters.")
