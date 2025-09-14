# signals_today.py (Phase 4: Unified Logging + Freshness-Aware)

import joblib
import pandas as pd
import numpy as np
from ai_trading_bot.core.config import (
    TICKERS, MODELS_DIR, REPORTS_DIR,
    PROB_THRESHOLD, TREND_FILTER, TP_ATR_MULT, SL_ATR_MULT, LOGS_DIR
)
from ai_trading_bot.core.data import fetch_ticker
from ai_trading_bot.core.features import add_indicators
from ai_trading_bot.core.logger import get_logger   # ✅ use shared logger
import yfinance as yf
from ai_trading_bot.core.config import (
    REGIME_NIFTY_RSI_MIN, REGIME_NIFTY_RSI_STRONG, RR_MIN, SWING_LOOKBACK, VOLUME_SPIKE_MULT,
    PULLBACK_ATR_MULT, RR_INTRADAY_MAX, RR_CNC_MIN, CONFIDENCE_CNC, LATEST_ENTRY_MIS_AFTER,
    APPLY_GAP_FILTER, GAP_ATR_MAX
)
from datetime import datetime
import pytz

# --- Setup logger ---
logger = get_logger("signals_today", LOGS_DIR / "signals_today.log")


def signal_row(ticker: str):
    # --- Load price data ---
    raw = fetch_ticker(ticker)
    if raw.empty:
        logger.error(f"{ticker}: No data available")
        return dict(
            ticker=ticker,
            error="No data",
            signal="ERROR",
            last_date=None,
            close=None,
            proba_up=None,
            trend_ok=None,
            stop_loss=None,
            take_profit=None,
        )

    last_date = str(raw.index[-1].date())
    try:
        last_close = float(raw["Close"].tail(1).item())
    except Exception:
        last_close = float(raw["Close"].iloc[-1])

    # --- Load model ---
    model_path = MODELS_DIR / f"{ticker.replace('.','_')}.joblib"
    if not model_path.exists():
        logger.error(f"{ticker}: Model not found")
        return dict(
            ticker=ticker,
            error="Model not found",
            signal="ERROR",
            last_date=last_date,
            close=last_close,
            proba_up=None,
            trend_ok=None,
            stop_loss=None,
            take_profit=None,
        )

    bundle = joblib.load(model_path)
    clf = bundle["model"]
    feats = bundle["features"]
    model_date = bundle.get("last_date", None)
    calibrated = bool(bundle.get("calibrated", False))
    calibration_method = bundle.get("calibration_method", None)
    best_threshold_buy = float(bundle.get("best_threshold_buy", bundle.get("best_threshold", PROB_THRESHOLD)))
    best_threshold_sell = float(bundle.get("best_threshold_sell", bundle.get("best_threshold", PROB_THRESHOLD)))

    # --- Freshness check ---
    if model_date and model_date < last_date:
        logger.warning(
            f"{ticker}: Model stale (trained={model_date}, data={last_date}) → retrain recommended."
        )

    # --- Build indicators only (match training core features) ---
    ml = add_indicators(raw)
    ml = ml.ffill()
    if ml.empty:
        logger.error(f"{ticker}: No rows after indicators")
        return dict(ticker=ticker, error="No rows after indicators", signal="ERROR",
                    last_date=last_date, close=last_close,
                    proba_up=None, trend_ok=None, stop_loss=None, take_profit=None)
    latest = ml.iloc[-1:].copy()

    # --- Predict probability ---
    try:
        proba_up = float(clf.predict_proba(latest[feats])[:, 1][0])
    except Exception as e:
        logger.error(f"{ticker}: Feature mismatch → {e}")
        return dict(
            ticker=ticker,
            error="Feature mismatch with model",
            signal="ERROR",
            last_date=last_date,
            close=last_close,
            proba_up=None,
            trend_ok=None,
            stop_loss=None,
            take_profit=None,
        )

    # --- Trend filter ---
    pass_trend = True
    if TREND_FILTER and "ema_20" in latest.columns and "ema_50" in latest.columns:
        pass_trend = bool(latest["ema_20"].iloc[0] > latest["ema_50"].iloc[0])

    # --- Multi-timeframe confirmation (weekly EMA10>EMA20) ---
    mtf_ok = True
    try:
        # Resample to weekly
        wk = raw[["Close"]].resample("W-FRI").last().dropna()
        wk["ema_10_w"] = wk["Close"].ewm(span=10, adjust=False).mean()
        wk["ema_20_w"] = wk["Close"].ewm(span=20, adjust=False).mean()
        if len(wk) >= 20:
            mtf_ok = bool(wk["ema_10_w"].iloc[-1] > wk["ema_20_w"].iloc[-1])
    except Exception:
        mtf_ok = True

    # --- Action decision ---
    action = "HOLD"
    if pass_trend and proba_up >= best_threshold_buy:
        action = "BUY"
    elif (not pass_trend) and (proba_up <= (1 - best_threshold_sell)):
        action = "SELL"

    # --- Price, SL, TP (directional) ---
    price = last_close
    atr = float(latest["atr"].iloc[0]) if "atr" in latest else 0.0
    sl = None
    tp = None
    if atr > 0:
        if action == "BUY":
            sl = round(price - SL_ATR_MULT * atr, 2)
            tp = round(price + TP_ATR_MULT * atr, 2)
        elif action == "SELL":
            sl = round(price + SL_ATR_MULT * atr, 2)
            tp = round(price - TP_ATR_MULT * atr, 2)

    # --- Extra features for transparency ---
    extras = {}
    for f in [
        "atr", "rsi", "macd", "stoch", "pe_ratio", "forward_pe", "div_yield",
        "market_cap", "news_sentiment", "nifty_rsi", "stock_vs_nifty"
    ]:
        if f in latest.columns:
            val = latest[f].iloc[0]
            extras[f] = round(float(val), 3) if pd.notna(val) else None

    # Add avg volume and sector (if available)
    try:
        if "Volume" in raw.columns:
            extras["avg_volume_20"] = int(raw["Volume"].tail(20).mean())
    except Exception:
        pass
    try:
        sec = yf.Ticker(ticker).info.get("sector")
        if sec:
            extras["sector"] = sec
        ind = yf.Ticker(ticker).info.get("industry")
        if ind:
            extras["industry"] = ind
    except Exception:
        pass

    # --- Entry quality metrics ---
    rr = None
    if sl is not None and tp is not None and price > 0:
        if action == "BUY":
            risk = max(1e-6, price - sl)
            reward = max(0.0, tp - price)
        else:
            risk = max(1e-6, sl - price)
            reward = max(0.0, price - tp)
        rr = round(float(reward / risk), 2) if risk > 0 else None
    extras["rr"] = rr

    # Breakout and pullback (for BUY bias)
    breakout_ok = None
    pullback_ok = None
    vol_spike = None
    try:
        swing_high = float(raw["High"].iloc[-SWING_LOOKBACK-1:-1].max())
        breakout_ok = bool(price > swing_high)
        if "Volume" in raw.columns:
            vol_spike = bool(raw["Volume"].iloc[-1] >= VOLUME_SPIKE_MULT * raw["Volume"].tail(20).mean())
    except Exception:
        breakout_ok = None
        vol_spike = None
    try:
        if "ema_20" in latest and atr > 0:
            pullback_ok = bool(abs(price - float(latest["ema_20"].iloc[0])) <= PULLBACK_ATR_MULT * atr and pass_trend)
    except Exception:
        pullback_ok = None
    extras["breakout_ok"] = breakout_ok
    extras["vol_spike"] = vol_spike
    extras["pullback_ok"] = pullback_ok
    extras["mtf_trend_ok"] = mtf_ok
    # Gap/volatility proxy (ATR/Close)
    try:
        extras["atr_close_ratio"] = round(float(atr / price), 4) if price > 0 else None
        extras["gap_ok"] = bool((atr / price) <= GAP_ATR_MAX) if (atr and price) else True
    except Exception:
        extras["gap_ok"] = True

    # Regime gating (Nifty RSI)
    regime_ok = True
    try:
        nrsi = extras.get("nifty_rsi")
        if nrsi is not None:
            regime_ok = bool(float(nrsi) >= REGIME_NIFTY_RSI_MIN)
    except Exception:
        regime_ok = True
    extras["regime_ok"] = regime_ok

    # Final action gating for BUY entries
    if action == "BUY":
        if not (pass_trend and mtf_ok and regime_ok):
            action = "HOLD"
        # Require quality entry: RR and (breakout+volume spike OR pullback)
        if rr is None or rr < RR_MIN:
            action = "HOLD"
        else:
            if breakout_ok is True:
                if not vol_spike:
                    action = "HOLD"
            elif pullback_ok is not True:
                action = "HOLD"
        # Optional gap filter
        if APPLY_GAP_FILTER and (extras.get("gap_ok") is False):
            action = "HOLD"

    # --- Product decision (CNC vs MIS) for BUYs ---
    recommended_product = None
    recommended_reason = None
    try:
        # Time of day (IST)
        ist = pytz.timezone("Asia/Kolkata")
        now_t = datetime.now(ist).strftime("%H:%M")
        latest_cut = LATEST_ENTRY_MIS_AFTER
        is_late = now_t >= latest_cut
        # Start optimistic with CNC, then veto down to MIS
        product = "CNC"
        reasons = []
        if action != "BUY":
            product = None
        else:
            # RR rules
            if rr is not None and rr < RR_INTRADAY_MAX:
                product = "MIS"; reasons.append(f"RR {rr}< {RR_INTRADAY_MAX}")
            elif rr is not None and rr >= RR_CNC_MIN:
                # keep CNC
                reasons.append(f"RR {rr}≥ {RR_CNC_MIN}")
            # Confidence
            if proba_up is not None and float(proba_up) >= CONFIDENCE_CNC:
                reasons.append(f"conf {proba_up}≥{CONFIDENCE_CNC}")
            else:
                if product != "MIS":
                    product = "MIS"; reasons.append("low confidence")
            # Regime
            if not regime_ok:
                product = "MIS"; reasons.append("weak regime")
            elif (extras.get("nifty_rsi") is not None and float(extras["nifty_rsi"]) >= REGIME_NIFTY_RSI_STRONG and pass_trend):
                reasons.append("strong regime+trend")
            # Signal type
            if breakout_ok and vol_spike:
                reasons.append("breakout+vol → CNC")
            if pullback_ok:
                if product != "MIS":
                    product = "MIS"; reasons.append("pullback → MIS")
            # Time veto
            if is_late:
                product = "MIS"; reasons.append(f"after {latest_cut}")
        recommended_product = product
        recommended_reason = ", ".join(reasons) if reasons else None
    except Exception:
        pass

    logger.info(f"{ticker}: {action} | proba_up={proba_up:.3f}{' (cal)' if calibrated else ''}, price={price}, SL={sl}, TP={tp}")
    return dict(
        ticker=ticker,
        last_date=last_date,
        model_date=model_date,
        close=round(price, 2),
        proba_up=round(proba_up, 3),
        calibrated=calibrated,
        calibration_method=calibration_method,
        threshold=round(best_threshold_buy, 3),
        trend_ok=pass_trend,
        signal=action,
        stop_loss=sl,
        take_profit=tp,
        **extras
        , recommended_product=recommended_product,
        recommended_reason=recommended_reason
    )


if __name__ == "__main__":
    rows = []
    for t in TICKERS:
        try:
            rows.append(signal_row(t))
        except Exception as e:
            logger.exception(f"{t}: failed to generate signal")
            rows.append(dict(ticker=t, error=str(e)))

    df = pd.DataFrame(rows)
    out = REPORTS_DIR / "signals_today.csv"
    df.to_csv(out, index=False)

    print(df.to_string(index=False))
    print(f"\n📄 Saved: {out}")
    logger.info(f"Saved signals → {out}")
    logger.info("==== Signals generation completed ====")
