# backtest_walkforward.py (Phase 6: Logging + Retry + Smarter Caching + Equity Export + Model Choice)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from sklearn.model_selection import TimeSeriesSplit

from ai_trading_bot.core.data import fetch_ticker
from ai_trading_bot.core.features import build_ml_frame, add_extra_features, feature_columns
from ai_trading_bot.core.utils_cache import get_cached_ml_frame
from ai_trading_bot.core.config import (
    TICKERS, PROB_THRESHOLD, TREND_FILTER,
    ROUND_TRIP_BPS, SLIPPAGE_BPS, RF_PARAMS, REPORTS_DIR, LOGS_DIR,
    BACKTEST_RETRAIN_FREQ, EXPORT_EQUITY, MODEL_TYPE,
    BACKTEST_FAST, BACKTEST_FAST_RETRAIN_FREQ, BACKTEST_FAST_CV_FOLDS, BACKTEST_FAST_ESTIMATORS,
)
from ai_trading_bot.core.logger import get_logger
from ai_trading_bot.utils.validation import PurgedTimeSeriesSplit

# Optional models
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
import numpy as np


# ------------------------------
# Logging (daily rotation)
# ------------------------------
LOG_FILE = LOGS_DIR / "backtest.log"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = get_logger("backtest", LOG_FILE)

# Silence sklearn deprecation noise for cv='prefit' calibration in backtest
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*cv='prefit'.*CalibratedClassifierCV.*",
)


# ------------------------------
# Caching
# ------------------------------
CACHE_DIR = REPORTS_DIR / "backtest_cache"
EQUITY_DIR = REPORTS_DIR / "backtest_equity"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
if EXPORT_EQUITY:
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------
# Utilities
# ------------------------------
def safe_fetch_ticker(ticker: str, retries: int = 3, delay: int = 5) -> pd.DataFrame | None:
    """Fetch ticker data with retry on transient failures."""
    for attempt in range(1, retries + 1):
        try:
            df = fetch_ticker(ticker)
            if df is not None and not df.empty:
                return df
            logger.warning(f"{ticker}: empty dataframe (attempt {attempt}/{retries})")
        except Exception as e:
            logger.error(f"{ticker}: fetch failed (attempt {attempt}/{retries}): {e}")
        time.sleep(delay)
    logger.error(f"{ticker}: giving up after {retries} attempts.")
    return None


def get_model():
    """Choose ML model by config.MODEL_TYPE."""
    mt = (MODEL_TYPE or "rf").lower()
    # Derive fast-mode flags (accuracy-preserving): only avoid nested parallelism
    fast = bool(BACKTEST_FAST)
    rf_params = RF_PARAMS.copy()
    if fast:
        rf_params["n_jobs"] = 1
    if mt == "rf":
        return RandomForestClassifier(**rf_params)
    if mt == "lgbm":
        if LGBMClassifier is None:
            raise ImportError("MODEL_TYPE=lgbm but LightGBM is not installed. `pip install lightgbm`")
        ne = 300
        nj = 1 if fast else -1
        return LGBMClassifier(n_estimators=ne, random_state=42, n_jobs=nj, verbosity=-1)
    if mt == "xgb":
        if XGBClassifier is None:
            raise ImportError("MODEL_TYPE=xgb but XGBoost is not installed. `pip install xgboost`")
        ne = 400
        nj = 1 if fast else -1
        return XGBClassifier(
            n_estimators=ne, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=nj, use_label_encoder=False, eval_metric="logloss", verbosity=0
        )
    if mt == "ensemble":
        # Build a simple ensemble with defaults
        models = []
        models.append(RandomForestClassifier(**rf_params))
        if LGBMClassifier is not None:
            nj = 1 if fast else -1
            models.append(LGBMClassifier(n_estimators=300, random_state=42, n_jobs=nj, verbosity=-1))
        if XGBClassifier is not None:
            nj = 1 if fast else -1
            models.append(XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=nj, use_label_encoder=False, eval_metric="logloss", verbosity=0
            ))

        class _Ensemble:
            def __init__(self, models):
                self.models = models
                self.classes_ = np.array([0, 1])
            def fit(self, X, y):
                self.fitted = []
                for m in self.models:
                    try:
                        m.fit(X, y)
                        self.fitted.append(m)
                    except Exception:
                        pass
                if not self.fitted:
                    raise RuntimeError("ensemble: no base models fitted")
                return self
            def predict_proba(self, X):
                probs = None
                for m in self.fitted:
                    p = m.predict_proba(X)[:, 1]
                    probs = p if probs is None else probs + p
                probs = probs / len(self.fitted)
                return np.vstack([1 - probs, probs]).T

        return _Ensemble(models)
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


# ------------------------------
# Walk-forward engine
# ------------------------------
def walk_forward(df: pd.DataFrame, ticker: str, min_train: int = 100, start_idx: int = 0,
                 retrain_freq: int = 20) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Walk-forward: train once, then re-train every `retrain_freq` steps.
    Returns:
      ml  : ML dataframe with features
      proba: pd.Series of P(up) aligned to ml index
    """
    # Build features
    ml = get_cached_ml_frame(df, ticker, build_ml_frame)
    ml = add_extra_features(ml, ticker)

    # Selected features
    feats = feature_columns() + [
        "macd", "bollinger_high", "bollinger_low", "stoch",
        "pe_ratio", "forward_pe", "div_yield", "market_cap",
        "news_sentiment", "nifty_rsi", "stock_vs_nifty"
    ]
    feats = [f for f in feats if f in ml.columns]

    proba = pd.Series(index=ml.index, dtype=float)
    thr_buy_series = pd.Series(index=ml.index, dtype=float)
    thr_sell_series = pd.Series(index=ml.index, dtype=float)

    clf = None
    start = max(min_train, start_idx)
    for i in range(start, len(ml) - 1):
        train = ml.iloc[:i]

        # only retrain at start or every retrain_freq days
        if (i == start) or ((i - start) % retrain_freq == 0):
            clf = get_model()
            X_tr, y_tr = train[feats], train["target"]

            # EV-based thresholds (BUY/SELL) using purged CV
            best_thr_buy = PROB_THRESHOLD
            best_thr_sell = PROB_THRESHOLD
            try:
                n_splits = min(5, max(2, len(X_tr) // 120))
                cv = PurgedTimeSeriesSplit(n_splits=n_splits, embargo=5, min_train=80)
                oof = pd.Series(index=X_tr.index, dtype=float)
                for tr_idx, va_idx in cv.split(X_tr):
                    Xtr, ytr = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
                    Xva = X_tr.iloc[va_idx]
                    mdl_ = get_model()
                    mdl_.fit(Xtr, ytr)
                    oof.iloc[va_idx] = mdl_.predict_proba(Xva)[:, 1]
                mask = oof.notna()
                if mask.sum() > 0:
                    ps = oof.values[mask]
                    idx = X_tr.index[mask]
                    # Build simple next-day return and trend
                    close_tr = ml.loc[idx, "Close"].values
                    ret1 = np.append((close_tr[1:] - close_tr[:-1]) / close_tr[:-1], np.nan)
                    ema20 = ml.loc[idx, "ema_20"].values if "ema_20" in ml.columns else np.full_like(close_tr, np.nan)
                    ema50 = ml.loc[idx, "ema_50"].values if "ema_50" in ml.columns else np.full_like(close_tr, np.nan)
                    trend = ema20 > ema50
                    scan = [j / 100 for j in range(45, 71)]

                    def ev_buy(th):
                        sel = (ps >= th) & trend & ~np.isnan(ret1)
                        if sel.sum() == 0:
                            return -1e9
                        r = ret1[sel]
                        wins = r[r > 0]; losses = -r[r <= 0]
                        win_rate = len(wins) / len(r)
                        avg_win = float(wins.mean()) if len(wins) else 0.0
                        avg_loss = float(losses.mean()) if len(losses) else 0.0
                        costs = (ROUND_TRIP_BPS + SLIPPAGE_BPS) / 10000.0
                        return win_rate * avg_win - (1 - win_rate) * avg_loss - costs

                    def ev_sell(th):
                        sel = (ps <= (1 - th)) & (~trend) & ~np.isnan(ret1)
                        if sel.sum() == 0:
                            return -1e9
                        r = -ret1[sel]
                        wins = r[r > 0]; losses = -r[r <= 0]
                        win_rate = len(wins) / len(r)
                        avg_win = float(wins.mean()) if len(wins) else 0.0
                        avg_loss = float(losses.mean()) if len(losses) else 0.0
                        costs = (ROUND_TRIP_BPS + SLIPPAGE_BPS) / 10000.0
                        return win_rate * avg_win - (1 - win_rate) * avg_loss - costs

                    evs_b = [ev_buy(t) for t in scan]
                    evs_s = [ev_sell(t) for t in scan]
                    best_thr_buy = scan[int(np.argmax(evs_b))]
                    best_thr_sell = scan[int(np.argmax(evs_s))]
            except Exception:
                pass

            # Fit with calibration if possible
            if len(train) >= 200:
                cal_n = min(100, len(train) // 5)
                if cal_n >= 30:
                    X_fit, y_fit = X_tr.iloc[:-cal_n], y_tr.iloc[:-cal_n]
                    X_cal, y_cal = X_tr.iloc[-cal_n:], y_tr.iloc[-cal_n:]
                    clf.fit(X_fit, y_fit)
                    try:
                        method = "isotonic" if len(y_cal) >= 100 else "sigmoid"
                        # Avoid deprecated cv='prefit' by using cv=3 on calibration folds
                        cal = CalibratedClassifierCV(estimator=get_model(), cv=3, method=method)
                        cal.fit(X_cal, y_cal)
                        clf = cal
                    except Exception:
                        pass
                else:
                    clf.fit(X_tr, y_tr)
            else:
                clf.fit(X_tr, y_tr)
            current_thr_buy = best_thr_buy
            current_thr_sell = best_thr_sell

        testX = ml.iloc[i:i + 1][feats]
        proba.iloc[i] = clf.predict_proba(testX)[:, 1][0]
        thr_buy_series.iloc[i] = current_thr_buy
        thr_sell_series.iloc[i] = current_thr_sell

    return ml, proba, thr_buy_series.ffill(), thr_sell_series.ffill()


def summarize(px: pd.DataFrame, ml: pd.DataFrame, proba: pd.Series, last_date: str,
              thr_buy_series: pd.Series | None = None,
              thr_sell_series: pd.Series | None = None) -> dict:
    """Compute strategy metrics."""
    # Trend filter
    feats_ok = pd.Series(True, index=ml.index)
    if TREND_FILTER and "ema_20" in ml.columns and "ema_50" in ml.columns:
        feats_ok = ml["ema_20"] > ml["ema_50"]

    # Signal: long when proba >= threshold AND trend_ok
    if thr_buy_series is not None and not thr_buy_series.dropna().empty and thr_sell_series is not None and not thr_sell_series.dropna().empty:
        thr_b = thr_buy_series.reindex(ml.index).fillna(PROB_THRESHOLD)
        thr_s = thr_sell_series.reindex(ml.index).fillna(PROB_THRESHOLD)
        signal_long = (proba >= thr_b) & feats_ok
        signal_short = (proba <= (1 - thr_s)) & (~feats_ok)
    else:
        signal_long = (proba >= PROB_THRESHOLD) & feats_ok
        signal_short = (proba <= (1 - PROB_THRESHOLD)) & (~feats_ok)

    # Avoid pandas FutureWarning by casting before fillna
    sig_bool = signal_long.astype("boolean")
    pos_long = sig_bool.shift(1).astype("boolean").fillna(False).astype(int)
    sig_short_bool = signal_short.astype("boolean")
    pos_short = sig_short_bool.shift(1).astype("boolean").fillna(False).astype(int)

    # Returns & costs
    idx = ml.index
    rets = px["Close"].pct_change().reindex(idx).fillna(0.0)
    pos = (pos_long - pos_short)
    trade_change = pos.diff().fillna(pos)
    trade_days = trade_change != 0
    cost_per_trade = (ROUND_TRIP_BPS + SLIPPAGE_BPS) / 10000.0 / 2.0
    costs = trade_days.astype(float) * cost_per_trade
    strat_rets = (pos_long * rets) - (pos_short * rets) - costs

    equity = (1 + strat_rets).cumprod()
    cagr = equity.iloc[-1] ** (252 / len(equity)) - 1 if len(equity) > 252 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    winrate = ((strat_rets > 0) & (trade_days)).sum() / max(1, trade_days.sum())

    return dict(
        cagr=round(float(cagr) * 100, 2) if pd.notna(cagr) else None,
        max_dd=round(float(dd) * 100, 2) if pd.notna(dd) else None,
        winrate=round(float(winrate) * 100, 2),
        n_days=len(equity),
        n_trades=int(trade_days.sum() / 2),
        last_date=last_date,
        equity=equity,
    )


def _compute_psi(old: pd.Series, new: pd.Series, bins: int = 10) -> float:
    try:
        q = np.linspace(0, 1, bins + 1)
        cuts = np.unique(np.quantile(pd.concat([old, new]).dropna(), q))
        if len(cuts) < 3:
            return 0.0
        old_hist = np.histogram(old.dropna(), bins=cuts)[0] + 1e-6
        new_hist = np.histogram(new.dropna(), bins=cuts)[0] + 1e-6
        old_pct = old_hist / old_hist.sum()
        new_pct = new_hist / new_hist.sum()
        psi = np.sum((new_pct - old_pct) * np.log(new_pct / old_pct))
        return float(psi)
    except Exception:
        return 0.0


def run_backtest_one(ticker: str, retrain_freq: int) -> dict:
    px = safe_fetch_ticker(ticker)
    if px is None or px.empty:
        return dict(ticker=ticker, error="no_data")

    current_days = len(px)
    last_date = str(px.index[-1].date())

    cache_file = CACHE_DIR / f"{ticker.replace('.','_')}_backtest.csv"

    # Check cache
    start_idx = 0
    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file)
            cached_days = int(cached["n_days"].iloc[0])
            cached_date = str(cached.get("last_date", [None])[0])

            if cached_days == current_days and cached_date == last_date:
                # Fresh
                return cached.iloc[0].to_dict()
            # Resume (incremental)
            start_idx = max(0, cached_days - 1)
            logger.info(f"{ticker}: cache outdated, resuming from idx {start_idx}")
        except Exception as e:
            logger.warning(f"{ticker}: cannot read cache, recomputing. ({e})")

    # Walk-forward
    ml, proba, thr_b, thr_s = walk_forward(px, ticker, min_train=400, start_idx=start_idx, retrain_freq=retrain_freq)

    # Metrics
    s = summarize(px, ml, proba, last_date, thr_b, thr_s)
    summary = dict(
        ticker=ticker,
        cagr=s["cagr"],
        max_dd=s["max_dd"],
        winrate=s["winrate"],
        n_days=s["n_days"],
        n_trades=s["n_trades"],
        last_date=last_date,
    )

    # Drift summary (simple PSI over rolling halves of the window on core features)
    try:
        core_feats = [c for c in ["rsi", "atr", "ema_20", "ema_50", "ema_200", "ret_1", "ret_5", "ret_10"] if c in ml.columns]
        if core_feats:
            mid = len(ml) // 2
            drift_rows = []
            for f in core_feats:
                psi = _compute_psi(ml[f].iloc[:mid], ml[f].iloc[mid:])
                drift_rows.append(dict(ticker=ticker, feature=f, psi=round(psi, 4)))
            drift_df = pd.DataFrame(drift_rows)
            drift_out = REPORTS_DIR / "drift_summary.csv"
            if drift_out.exists():
                # append
                prev = pd.read_csv(drift_out)
                prev = prev[prev["ticker"] != ticker]
                pd.concat([prev, drift_df], ignore_index=True).to_csv(drift_out, index=False)
            else:
                drift_df.to_csv(drift_out, index=False)
    except Exception:
        pass

    # Save equity (optional)
    if EXPORT_EQUITY and isinstance(s["equity"], pd.Series):
        (EQUITY_DIR / f"{ticker.replace('.','_')}_equity.csv").write_text(
            pd.DataFrame({"date": s["equity"].index.strftime("%Y-%m-%d"), "equity": s["equity"].values})
            .to_csv(index=False)
        )

    # Save cache summary
    pd.DataFrame([summary]).to_csv(cache_file, index=False)
    return summary


def main():
    """Main function to run the backtest."""
    freq = BACKTEST_RETRAIN_FREQ
    logger.info("==== Backtest start ====")
    logger.info(f"Universe: {len(TICKERS)} tickers, model={MODEL_TYPE}, retrain_freq={freq}, fast={BACKTEST_FAST}")

    # Use half the cores to keep the laptop responsive
    n_jobs = max(1, cpu_count() // 2)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_backtest_one)(t, retrain_freq=freq) for t in TICKERS
    )

    df = pd.DataFrame(results)
    out = REPORTS_DIR / "backtest_walkforward_summary.csv"
    df.to_csv(out, index=False)

    logger.info(f"Saved summary → {out}")
    logger.info("==== Backtest done ====")

    # Also print a compact table in terminal
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(df.to_string(index=False))
    print(f"\n📄 Logs: {LOG_FILE}")


if __name__ == "__main__":
    main()
