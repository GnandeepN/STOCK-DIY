# train_models.py (Phase 6: Unified Logging + Configurable Models + Retry + Parallel)

import argparse
import joblib
import time
from pathlib import Path
from datetime import date
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score

from ai_trading_bot.core.config import TICKERS, MODELS_DIR, RF_PARAMS, LGBM_PARAMS, XGB_PARAMS, LOGS_DIR, MODEL_TYPE, ROUND_TRIP_BPS, SLIPPAGE_BPS
import json
from ai_trading_bot.core.data import fetch_ticker
from ai_trading_bot.core.features import build_ml_frame, feature_columns
from ai_trading_bot.core.utils_cache import get_cached_ml_frame
from ai_trading_bot.core.logger import get_logger   # ✅ use shared logger

# Optional imports
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import List
from sklearn.metrics import brier_score_loss
from ai_trading_bot.core.utils.validation import PurgedTimeSeriesSplit

# --- Setup logger ---
logger = get_logger("train_models", LOGS_DIR / "train_models.log")

# Silence sklearn deprecation noise for cv='prefit' calibration (we keep logic lightweight)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*cv='prefit'.*CalibratedClassifierCV.*",
)


class EnsembleClassifier:
    """Average-probability ensemble over available base models (RF/LGBM/XGB)."""

    def __init__(self, rf_params, lgbm_params, xgb_params):
        self.rf_params = rf_params
        self.lgbm_params = lgbm_params
        self.xgb_params = xgb_params
        self.models: List = []
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.models = []
        # RF
        try:
            rf = RandomForestClassifier(**self.rf_params)
            rf.fit(X, y)
            self.models.append(rf)
        except Exception:
            pass
        # LGBM
        if LGBMClassifier is not None:
            try:
                lgbm = LGBMClassifier(**self.lgbm_params)
                lgbm.fit(X, y)
                self.models.append(lgbm)
            except Exception:
                pass
        # XGB
        if XGBClassifier is not None:
            try:
                xgb = XGBClassifier(**self.xgb_params)
                xgb.fit(X, y)
                self.models.append(xgb)
            except Exception:
                pass
        if not self.models:
            raise RuntimeError("EnsembleClassifier: no base models available")
        return self

    def predict_proba(self, X):
        probs = None
        for m in self.models:
            p = m.predict_proba(X)[:, 1]
            probs = p if probs is None else probs + p
        probs = probs / len(self.models)
        return np.vstack([1 - probs, probs]).T


def safe_fetch_ticker(ticker: str, retries: int = 3, delay: int = 5):
    """Fetch ticker data with retry on failure."""
    for attempt in range(1, retries + 1):
        try:
            df = fetch_ticker(ticker)
            if not df.empty:
                return df
            logger.warning(f"{ticker}: Empty data returned (attempt {attempt}/{retries})")
        except Exception as e:
            logger.error(f"{ticker}: Fetch attempt {attempt} failed: {e}")
        time.sleep(delay)
    return None


def _load_tuned_params(ticker: str, model_type: str):
    params_path = MODELS_DIR / "params" / f"{ticker.replace('.', '_')}.json"
    if params_path.exists():
        try:
            return json.loads(params_path.read_text())
        except Exception:
            pass
    return None


def get_model(ticker: str = None):
    """Return the ML model based on config.MODEL_TYPE."""
    if MODEL_TYPE.lower() == "rf":
        params = _load_tuned_params(ticker, "rf") or RF_PARAMS
        return RandomForestClassifier(**params)
    elif MODEL_TYPE.lower() == "lgbm":
        if LGBMClassifier is None:
            raise ImportError("LightGBM is not installed. Run `pip install lightgbm`.")
        params = _load_tuned_params(ticker, "lgbm") or LGBM_PARAMS
        return LGBMClassifier(**params)
    elif MODEL_TYPE.lower() == "xgb":
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed. Run `pip install xgboost`.")
        params = _load_tuned_params(ticker, "xgb") or XGB_PARAMS
        return XGBClassifier(**params)
    elif MODEL_TYPE.lower() == "ensemble":
        rf = _load_tuned_params(ticker, "rf") or RF_PARAMS
        lgbm = _load_tuned_params(ticker, "lgbm") or LGBM_PARAMS
        xgb = _load_tuned_params(ticker, "xgb") or XGB_PARAMS
        return EnsembleClassifier(rf, lgbm, xgb)
    elif MODEL_TYPE.lower() == "stacking":
        # Simple stacking with logistic meta-learner on predicted probabilities
        estimators = []
        estimators.append(("rf", RandomForestClassifier(**( _load_tuned_params(ticker, "rf") or RF_PARAMS))))
        if LGBMClassifier is not None:
            estimators.append(("lgbm", LGBMClassifier(**( _load_tuned_params(ticker, "lgbm") or LGBM_PARAMS))))
        if XGBClassifier is not None:
            estimators.append(("xgb", XGBClassifier(**( _load_tuned_params(ticker, "xgb") or XGB_PARAMS))))
        from sklearn.ensemble import StackingClassifier as SKStack
        meta = LogisticRegression(max_iter=1000)
        return SKStack(estimators=estimators, final_estimator=meta, stack_method="predict_proba", passthrough=False, n_jobs=-1)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


def train_one(ticker: str, force: bool = False):
    outpath = MODELS_DIR / f"{ticker.replace('.','_')}.joblib"

    # --- Fetch data ---
    raw = safe_fetch_ticker(ticker)
    if raw is None or raw.empty:
        logger.error(f"{ticker}: No data available after retries. Skipping.")
        return None

    last_date = str(raw.index[-1].date())

    # --- Build ML frame (needed to compare features reliably) ---
    ml = get_cached_ml_frame(raw, ticker, build_ml_frame)

    # Use only core, stable features to maximize coverage
    feats = feature_columns()

    # --- Skip retrain if model is fresh AND features match ---
    if outpath.exists() and not force:
        try:
            bundle = joblib.load(outpath)
            model_date = bundle.get("last_date")
            saved_feats = bundle.get("features") or []
            saved_type = bundle.get("model_type")
            same_feats = list(saved_feats) == list(feats)
            same_type = (saved_type or MODEL_TYPE) == MODEL_TYPE
            if (model_date == last_date) and same_feats and same_type:
                logger.info(f"⏩ {ticker}: model fresh and features unchanged; skipping retrain.")
                return None
            else:
                if model_date == last_date and (not same_feats or not same_type):
                    logger.info(f"🔁 {ticker}: features or model_type changed; retraining.")
        except Exception:
            logger.warning(f"{ticker}: model file unreadable, forcing retrain...")

    # --- Train/Test split ---
    split_n = 30  # reduced holdout to increase training coverage
    if len(ml) <= split_n + 30:
        logger.warning(f"{ticker}: Not enough rows for training. Skipping.")
        # Remove any stale model to avoid feature-mismatch at inference
        if outpath.exists():
            try:
                outpath.unlink()
                logger.info(f"{ticker}: removed stale model → insufficient data to retrain.")
            except Exception:
                logger.warning(f"{ticker}: could not remove stale model at {outpath}")
        return None

    train, test = ml.iloc[:-split_n], ml.iloc[-split_n:]
    X_tr_full, y_tr_full = train[feats], train["target"]
    X_te, y_te = test[feats], test["target"]

    # --- Purged CV and EV-based thresholding ---
    best_thr_buy = 0.55
    best_thr_sell = 0.55
    try:
        n_splits = min(5, max(2, len(X_tr_full) // 120))
        cv = PurgedTimeSeriesSplit(n_splits=n_splits, embargo=5, min_train=80)
        oof = np.full(len(X_tr_full), np.nan)
        for tr_idx, va_idx in cv.split(X_tr_full):
            Xtr, ytr = X_tr_full.iloc[tr_idx], y_tr_full.iloc[tr_idx]
            Xva = X_tr_full.iloc[va_idx]
            mdl = get_model(ticker)
            mdl.fit(Xtr, ytr)
            oof[va_idx] = mdl.predict_proba(Xva)[:, 1]
        mask = ~np.isnan(oof)
        if mask.sum() > 0:
            ys = y_tr_full.values[mask]
            ps = oof[mask]
            idx = y_tr_full.index[mask]
            # Build simple next-day return and trend for EV
            close = ml.loc[idx, "Close"].values
            ret1 = np.append((close[1:] - close[:-1]) / close[:-1], np.nan)
            # align ret1 to same length; last NaN ignored by mask2
            ema20 = ml.loc[idx, "ema_20"].values if "ema_20" in ml.columns else np.full_like(close, np.nan)
            ema50 = ml.loc[idx, "ema_50"].values if "ema_50" in ml.columns else np.full_like(close, np.nan)
            trend = ema20 > ema50
            scan = [i / 100 for i in range(45, 71)]

            def ev_buy(thr):
                sel = (ps >= thr) & trend & ~np.isnan(ret1)
                if sel.sum() == 0:
                    return -1e9
                r = ret1[sel]
                wins = r[r > 0]; losses = -r[r <= 0]
                win_rate = len(wins) / len(r)
                avg_win = float(wins.mean()) if len(wins) else 0.0
                avg_loss = float(losses.mean()) if len(losses) else 0.0
                costs = (ROUND_TRIP_BPS + SLIPPAGE_BPS) / 10000.0
                return win_rate * avg_win - (1 - win_rate) * avg_loss - costs

            def ev_sell(thr):
                sel = (ps <= (1 - thr)) & (~trend) & ~np.isnan(ret1)
                if sel.sum() == 0:
                    return -1e9
                r = -ret1[sel]  # short gains when price drops
                wins = r[r > 0]; losses = -r[r <= 0]
                win_rate = len(wins) / len(r)
                avg_win = float(wins.mean()) if len(wins) else 0.0
                avg_loss = float(losses.mean()) if len(losses) else 0.0
                costs = (ROUND_TRIP_BPS + SLIPPAGE_BPS) / 10000.0
                return win_rate * avg_win - (1 - win_rate) * avg_loss - costs

            evs_buy = [ev_buy(t) for t in scan]
            evs_sell = [ev_sell(t) for t in scan]
            best_thr_buy = scan[int(np.argmax(evs_buy))]
            best_thr_sell = scan[int(np.argmax(evs_sell))]
    except Exception as e:
        logger.warning(f"{ticker}: EV threshold selection failed → {e}")

    # --- Split off a small calibration set from the end of training ---
    cal_method = None
    calibrated = False
    X_tr, y_tr = X_tr_full, y_tr_full
    X_cal = y_cal = None
    if len(train) >= 200:
        cal_n = min(100, len(train) // 5)
        if cal_n >= 30:
            X_tr, y_tr = X_tr_full.iloc[:-cal_n], y_tr_full.iloc[:-cal_n]
            X_cal, y_cal = X_tr_full.iloc[-cal_n:], y_tr_full.iloc[-cal_n:]

    # --- Train base model ---
    clf_base = get_model(ticker)
    clf_base.fit(X_tr, y_tr)

    # --- Calibrate with time-based folds; compare Brier score and keep best ---
    clf = clf_base
    if X_cal is not None and y_cal is not None:
        try:
            method = "isotonic" if len(y_cal) >= 100 else "sigmoid"
            cv_cal = PurgedTimeSeriesSplit(n_splits=3, embargo=3, min_train=50)
            cal = CalibratedClassifierCV(estimator=get_model(ticker), cv=cv_cal, method=method)
            cal.fit(X_cal, y_cal)
            # Compare Brier on holdout
            p_base = clf_base.predict_proba(X_te)[:, 1]
            p_cal = cal.predict_proba(X_te)[:, 1]
            b_base = brier_score_loss(y_te, p_base)
            b_cal = brier_score_loss(y_te, p_cal)
            if b_cal <= b_base:
                clf = cal; calibrated = True; cal_method = method
        except Exception as e:
            logger.warning(f"{ticker}: calibration failed → {e}")

    # --- Evaluate on holdout with chosen threshold ---
    proba = clf.predict_proba(X_te)[:, 1]
    # For simple test accuracy reporting, use BUY threshold
    pred = (proba >= best_thr_buy).astype(int)
    acc = accuracy_score(y_te, pred)

    # --- Save model ---
    joblib.dump(
        {
            "model": clf,
            "features": feats,
            "last_date": last_date,
            "model_type": MODEL_TYPE,
            "calibrated": calibrated,
            "calibration_method": cal_method,
            "best_threshold": best_thr_buy,
            "best_threshold_buy": best_thr_buy,
            "best_threshold_sell": best_thr_sell,
        },
        outpath
    )
    cal_tag = f", cal={cal_method}" if calibrated else ""
    logger.info(f"✅ {ticker}: {MODEL_TYPE} acc={acc:.3f} thr={best_thr_buy:.2f} (train={len(train)}, test={len(test)}{cal_tag}) saved={outpath.name}")
    return acc


def main():
    """Main function to train models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")
    parser.add_argument("--parallel", action="store_true", help="Train tickers in parallel")
    args = parser.parse_args()

    if args.parallel:
        Parallel(n_jobs=-1)(
            delayed(train_one)(t, force=args.force) for t in TICKERS
        )
    else:
        for t in TICKERS:
            try:
                train_one(t, force=args.force)
            except Exception as e:
                logger.error(f"{t}: {e}")

    logger.info("==== Training completed ====")
    print(f"📄 Logs saved to {LOGS_DIR}/train_models.log")


if __name__ == "__main__":
    main()
