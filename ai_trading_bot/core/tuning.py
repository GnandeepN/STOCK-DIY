import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score

from ai_trading_bot.core.config import MODEL_TYPE, RF_PARAMS, LGBM_PARAMS, XGB_PARAMS, LOGS_DIR
from ai_trading_bot.core.data import fetch_ticker
from ai_trading_bot.core.features import build_ml_frame, feature_columns
from ai_trading_bot.core.logger import get_logger


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


optuna = _try_import("optuna")
mlflow = _try_import("mlflow")

logger = get_logger("tuning", LOGS_DIR / "tuning.log")


def default_space(model_type: str, trial) -> Dict[str, Any]:
    mt = model_type.lower()
    if mt == "rf":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            n_jobs=-1,
            random_state=42,
        )
    if mt == "lgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            n_jobs=-1,
            random_state=42,
        )
    if mt == "xgb":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def _get_model(model_type: str, params: Dict[str, Any]):
    mt = model_type.lower()
    if mt == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    if mt == "lgbm":
        LGBMClassifier = None
        try:
            from lightgbm import LGBMClassifier as _LGBM
            LGBMClassifier = _LGBM
        except Exception:
            pass
        if LGBMClassifier is None:
            raise ImportError("lightgbm not installed")
        return LGBMClassifier(**params)
    if mt == "xgb":
        XGBClassifier = None
        try:
            from xgboost import XGBClassifier as _XGB
            XGBClassifier = _XGB
        except Exception:
            pass
        if XGBClassifier is None:
            raise ImportError("xgboost not installed")
        return XGBClassifier(**params)
    raise ValueError(mt)


def tune_ticker(ticker: str,
                model_type: Optional[str] = None,
                n_trials: int = 25,
                timeout: Optional[int] = 180,
                params_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Tune hyperparameters for a single ticker on a small validation split.

    Saves best params to params_dir/<ticker>.json if provided.
    Returns the best params dict or None on failure.
    """
    if optuna is None:
        logger.warning("Optuna not installed; skipping tuning.")
        return None

    mt = (model_type or MODEL_TYPE).lower()

    # Data and features (core only, same as training)
    px = fetch_ticker(ticker)
    if px is None or px.empty:
        logger.error(f"{ticker}: no data for tuning")
        return None
    ml = build_ml_frame(px)
    feats = feature_columns()
    if len(ml) < 80:  # minimal length for quick split
        logger.warning(f"{ticker}: too few rows for tuning")
        return None

    split_n = 30
    train, valid = ml.iloc[:-split_n], ml.iloc[-split_n:]
    X_tr, y_tr = train[feats], train["target"]
    X_va, y_va = valid[feats], valid["target"]

    def objective(trial):
        params = default_space(mt, trial)
        model = _get_model(mt, params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_va, pred)
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = study.best_params

    # Merge with sensible defaults so training has complete param sets
    base = {"rf": RF_PARAMS, "lgbm": LGBM_PARAMS, "xgb": XGB_PARAMS}[mt].copy()
    base.update(best)

    if params_dir:
        params_dir.mkdir(parents=True, exist_ok=True)
        (params_dir / f"{ticker.replace('.', '_')}.json").write_text(json.dumps(base, indent=2))

    # Optional MLflow logging
    if mlflow is not None:
        try:
            mlflow.set_tracking_uri(str((Path.cwd() / "mlruns").resolve()))
            mlflow.set_experiment("ai_trading_bot_tuning")
            with mlflow.start_run(run_name=f"{ticker}_{mt}"):
                mlflow.log_param("ticker", ticker)
                mlflow.log_param("model_type", mt)
                for k, v in base.items():
                    mlflow.log_param(k, v)
                mlflow.log_metric("best_accuracy", float(study.best_value))
        except Exception:
            pass

    logger.info(f"{ticker}: tuned {mt} → best_acc={study.best_value:.3f}")
    return base

