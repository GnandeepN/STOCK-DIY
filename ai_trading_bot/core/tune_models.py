"""
Tune per-ticker hyperparameters with Optuna and save best params.
Usage:
  venv/bin/python tune_models.py --model lgbm --timeout 180 --trials 30
"""

import argparse
from pathlib import Path
from joblib import Parallel, delayed

from ai_trading_bot.core.config import TICKERS, LOGS_DIR
from ai_trading_bot.core.tuning import tune_ticker
from ai_trading_bot.core.logger import get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Override MODEL_TYPE: rf|lgbm|xgb")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    logger = get_logger("tune_models", LOGS_DIR / "tune_models.log")
    params_dir = Path(__file__).resolve().parent / "models" / "params"

    logger.info(f"==== Tuning start (model={args.model or 'config'}, timeout={args.timeout}s, trials={args.trials}) ====")

    if args.parallel:
        Parallel(n_jobs=-1)(
            delayed(tune_ticker)(t, model_type=args.model, n_trials=args.trials, timeout=args.timeout, params_dir=params_dir)
            for t in TICKERS
        )
    else:
        for t in TICKERS:
            try:
                tune_ticker(t, model_type=args.model, n_trials=args.trials, timeout=args.timeout, params_dir=params_dir)
            except Exception as e:
                logger.error(f"{t}: tuning failed → {e}")

    logger.info("==== Tuning completed ====")


if __name__ == "__main__":
    main()

