"""
run_all.py — orchestrates the full pipeline and surfaces logs to terminal.

Steps:
 1) data.py                 → update or load cached data
 2) train_models.py         → train per-ticker models
 3) backtest_walkforward.py → backtest with walk-forward
 4) signals_today.py        → score today’s signals
 5) filter_signals.py       → rank and filter signals

All steps stream output to terminal and log to logs/run_all.log.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path

from ai_trading_bot.core.config import LOGS_DIR
from ai_trading_bot.core.logger import get_logger


def run(cmd: list[str]) -> int:
    """Run a command, streaming stdout/stderr to terminal; return exit code."""
    env = os.environ.copy()
    # Silence sklearn FutureWarning about cv='prefit' calibration in child processes
    warn_filter = "ignore:.*cv='prefit'.*CalibratedClassifierCV.*:FutureWarning"
    prev = env.get("PYTHONWARNINGS", "")
    env["PYTHONWARNINGS"] = warn_filter if not prev else f"{prev},{warn_filter}"
    # Add the project root to PYTHONPATH to allow absolute imports
    project_root = Path(__file__).resolve().parent.parent.parent
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{python_path}" if python_path else str(project_root)
    return subprocess.run(cmd, env=env).returncode


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger("run_all", LOGS_DIR / "run_all.log")

    parser = argparse.ArgumentParser(description="Run full AI trading pipeline")
    parser.add_argument("--force", action="store_true", help="Force data refresh and model retrain")
    parser.add_argument("--parallel", action="store_true", help="Parallelize model training")
    parser.add_argument("--skip-data", action="store_true", help="Skip data update step")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtest step")
    parser.add_argument("--skip-signals", action="store_true", help="Skip signals step")
    parser.add_argument("--skip-filter", action="store_true", help="Skip filter step")
    args = parser.parse_args()

    py = sys.executable

    # 1) Data update
    if not args.skip_data:
        logger.info("==== Step 1/5: Data update ====")
        code = run([py, "-m", "ai_trading_bot.core.data"] + (["--force"] if args.force else []))
        if code != 0:
            logger.error(f"Data step failed with exit code {code}")
            return code

    # 2) Train models
    if not args.skip_train:
        logger.info("==== Step 2/5: Train models ====")
        targs = [py, "-m", "ai_trading_bot.core.train_models"]
        if args.force:
            targs.append("--force")
        if args.parallel:
            targs.append("--parallel")
        code = run(targs)
        if code != 0:
            logger.error(f"Train step failed with exit code {code}")
            return code

    # 3) Backtest
    if not args.skip_backtest:
        logger.info("==== Step 3/5: Backtest walk-forward ====")
        code = run([py, "-m", "ai_trading_bot.core.backtest_walkforward"])
        if code != 0:
            logger.error(f"Backtest step failed with exit code {code}")
            return code

    # 4) Signals today
    if not args.skip_signals:
        logger.info("==== Step 4/5: Signals today ====")
        code = run([py, "-m", "ai_trading_bot.core.signals_today"])
        if code != 0:
            logger.error(f"Signals step failed with exit code {code}")
            return code

    # 5) Filter signals
    if not args.skip_filter:
        logger.info("==== Step 5/5: Filter signals ====")
        code = run([py, "-m", "ai_trading_bot.core.filter_signals"])
        if code != 0:
            logger.error(f"Filter step failed with exit code {code}")
            return code

    logger.info("==== Pipeline complete ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
