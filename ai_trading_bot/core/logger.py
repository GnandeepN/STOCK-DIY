import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Union
from ai_trading_bot.core.config import LOGS_DIR, LOG_LEVEL
import time

# Ensure log dir exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Default log file (used when no file is provided)
DEFAULT_LOG_FILE = LOGS_DIR / "ai_trading_bot.log"


def _level_to_int(level: Union[str, int]) -> int:
    if isinstance(level, int):
        return level
    try:
        return getattr(logging, str(level).upper())
    except AttributeError:
        return logging.INFO


def _prune_old_logs(retain_days: int = 3):
    try:
        cutoff = time.time() - retain_days * 24 * 3600
        for p in LOGS_DIR.glob("*.log*"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
            except Exception:
                pass
    except Exception:
        pass


def get_logger(name: str,
               log_file: Optional[Union[str, Path]] = None,
               level: Union[str, int] = LOG_LEVEL,
               backup_count: int = 3,
               prune_days: int = 3) -> logging.Logger:
    """Create or return a configured logger.

    - name: logger name (e.g., "train_models")
    - log_file: path to log file; defaults to DEFAULT_LOG_FILE
    - level: logging level (str or int)
    """
    logfile = Path(log_file) if log_file else DEFAULT_LOG_FILE
    logfile.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(_level_to_int(level))

    # Avoid duplicate handlers on re-import
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Attach file handler if not already attached for this file
    needs_file = True
    for h in logger.handlers:
        if isinstance(h, TimedRotatingFileHandler) and getattr(h, "baseFilename", None) == str(logfile):
            needs_file = False
            break
    if needs_file:
        fh = TimedRotatingFileHandler(
            logfile, when="midnight", interval=1, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Attach a single console handler
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, TimedRotatingFileHandler)
               for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # Opportunistic prune
    _prune_old_logs(prune_days)

    return logger
