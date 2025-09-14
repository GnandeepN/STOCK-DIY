import joblib
from pathlib import Path
from ai_trading_bot.core.config import REPORTS_DIR

FEATURES_CACHE_DIR = REPORTS_DIR / "features_cache"
FEATURES_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cached_ml_frame(raw_df, ticker: str, build_fn):
    """Cache ML frame keyed by ticker and last data date to speed reruns.

    build_fn: function to build the ML frame from raw df (e.g., features.build_ml_frame)
    """
    try:
        if raw_df is None or raw_df.empty:
            return raw_df
        last_date = str(raw_df.index[-1].date())
        key = f"{ticker.replace('.', '_')}_{last_date}.joblib"
        fpath = FEATURES_CACHE_DIR / key
        if fpath.exists():
            return joblib.load(fpath)
        ml = build_fn(raw_df)
        joblib.dump(ml, fpath)
        return ml
    except Exception:
        return build_fn(raw_df)

