import pandas as pd
import yfinance as yf
import time
import logging
import argparse
from joblib import Parallel, delayed
from ai_trading_bot.core.config import DATA_DIR, PERIOD, INTERVAL, LOGS_DIR, TICKERS
from ai_trading_bot.core.logger import get_logger

# --- Setup logger ---
LOG_FILE = LOGS_DIR / "data.log"
logger = get_logger("data", LOG_FILE)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Yahoo Finance dataframe into a consistent format."""
    df = df.copy()
    # Flatten MultiIndex columns (yfinance may return multi-level even for one ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df = df.rename(columns={"Adj Close": "AdjClose"})
    df = df.dropna()

    # Ensure timezone-naive index
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # Normalize to date only
    df.index = df.index.normalize()
    return df


def _fetch_and_save(ticker: str, dry_run: bool):
    """
    Wrapper for fetch_ticker to be used with joblib.
    """
    return fetch_ticker(ticker, dry_run=dry_run)


def fetch_ticker(ticker: str, force: bool = False, retries: int = 3, delay: int = 5, dry_run: bool = False) -> pd.DataFrame:
    """
    Download from Yahoo Finance and cache as CSV.
    - If CSV exists and force=False, load cached.
    - If CSV exists and force=True, refresh (full re-download).
    """
    fpath = DATA_DIR / f"{ticker.replace('.','_')}_{INTERVAL}.csv"

    # --- Cached load with incremental refresh ---
    if fpath.exists() and not force:
        try:
            # Specify a date format to avoid pandas "could not infer format" warning
            df = pd.read_csv(fpath, index_col=0, parse_dates=True, date_format="%Y-%m-%d")
            df.index = df.index.normalize()

            # If new data might be available, try incremental append
            try:
                last_cached = df.index[-1].date()
                today = pd.Timestamp.utcnow().normalize().date()
                if last_cached < today:
                    start = (pd.Timestamp(last_cached) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    inc = yf.download(
                        ticker,
                        start=start,
                        interval=INTERVAL,
                        auto_adjust=False,
                        progress=False,
                        threads=False,
                    )
                    if not inc.empty:
                        inc = _clean_df(inc)
                        before = len(df)
                        df = pd.concat([df, inc[~inc.index.isin(df.index)]], axis=0).sort_index()
                        added = len(df) - before
                        if added > 0:
                            if not dry_run:
                                df.to_csv(fpath, index_label="Date")
                                logger.info(f"{ticker}: appended {added} new rows (now {len(df)})")
                            else:
                                logger.info(f"{ticker}: [DRY RUN] would append {added} new rows")
                        else:
                            logger.info(f"{ticker}: cache already up-to-date (last={last_cached})")
                    else:
                        logger.info(f"{ticker}: no new rows from Yahoo after {last_cached}; using cache")
                else:
                    logger.info(f"{ticker}: cache fresh (last={last_cached})")
            except Exception as e:
                logger.warning(f"{ticker}: incremental refresh skipped → {e}")

            return df
        except Exception as e:
            logger.warning(f"{ticker}: failed to load cache, refetching… ({e})")

    # --- Retry loop for Yahoo API ---
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                period=PERIOD,
                interval=INTERVAL,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if not df.empty:
                df = _clean_df(df)
                if not dry_run:
                    df.to_csv(fpath, index_label="Date")
                    logger.info(f"{ticker}: downloaded {len(df)} rows (saved to {fpath.name})")
                else:
                    logger.info(f"{ticker}: [DRY RUN] downloaded {len(df)} rows")
                return df
            else:
                logger.warning(f"{ticker}: empty response (attempt {attempt}/{retries})")
        except Exception as e:
            logger.error(f"{ticker}: fetch attempt {attempt} failed: {e}")
        time.sleep(delay)

    # --- Fallback: some newer/quirky symbols fail for long periods; try shorter ---
    FALLBACK_PERIODS = ["3y", "2y", "1y", "max"]
    for p in FALLBACK_PERIODS:
        try:
            df = yf.download(
                ticker,
                period=p,
                interval=INTERVAL,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if not df.empty:
                df = _clean_df(df)
                if not dry_run:
                    df.to_csv(fpath, index_label="Date")
                    logger.info(f"{ticker}: fallback period={p} → downloaded {len(df)} rows (saved {fpath.name})")
                else:
                    logger.info(f"{ticker}: [DRY RUN] fallback period={p} → downloaded {len(df)} rows")
                return df
        except Exception:
            pass

    logger.error(f"{ticker}: all retries failed, returning None")
    return None


def update_all(tickers, force: bool = False):
    """Download/update data for all tickers in the list."""
    out = {}
    for t in tickers:
        df = fetch_ticker(t, force=force)
        if df is not None:
            out[t] = df
    return out


def main():
    """Main function to fetch and save ticker data."""
    parser = argparse.ArgumentParser(description="Fetch and save stock data.")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=TICKERS,
        help="Space-separated list of tickers to fetch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but do not save to disk.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching data for {len(args.tickers)} tickers...")

    # Use joblib for parallel fetching
    results = Parallel(n_jobs=-1)(
        delayed(_fetch_and_save)(ticker, args.dry_run) for ticker in args.tickers
    )

    # Log summary
    success_count = sum(1 for r in results if r is not None)
    fail_count = len(results) - success_count
    logger.info(
        f"Data fetch complete. Success: {success_count}, Failed: {fail_count}"
    )
    logger.info("==== Data ingestion completed ====")


if __name__ == "__main__":
    main()
