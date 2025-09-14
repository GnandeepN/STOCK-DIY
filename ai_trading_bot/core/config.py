from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
for p in (DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --- Universe (choose from external ticker files) ---
from ai_trading_bot.core.tickers_nifty50 import TICKERS as NIFTY50_TICKERS
from ai_trading_bot.core.tickers_banknifty import TICKERS as BANKNIFTY_TICKERS
from ai_trading_bot.core.tickers_custom import TICKERS as CUSTOM_TICKERS

# Merge all tickers together (remove duplicates) and optionally include holdings watchlist
TICKERS = list(set(NIFTY50_TICKERS + BANKNIFTY_TICKERS + CUSTOM_TICKERS))

# Optionally include holdings tickers if available
HOLDINGS_CSV = REPORTS_DIR / "holdings_tickers.csv"
if HOLDINGS_CSV.exists():
    try:
        import pandas as _pd
        _h = _pd.read_csv(HOLDINGS_CSV)
        if "ticker" in _h.columns:
            _hs = [str(t).strip() for t in _h["ticker"].dropna().tolist()]
            # Add only NSE/BSE style tickers (e.g., .NS/.BO) to avoid mismatches
            _hs = [t for t in _hs if t.endswith(".NS") or t.endswith(".BO")]
            TICKERS = list(set(TICKERS + _hs))
    except Exception:
        pass

# --- Data & ML ---
PERIOD = "5y"
INTERVAL = "1d"
TARGET_HORIZON = 1
RANDOM_SEED = 42
TARGET_MODE = "direction"  # "direction" or "r_multiple" (scaffold)
R_MULT_HORIZON = 1         # bars ahead for R-multiple target if enabled

# --- Features ---
RSI_WINDOW = 14
EMA_WINDOWS = [20, 50, 200]
ATR_WINDOW = 14
RET_WINDOWS = [1, 5, 10]

# --- Model selection ---
MODEL_TYPE = "lgbm"   # "rf", "lgbm", or "xgb"

# --- RandomForest params ---
RF_PARAMS = dict(
    n_estimators=50,
    max_depth=6,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=RANDOM_SEED,
)

# --- LightGBM params ---
LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=-1,  # quiet LightGBM logs
)

# --- XGBoost params ---
XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric="logloss",
    verbosity=0,  # quiet XGBoost logs
)

# Centralized model params lookup
MODEL_PARAMS = {
    "rf": RF_PARAMS,
    "lgbm": LGBM_PARAMS,
    "xgb": XGB_PARAMS,
}

# --- Signal policy ---
PROB_THRESHOLD = 0.55
TREND_FILTER = True
MIN_CONFIDENCE = 0.55  # additional filter: BUY proba_up>=this, SELL<=1-this
MIN_AVG_VOLUME = 0     # set >0 to require 20D avg volume
# Optional sector exposure caps: {"Financial Services": 5, "Technology": 4}
SECTOR_CAPS = {}
INDUSTRY_CAPS = {}     # e.g., {"Software": 3}
REGIME_NIFTY_RSI_MIN = 45  # skip longs if Nifty RSI below this
REGIME_NIFTY_RSI_STRONG = 55  # strong regime threshold for CNC eligibility
RR_MIN = 1.5                # minimum risk/reward at entry
SWING_LOOKBACK = 20         # bars for swing high
VOLUME_SPIKE_MULT = 1.3     # today vol >= this * avg20
PULLBACK_ATR_MULT = 0.5     # allow BUY if |price-ema20| <= k*ATR with trend up
TP_ATR_MULT = 1.5
SL_ATR_MULT = 1.0

# --- Backtest assumptions ---
ROUND_TRIP_BPS = 12
SLIPPAGE_BPS = 4

# --- Backtest runtime knobs ---
BACKTEST_RETRAIN_FREQ = 20
EXPORT_EQUITY = False

# --- Logging ---
LOG_LEVEL = "INFO"

# --- Backtest speed toggle ---
# Enable to make walk-forward backtests much faster (reduced retrains, fewer CV folds,
# single-threaded models during backtest fits, and smaller ensembles). Daily training/signals
# continue to use full settings.
BACKTEST_FAST = True
BACKTEST_FAST_RETRAIN_FREQ = 60
BACKTEST_FAST_CV_FOLDS = 2
BACKTEST_FAST_ESTIMATORS = {
    "rf": 150,
    "lgbm": 150,
    "xgb": 200,
}

# --- Exits / monitor ---
TRAIL_ATR_MULT = 1.0        # trailing stop distance in ATRs
STAGNATION_R_MULT = 0.2     # consider stagnation if |price-entry| < k*R
STAGNATION_MAX_BARS = 20    # exit after this many stagnant checks

# --- Over-trading guards ---
MAX_DAILY_ENTRIES = 5
COOLDOWN_MINUTES = 30

# --- Product decision (CNC vs MIS) ---
RR_INTRADAY_MAX = 1.5      # RR below this → MIS forced
RR_CNC_MIN = 2.0           # RR above/equal → CNC eligible
CONFIDENCE_CNC = 0.70      # proba_up ≥ → favor CNC
LATEST_ENTRY_MIS_AFTER = "14:30"  # entries after this (IST) → MIS

# --- Execution realism / gap filter ---
APPLY_GAP_FILTER = False
GAP_ATR_MAX = 0.06  # if ATR/Close exceeds this, flag as high gap/vol risk

# --- Index volatility sizing ---
INDEX_SYMBOL = "^NSEI"
INDEX_TARGET_VOL_PCT = 20.0

# --- SELL semantics / portfolio policy ---
SELL_BEHAVIOR = "exit_then_short"   # "exit_only" | "short_only" | "exit_then_short" | "exit_or_ignore"
ALLOW_SHORTS = True
ALLOW_SHORT_WHEN_LONG = False

# --- Holdings guardian (optional future use) ---
HOLDINGS_TRAIL_ATR_MULT = 2.0
HOLDINGS_TP1_R = 1.0
HOLDINGS_TIME_MAX_BARS = 10

# --- Planner & caps (placeholders; planner enforces) ---
MAX_SINGLE_NAME_PCT = 10
MAX_SECTOR_PCT = 30
MAX_INDUSTRY_PCT = 20
MAX_PORTFOLIO_NOTIONAL = 1_000_000
SLIPPAGE_GUARD_PCT = 0.003
RESPECT_PRICE_BANDS = True
CHECK_SHORTABLE = True
