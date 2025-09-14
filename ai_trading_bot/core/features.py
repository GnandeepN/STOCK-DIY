# features.py (Phase 2 upgrade)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
try:
    import ta  # optional; we provide fallbacks below
    _HAS_TA = True
except Exception:
    ta = None
    _HAS_TA = False
import yfinance as yf
from ai_trading_bot.core.config import RSI_WINDOW, EMA_WINDOWS, ATR_WINDOW, RET_WINDOWS, TARGET_HORIZON, TARGET_MODE, R_MULT_HORIZON


# ✅ Preload sentiment model (lazy init to avoid reload per call)
_sentiment_model = None
def get_sentiment_model():
    """Lazy-load FinBERT via importlib to avoid static import warnings when transformers
    is not installed in the environment. Returns a callable pipeline or None.
    """
    global _sentiment_model
    if _sentiment_model is None:
        try:
            import importlib
            mod = importlib.import_module("transformers")
            pipeline = getattr(mod, "pipeline", None)
            if pipeline is None:
                _sentiment_model = None
            else:
                _sentiment_model = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
        except Exception:
            _sentiment_model = None
    return _sentiment_model


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # RSI
    if _HAS_TA:
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_WINDOW).rsi()
    else:
        # Wilder's RSI approximation
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / RSI_WINDOW, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / RSI_WINDOW, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
    # EMAs
    for w in EMA_WINDOWS:
        df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    # ATR
    if _HAS_TA:
        atr = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"], window=ATR_WINDOW
        ).average_true_range()
        df["atr"] = atr
    else:
        prev_close = df["Close"].shift(1)
        tr = np.maximum(df["High"] - df["Low"],
                        np.maximum((df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()))
        df["atr"] = tr.ewm(alpha=1 / ATR_WINDOW, adjust=False).mean()
    # Returns
    df["ret_1"] = df["Close"].pct_change(1)
    for w in RET_WINDOWS:
        df[f"ret_{w}"] = df["Close"].pct_change(w)

    # Labels (default): 1 if next day close > today close
    df["target"] = (df["Close"].shift(-TARGET_HORIZON) > df["Close"]).astype(int)
    # Optional scaffold for R-multiple regression/classification (not enabled by default)
    if TARGET_MODE != "direction":
        try:
            r = (df["Close"].shift(-R_MULT_HORIZON) - df["Close"]) / df["atr"].replace(0, np.nan)
            df["target_r"] = r
        except Exception:
            df["target_r"] = np.nan
    return df


def build_ml_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_indicators(df)
    # Strict time alignment: lag features by 1 bar so decisions at t use info up to t-1
    feat_cols = ["rsi", "atr"] + [f"ema_{w}" for w in EMA_WINDOWS] + [f"ret_{w}" for w in RET_WINDOWS]
    for c in feat_cols:
        if c in df.columns:
            df[c] = df[c].shift(1)
    df = df.dropna().copy()
    cols = ["Open", "High", "Low", "Close", "Volume"] + feat_cols + ["target"]
    if "target_r" in df.columns and TARGET_MODE != "direction":
        cols.append("target_r")
    return df[cols].copy()


def feature_columns():
    return ["rsi", "atr"] + [f"ema_{w}" for w in EMA_WINDOWS] + [f"ret_{w}" for w in RET_WINDOWS]


# ======================
# ✅ Phase 2 Additions
# ======================

def get_fundamentals(ticker: str):
    """Fetch basic fundamentals from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio": info.get("trailingPE", np.nan),
            "forward_pe": info.get("forwardPE", np.nan),
            "div_yield": info.get("dividendYield", np.nan),
            "market_cap": info.get("marketCap", np.nan),
        }
    except Exception:
        return {"pe_ratio": np.nan, "forward_pe": np.nan,
                "div_yield": np.nan, "market_cap": np.nan}


def get_news_sentiment(ticker: str):
    """Scrape headlines and run FinBERT sentiment."""
    try:
        # Optional deps loaded lazily
        import requests  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore

        url = f"https://news.google.com/search?q={ticker}+stock+india"
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        headlines = [a.text for a in soup.find_all("a")][:5]
        if not headlines:
            return 0.0

        model = get_sentiment_model()
        if model is None:
            return 0.0

        scores = []
        for h in headlines:
            res = model(h)[0]
            if res["label"] == "Positive":
                scores.append(1)
            elif res["label"] == "Negative":
                scores.append(-1)
            else:
                scores.append(0)
        return np.mean(scores)
    except Exception:
        return 0.0


def add_extra_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add advanced indicators, fundamentals, sentiment, and NIFTY context."""
    df = df.copy()

    # --- Extra Technical Indicators ---
    if _HAS_TA:
        df["macd"] = ta.trend.MACD(df["Close"]).macd()
        boll = ta.volatility.BollingerBands(df["Close"])
        df["bollinger_high"] = boll.bollinger_hband()
        df["bollinger_low"] = boll.bollinger_lband()
        stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
        df["stoch"] = stoch.stoch()
    else:
        # MACD (12,26,9) using EMAs
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        df["macd"] = macd
        # Bollinger Bands (20, 2 std)
        ma20 = df["Close"].rolling(window=20, min_periods=1).mean()
        std20 = df["Close"].rolling(window=20, min_periods=1).std()
        df["bollinger_high"] = ma20 + 2 * std20
        df["bollinger_low"] = ma20 - 2 * std20
        # Stochastic %K (14)
        low14 = df["Low"].rolling(window=14, min_periods=1).min()
        high14 = df["High"].rolling(window=14, min_periods=1).max()
        denom = (high14 - low14).replace(0, np.nan)
        df["stoch"] = ((df["Close"] - low14) / denom * 100).fillna(0)

    # --- Fundamentals ---
    funds = get_fundamentals(ticker)
    for k, v in funds.items():
        df[k] = v  # repeated across all rows

    # --- News Sentiment ---
    sentiment_score = get_news_sentiment(ticker)
    df["news_sentiment"] = sentiment_score

    # --- NIFTY Index Context ---
    try:
        nifty = yf.download("^NSEI", period="1y", interval="1d", auto_adjust=False, progress=False)
        if _HAS_TA:
            nifty["rsi"] = ta.momentum.RSIIndicator(nifty["Close"], window=14).rsi()
        else:
            delta = nifty["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            nifty["rsi"] = 100 - (100 / (1 + rs))
        df["nifty_rsi"] = nifty["rsi"].reindex(df.index)
        df["stock_vs_nifty"] = df["Close"] / nifty["Close"].reindex(df.index)
    except Exception:
        df["nifty_rsi"] = np.nan
        df["stock_vs_nifty"] = np.nan

    return df
