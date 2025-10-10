"""Feature engineering utilities: returns, direction labels, lags, and technical indicators.

The module adds SMA/EMA, RSI, MACD, Bollinger Bands, rolling volatility and provides
`prepare_for_classification` which returns X,y ready for model training.
"""
import pandas as pd
import numpy as np


def add_returns(df, price_col='Close'):
    df = df.copy()
    pc = _resolve_price_col(df, price_col)
    df['return'] = df[pc].pct_change()
    return df


def add_direction_label(df, price_col='Close'):
    df = df.copy()
    pc = _resolve_price_col(df, price_col)
    df['direction'] = (df[pc].shift(-1) > df[pc]).astype(int)
    return df


def add_lags(df, cols=['Close'], n_lags=5):
    df = df.copy()
    # Resolve columns if they refer to price
    resolved = []
    for col in cols:
        if col in df.columns:
            resolved.append(col)
        else:
            # try resolving common price column names
            resolved.append(_resolve_price_col(df, col))
    for col in resolved:
        for lag in range(1, n_lags+1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def _resolve_price_col(df, preferred='Close'):
    """Return the column name to use for price-sensitive calculations.

    It tries the preferred name first, then several common alternatives (case-insensitive).
    Raises KeyError if no suitable price column is found.
    """
    cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in cols}
    candidates = [preferred, 'close', 'adj close', 'adjusted close', 'last', 'close_price', 'close*']
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    # also try exact match
    if preferred in df.columns:
        return preferred
    raise KeyError(f"No price column found among candidates: {candidates}. Available columns: {cols}")


def sma(series, window):
    return series.rolling(window).mean()


def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal


def bollinger_bands(series, window=20, n_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + (std * n_std)
    lower = ma - (std * n_std)
    return upper, lower


def add_technical_indicators(df, price_col='Close'):
    df = df.copy()
    df['sma_10'] = sma(df[price_col], 10)
    df['sma_20'] = sma(df[price_col], 20)
    df['ema_12'] = ema(df[price_col], 12)
    df['ema_26'] = ema(df[price_col], 26)
    macd_line, signal = macd(df[price_col])
    df['macd'] = macd_line
    df['macd_signal'] = signal
    df['rsi_14'] = rsi(df[price_col], 14)
    upper, lower = bollinger_bands(df[price_col], window=20, n_std=2)
    df['bb_upper'] = upper
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / (df[price_col].rolling(20).mean() + 1e-9)
    # rolling volatility
    df['vol_10'] = df[price_col].pct_change().rolling(10).std()
    df['vol_20'] = df[price_col].pct_change().rolling(20).std()
    return df


def prepare_for_classification(df, price_col='Close', n_lags=5, add_tech=True, dropna=True):
    df = df.copy()
    df = add_returns(df, price_col=price_col)
    if add_tech:
        df = add_technical_indicators(df, price_col=price_col)
    df = add_direction_label(df, price_col=price_col)
    df = add_lags(df, cols=[price_col], n_lags=n_lags)
    if dropna:
        df = df.dropna()
    feature_cols = [c for c in df.columns if c not in ['direction']]
    # Keep numeric features only
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['direction']
    return X, y

