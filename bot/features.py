
import numpy as np, pandas as pd
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))
def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = (high - low).abs()
    tr = pd.concat([tr, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()
def compute_indicators(df: pd.DataFrame):
    out = {}
    for win in [10,20,50,200]:
        out[f"ma_{win}"] = df["close"].rolling(win).mean()
    out["rsi_14"] = rsi(df["close"], 14)
    out["atr_14"] = atr(df["high"], df["low"], df["close"], 14)
    out["ret_5d"] = df["close"].pct_change(5)
    out["ret_20d"] = df["close"].pct_change(20)
    out["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
    return pd.DataFrame(out, index=df.index)
