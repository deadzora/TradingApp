
def mean_reversion_signal(df):
    r = df["close"].pct_change()
    z = (r - r.rolling(20).mean()) / (r.rolling(20).std() + 1e-9)
    return z.iloc[-1] < -2
def momentum_signal(df):
    return df["close"].iloc[-1] >= df["close"].rolling(20).max().iloc[-2]
