
import yfinance as yf, pandas as pd
from .base_data import BaseData
class YFinanceData(BaseData):
    def fetch_history(self, symbol: str, period: str = "6mo", interval: str = "1d"):
        d = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if d is None or d.empty: return None
        d = d.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
        return d[["open","high","low","close","volume"]].dropna()
