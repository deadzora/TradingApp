
from alpaca_trade_api import REST
import os, pandas as pd
from .base_data import BaseData
class AlpacaData(BaseData):
    def __init__(self, cfg):
        base = cfg["alpaca"]["paper_base_url"] if cfg["mode"]=="paper" else cfg["alpaca"]["live_base_url"]
        self.api = REST(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"), base_url=base)
    def fetch_history(self, symbol: str, period: str = "6mo", interval: str = "1d"):
        tf = "15Min" if interval in ("15m","15Min") else "1Day"
        limit = 800 if tf=="15Min" else 180
        bars = self.api.get_bars(symbol, tf, limit=limit).df
        if bars is None or len(bars)==0: return None
        d = bars.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})
        return d[["open","high","low","close","volume"]].dropna()
