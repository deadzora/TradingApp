# swarms/data.py
from __future__ import annotations
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
from alpaca_trade_api import REST
from dotenv import load_dotenv
from pathlib import Path

# Load .env from repo root (one level up from /swarms)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

def _iso(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

class DataAPI:
    def __init__(self):
        self.api = REST(
            os.getenv("APCA_API_KEY_ID"),
            os.getenv("APCA_API_SECRET_KEY"),
            base_url=os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets",
            api_version="v2",
        )

    def bars(self, symbol: str, interval: str = "15m", lookback: int = 200) -> Optional[pd.DataFrame]:
        tf = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1d": "1Day"}[interval.lower()]
        start = datetime.now(timezone.utc) - timedelta(days=60 if "Min" in tf else 365)
        try:
            bars = self.api.get_bars(symbol, tf, start=_iso(start), limit=lookback, feed="iex").df
            if bars is None or len(bars) == 0:
                return None
            # Ensure expected columns
            cols = ["open","high","low","close","volume"]
            return bars[cols].copy()
        except Exception:
            return None

    def last_price(self, symbol: str) -> Optional[float]:
        try:
            lt = self.api.get_latest_trade(symbol)
            return float(lt.price)
        except Exception:
            # fallback to last close from 1m
            df = self.bars(symbol, "1m", 1)
            if df is not None and len(df):
                return float(df["close"].iloc[-1])
            return None

    # stubs (fill later)
    def news_flags(self, symbol: str) -> dict:
        return {"has_fresh_news": False, "is_promo_like": False}

    def options_snapshot(self, symbol: str) -> dict:
        return {"ivr": 0.0, "vol_oi_ratio": 0.0, "avg_spread_bps": 0.0}

    def crypto_bars(self, symbol: str, interval: str = "15m", lookback: int = 200) -> Optional[pd.DataFrame]:
        return None
