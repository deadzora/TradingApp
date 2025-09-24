# swarms/data.py
from __future__ import annotations
import pandas as pd
from typing import List, Optional

class DataAPI:
    """Replace internals with your Alpaca data / options / crypto providers."""

    def bars(self, symbol: str, interval: str = "15m", lookback: int = 200) -> Optional[pd.DataFrame]:
        # TODO: plug in your existing fetch_with_routing / alpaca_sdk
        return None

    def news_flags(self, symbol: str) -> dict:
        # TODO: stub -> {"has_fresh_news": False, "is_promo_like": False}
        return {"has_fresh_news": False, "is_promo_like": False}

    def options_snapshot(self, symbol: str) -> dict:
        # TODO: IV rank, vol/oi summaries; start with dummy numbers
        return {"ivr": 0.0, "vol_oi_ratio": 0.0, "avg_spread_bps": 0.0}

    def crypto_bars(self, symbol: str, interval: str = "15m", lookback: int = 200) -> Optional[pd.DataFrame]:
        # TODO
        return None
