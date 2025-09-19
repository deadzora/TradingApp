import os
from alpaca_trade_api import REST

BASE = "https://paper-api.alpaca.markets"
api = REST(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"), base_url=BASE)

for sym in ["AAPL","SPY","QQQ"]:
    try:
        bars = api.get_bars(sym, "15Min", limit=200).df
        print(sym, "rows:", 0 if bars is None else len(bars))
        if bars is not None and len(bars) > 0:
            print(bars.tail(3)[["open","high","low","close","volume"]])
    except Exception as e:
        print(sym, "error:", e)
