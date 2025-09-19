
import os
from alpaca_trade_api import REST
from .base_broker import BaseBroker

class AlpacaBroker(BaseBroker):
    def __init__(self, cfg):
        base = cfg["alpaca"]["paper_base_url"] if cfg["mode"]=="paper" else cfg["alpaca"]["live_base_url"]
        self.api = REST(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"), base_url=base)
    def get_equity(self): return float(self.api.get_account().equity)
    def get_buying_power(self): return float(self.api.get_account().buying_power)
    def get_cash(self): return float(self.api.get_account().cash)
    def has_open_position(self, symbol: str) -> bool:
        try:
            pos = self.api.get_position(symbol); return abs(float(pos.qty))>0
        except Exception: return False
    def last_price(self, symbol: str):
        try: return float(self.api.get_latest_trade(symbol).price)
        except Exception:
            try: return float(self.api.get_latest_quote(symbol).ap)
            except Exception: return None
    def place_bracket_buy(self, symbol: str, qty: float, stop_price: float, take_price: float, tif: str = "gtc"):
        return self.api.submit_order(symbol=symbol, qty=str(qty), side="buy", type="market", time_in_force=tif,
                                     order_class="bracket",
                                     stop_loss={"stop_price": f"{stop_price:.2f}"},
                                     take_profit={"limit_price": f"{take_price:.2f}"})
