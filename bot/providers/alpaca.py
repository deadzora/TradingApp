# bot/providers/alpaca.py
from __future__ import annotations

import os
from typing import Optional

try:
    from alpaca_trade_api import REST
except Exception as _imp_err:
    REST = None
    _ALPACA_IMPORT_ERR = _imp_err


class AlpacaBroker:
    def __init__(self, *args, base_url: Optional[str] = None, **kwargs):
        """
        Flexible signature so registry calls like AlpacaBroker(cfg=...) don't crash.
        Respects APCA_API_BASE_URL if base_url isn't passed.
        """
        if REST is None:
            raise RuntimeError(
                f"alpaca-trade-api import failed. Ensure it's installed in THIS venv. "
                f"Underlying error: {_ALPACA_IMPORT_ERR!r}"
            )

        key = os.getenv("APCA_API_KEY_ID")
        sec = os.getenv("APCA_API_SECRET_KEY")
        base = base_url or os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"

        if not key or not sec:
            raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment.")

        self.api = REST(key, sec, base_url=base, api_version="v2")

    # --- account / equity ---
    def get_equity(self) -> float:
        try:
            acct = self.api.get_account()
            return float(getattr(acct, "equity", 0) or 0)
        except Exception:
            return 0.0

    # --- positions ---
    def has_open_position(self, symbol: str) -> bool:
        try:
            pos = self.api.get_position(symbol)
            return bool(pos and float(getattr(pos, "qty", 0) or 0) != 0)
        except Exception:
            # raises on not found
            return False

    # --- orders ---
    def place_bracket_buy(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
        take_price: float,
        entry_price: Optional[float] = None,
        extended_hours: bool = False,
        tif: str = "gtc",
    ):
        """
        Submit a LIMIT parent bracket so Alpaca has a definite base_price.
        Enforce:
          take_profit.limit_price >= entry + $0.01
          stop_loss.stop_price  <= entry - $0.01
        """
        if qty <= 0:
            raise ValueError("qty must be > 0")

        def _tick(x: float) -> float:
            # round to cents with a tiny nudge to avoid equality edge cases
            return round(float(x) + 1e-9, 2)

        # Determine entry
        ent = float(entry_price or 0.0)
        if ent <= 0:
            try:
                lt = self.api.get_latest_trade(symbol)
                ent = float(getattr(lt, "price", 0) or 0)
            except Exception:
                ent = 0.0
        if ent <= 0:
            ent = (float(take_price) + float(stop_price)) / 2.0  # last-resort base reference

        ent = _tick(ent)
        tp  = _tick(float(take_price))
        sl  = _tick(float(stop_price))

        # Enforce 1¢ away from entry per Alpaca validation
        if tp < ent + 0.01:
            tp = _tick(ent + 0.02)
        if sl > ent - 0.01:
            sl = _tick(ent - 0.02)

        try:
            self.api.submit_order(
                symbol=symbol,
                qty=str(int(qty)),
                side="buy",
                type="limit",                 # LIMIT parent (not market)
                limit_price=str(ent),
                time_in_force=tif,
                order_class="bracket",
                take_profit={"limit_price": str(tp)},
                stop_loss={"stop_price": str(sl)},
                extended_hours=bool(extended_hours),
            )
        except Exception as e:
            # bubble up with useful context
            raise RuntimeError(
                f"submit_order failed: sym={symbol} qty={qty} entry={ent} "
                f"tp={tp} sl={sl} tif={tif} ext={extended_hours} err={e}"
            ) from e
