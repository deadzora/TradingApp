# swarms/risk.py
from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class AccountState:
    equity: float
    open_positions: Dict[str, float]  # symbol -> qty
    todays_pnl_fraction: float        # -0.03 means -3% on day

def banned(symbol: str, ban_list: List[str]) -> bool:
    return symbol.upper() in {b.upper() for b in ban_list}

def can_open_more(state: AccountState, cfg: dict, agent_open_count: int) -> bool:
    if len(state.open_positions) >= cfg["risk"]["max_positions_total"]:
        return False
    if agent_open_count >= cfg["risk"]["max_positions_per_agent"]:
        return False
    if state.todays_pnl_fraction <= -abs(cfg["risk"]["per_day_loss_limit_fraction"]):
        return False
    return True

def position_size(equity: float, px: float, per_trade_risk_fraction: float, stop_fraction: float, min_qty: int) -> int:
    risk_dollars = equity * per_trade_risk_fraction
    dollars_per_share = max(px * stop_fraction, 0.01)
    qty = int(risk_dollars / dollars_per_share)
    return max(qty, min_qty)
