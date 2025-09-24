# swarms/agents/daytrade.py
from __future__ import annotations
from typing import List
from .base import Agent
from ..schemas import Proposal, ExitPlan

class DaytradeAgent(Agent):
    name = "daytrade"

    def scan(self) -> List[Proposal]:
        out: List[Proposal] = []
        for sym in self.cfg["universe"]["equities_base"]:
            score = 62.0  # placeholder MOM strength
            out.append(Proposal(
                ts=self.now(), agent=self.name, symbol=sym, side="LONG",
                horizon_min=60, score=score, confidence=0.65,
                entry_px=None, exits=[ExitPlan("tp_limit", 0.0), ExitPlan("sl_stop", 0.0)],
                features={"mom_strength": 0.7, "rv_15m": 2.0}
            ))
        return out
