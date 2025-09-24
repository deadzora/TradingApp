# swarms/agents/penny.py
from __future__ import annotations
from typing import List
from .base import Agent
from ..schemas import Proposal, ExitPlan

class PennyAgent(Agent):
    name = "penny"

    def scan(self) -> List[Proposal]:
        out: List[Proposal] = []
        # TODO: pull your penny universe; start with cfg.universe.equities_base for demo
        for sym in self.cfg["universe"]["equities_base"]:
            # TODO: compute rel vol, gap, spreads, etc.
            score = 55.0  # placeholder
            exits = [ExitPlan("tp_limit", 0.0), ExitPlan("sl_stop", 0.0)]
            out.append(Proposal(
                ts=self.now(), agent=self.name, symbol=sym, side="LONG",
                horizon_min=240, score=score, confidence=0.6,
                entry_px=None, exits=exits,
                features={"rv_5m": 2.5, "spread_bps": 30}, notes=[]
            ))
        return out
