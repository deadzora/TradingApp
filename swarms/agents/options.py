# swarms/agents/options.py
from __future__ import annotations
from typing import List
from .base import Agent
from ..schemas import Proposal, ExitPlan

class OptionsAgent(Agent):
    name = "options"

    def scan(self) -> List[Proposal]:
        out: List[Proposal] = []
        for sym in self.cfg["universe"]["equities_base"]:
            ivr = 0.6  # TODO from data.options_snapshot
            flow = 0.7 # TODO unusual flow metric
            score = 100 * (0.5*ivr + 0.5*flow)  # simple blend
            out.append(Proposal(
                ts=self.now(), agent=self.name, symbol=sym, side="LONG",
                horizon_min=1440, score=score, confidence=0.6,
                entry_px=None, exits=[ExitPlan("tp_limit", 0.0), ExitPlan("sl_stop", 0.0)],
                features={"ivr": ivr, "flow": flow}
            ))
        return out
