# swarms/agents/crypto.py
from __future__ import annotations
from typing import List
from .base import Agent
from ..schemas import Proposal, ExitPlan

class CryptoAgent(Agent):
    name = "crypto"

    def scan(self) -> List[Proposal]:
        out: List[Proposal] = []
        for sym in self.cfg["universe"]["crypto_base"]:
            # TODO: funding/oi/fwd basis
            score = 58.0
            out.append(Proposal(
                ts=self.now(), agent=self.name, symbol=sym, side="LONG",
                horizon_min=240, score=score, confidence=0.55,
                entry_px=None, exits=[ExitPlan("tp_limit", 0.0), ExitPlan("sl_stop", 0.0)],
                features={"funding_z": 1.2, "oi_change": 0.3}
            ))
        return out
