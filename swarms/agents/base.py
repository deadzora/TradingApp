# swarms/agents/base.py
from __future__ import annotations
from typing import List
from datetime import datetime, timezone
from ..schemas import Proposal, ExitPlan

class Agent:
    name = "base"
    def __init__(self, data_api, cfg): self.data = data_api; self.cfg = cfg
    def now(self): return datetime.now(timezone.utc)

    def scan(self) -> List[Proposal]:
        raise NotImplementedError
