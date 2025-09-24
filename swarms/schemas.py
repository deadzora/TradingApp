# swarms/schemas.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
from datetime import datetime

Side = Literal["LONG","SHORT"]

@dataclass
class ExitPlan:
    kind: Literal["tp_limit", "sl_stop"]
    px: float

@dataclass
class Proposal:
    ts: datetime
    agent: str
    symbol: str
    side: Side
    horizon_min: int
    score: float                 # 0..100 (agent-specific -> normalized)
    confidence: float            # 0..1
    entry_px: Optional[float]    # agent’s suggested entry, can be None
    exits: List[ExitPlan]
    features: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

@dataclass
class OrderIntent:
    ts: datetime
    symbol: str
    side: Side
    qty: int
    entry_limit: float
    take_limit: float
    stop_price: float
    meta: Dict[str, str] = field(default_factory=dict)
