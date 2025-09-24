# swarms/scoring.py
from __future__ import annotations

def clamp(v, lo=0.0, hi=100.0): return max(lo, min(hi, v))

def normalize_agent_score(raw: float, penalties: float = 0.0, boosts: float = 0.0) -> float:
    return clamp(raw - penalties + boosts, 0, 100)
