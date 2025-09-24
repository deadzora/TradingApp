# swarms/orchestrator.py
from __future__ import annotations
from typing import List, Dict, DefaultDict
from collections import defaultdict
from datetime import datetime, timezone
from .schemas import Proposal, OrderIntent
from .risk import AccountState, banned, can_open_more, position_size
from .scoring import normalize_agent_score

def fuse_proposals(proposals: List[Proposal], cfg: dict) -> List[OrderIntent]:
    if not proposals: return []
    by_symbol: DefaultDict[str, List[Proposal]] = defaultdict(list)
    for p in proposals: by_symbol[p.symbol].append(p)

    intents: List[OrderIntent] = []
    for sym, props in by_symbol.items():
        # aggregate: if multi-agent same direction → boost; if conflict → penalty
        sides = {p.side for p in props}
        base = max(p.score for p in props)
        boost = cfg["orchestrator"]["boost_if_multi_agent_agree"] if len(props) > 1 and len(sides) == 1 else 0
        penalty = cfg["orchestrator"]["penalty_if_conflict"] if len(sides) > 1 else 0
        combined = normalize_agent_score(base, penalties=penalty, boosts=boost)

        # pick representative proposal (highest score)
        leader = max(props, key=lambda p: p.score)

        # skip low combined
        if combined < cfg["orchestrator"]["min_score_to_trade"]:
            continue

        # turn into order (qty computed later by caller)
        intents.append(OrderIntent(
            ts=datetime.now(timezone.utc),
            symbol=sym,
            side=leader.side,
            qty=0,  # fill later
            entry_limit=leader.entry_px or 0.0,  # caller may override with slop
            take_limit=0.0,
            stop_price=0.0,
            meta={"combined_score": f"{combined:.1f}", "agents": ",".join({p.agent for p in props})}
        ))
    # sort by score desc (stored in meta)
    intents.sort(key=lambda o: float(o.meta["combined_score"]), reverse=True)
    return intents
