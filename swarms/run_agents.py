# swarms/run_agents.py
# Multi-agent trading scout using OpenAI Agents SDK + Alpaca paper broker
# Requires: pip install openai-agents pydantic python-dotenv requests

import os
import math
import asyncio
from typing import List, Literal
from datetime import datetime, timezone, timedelta
from pathlib import Path

# -------- load env from repo root (.env) --------
ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Add it to your repo-root .env or set it in the environment.")

# -------- OpenAI Agents SDK --------
# pip install openai-agents
from agents import Agent, Runner, function_tool

# -------- other deps --------
from pydantic import BaseModel, Field, conint

# your existing broker wrapper
from bot.providers.alpaca_broker import AlpacaBroker

# ---- config knobs ----
DRY_RUN = True                     # set to False to actually place orders via paper broker
CADENCE_SECONDS = 60               # run cadence if you loop this; this file runs once
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

MAX_PROPOSALS_PER_AGENT = 3
STOP_FRAC = 0.01
TAKE_FRAC = 0.02
MIN_QTY = 1

def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# ===== Structured output =====
class Proposal(BaseModel):
    agent: Literal["penny", "daytrade", "options", "crypto"] = Field(...)
    symbol: str = Field(..., description="Ticker symbol, upper-case")
    side: Literal["BUY"] = "BUY"
    entry: float = Field(..., gt=0)
    stop: float = Field(..., gt=0)
    take: float = Field(..., gt=0)
    confidence: conint(ge=1, le=100) = 60
    timeframe: Literal["intraday", "swing", "multi-day"] = "intraday"
    thesis: str = Field(..., description="Short rationale")

class Proposals(BaseModel):
    proposals: List[Proposal]

# ===== tools (agents can call these) =====
def _alpaca_headers():
    key = os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec, "Accept": "application/json"}

@function_tool
def get_last_price(symbol: str) -> float:
    """
    Return the last 15m close for symbol using Alpaca Data (IEX).
    """
    try:
        import requests
        start = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat().replace("+00:00","Z")
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        params = {"timeframe":"15Min","start":start,"limit":50,"feed":"iex","adjustment":"raw"}
        r = requests.get(url, params=params, headers=_alpaca_headers(), timeout=15)
        r.raise_for_status()
        bars = r.json().get("bars") or []
        if not bars:
            return -1.0
        return float(bars[-1]["c"])
    except Exception:
        return -1.0

@function_tool
def get_basic_volatility(symbol: str) -> float:
    """
    Return a naive ATR% proxy from recent 15m bars (avg (high-low)/close).
    """
    try:
        import requests
        start = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat().replace("+00:00","Z")
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        params = {"timeframe":"15Min","start":start,"limit":40,"feed":"iex","adjustment":"raw"}
        r = requests.get(url, params=params, headers=_alpaca_headers(), timeout=15)
        r.raise_for_status()
        bars = r.json().get("bars") or []
        if len(bars) < 10:
            return 0.0
        s = 0.0; n = 0
        for b in bars[-30:]:
            c = float(b["c"]); h = float(b["h"]); l = float(b["l"])
            if c > 0:
                s += (h - l) / c
                n += 1
        return (s / max(n,1)) if n else 0.0
    except Exception:
        return 0.0

# ---- order placement (only if DRY_RUN=False) ----
def place_bracket(broker: AlpacaBroker, symbol: str, entry: float, qty: int, stop: float, take: float):
    # Alpaca bracket constraints: tp >= entry+0.01, sl <= entry-0.01
    tp = max(take, entry + 0.01)
    sl = min(stop, entry - 0.01)
    broker.place_bracket_buy(symbol, qty, sl, tp, entry_price=entry, extended_hours=False)

# ===== Agent definitions =====
COMMON_TOOLING = [get_last_price, get_basic_volatility]

PENNY_INSTRUCTIONS = """
You are a penny stock scout focusing on liquid U.S. equities under $5 with avg dollar volume > $5M.
Use tools to sanity-check last price and volatility. Avoid illiquid tickers and OTC.
Return 1–3 long ideas with absolute entry/stop/take prices. Use ~1% stops and ~2% takes unless volatility suggests otherwise.
Return only structured output.
"""

DAYTRADE_INSTRUCTIONS = """
You are a day-trading scout. Favor highly liquid large-caps (SPY, QQQ, AAPL, MSFT, NVDA, etc.).
Look for momentum continuation or clean pullbacks; use tools to size targets.
Return 1–3 long ideas with tight ~1% stops and ~2% takes. Structured output only.
"""

OPTIONS_INSTRUCTIONS = """
You are an options-friendly scout proposing underlying STOCK entries (not option chains) suitable for options trading:
large-cap, tight spreads, liquid. Provide entry/stop/take with ~1%/~2% targets. Structured output only.
"""

CRYPTO_INSTRUCTIONS = """
You are a crypto proxy scout using equities/ETFs (e.g., COIN, BTC/ETH ETFs). Provide stock/ETF tickers only.
Use tools to sanity-check. 1% stops / 2% takes; intraday bias. Structured output only.
"""

def make_agent(name: str, instructions: str) -> Agent:
    return Agent(
        name=name,
        instructions=instructions,
        model=MODEL_NAME,
        output_type=Proposals,     # Pydantic-typed structured outputs
        tools=COMMON_TOOLING,
    )

AGENTS = [
    ("penny",    make_agent("penny",    PENNY_INSTRUCTIONS)),
    ("daytrade", make_agent("daytrade", DAYTRADE_INSTRUCTIONS)),
    ("options",  make_agent("options",  OPTIONS_INSTRUCTIONS)),
    ("crypto",   make_agent("crypto",   CRYPTO_INSTRUCTIONS)),
]

ONLY = set((os.getenv("SWARM_ONLY") or "").split(",")) - {""}
if ONLY:
    AGENTS = [(n, a) for (n, a) in AGENTS if n in ONLY]

# ===== simple coordinator =====
def combine_proposals(all_props: List[Proposal]) -> List[Proposal]:
    """
    Merge by symbol; average confidence; tighten stop / widen take.
    """
    by_sym: dict[str, Proposal] = {}
    for p in all_props:
        key = p.symbol.upper()
        if key not in by_sym:
            by_sym[key] = p
        else:
            cur = by_sym[key]
            cur.confidence = int((cur.confidence + p.confidence) / 2)
            cur.stop = min(cur.stop, p.stop)
            cur.take = max(cur.take, p.take)
            cur.thesis = (cur.thesis + f" | also liked by {p.agent}")[:400]
            by_sym[key] = cur
    out = list(by_sym.values())
    out.sort(key=lambda x: x.confidence, reverse=True)
    return out

async def run_once():
    print(f"[boot] swarms runner | ts={now_iso()} | cadence={CADENCE_SECONDS}s")

    # show account basics
    broker = AlpacaBroker({"mode": os.getenv("MODE","paper")})
    try:
        eq = float(broker.get_equity())
    except Exception:
        eq = 0.0
    print(f"[acct] equity={eq:.2f}")

    universe = ["SPY","QQQ","AAPL","MSFT","NVDA","AMD","META","TSLA","COIN","IWM","AVGO","GOOGL"]

    all_props: List[Proposal] = []
    for name, agent in AGENTS:
        prompt = (
            f"Universe: {', '.join(universe)}.\n"
            f"Return up to {MAX_PROPOSALS_PER_AGENT} proposals as structured output."
        )
        res = await Runner.run(agent, input=prompt)
        props = (getattr(res, "final_output", None).proposals if getattr(res, "final_output", None) else []) or []
        # Ensure agent field is set
        for p in props:
            if not p.agent:
                p.agent = name
        print(f"[{name}] {len(props)} proposal(s)")
        for p in props:
            print(f"  - {p.symbol} @{p.entry:.2f} SL {p.stop:.2f} TP {p.take:.2f} conf={p.confidence}")
        all_props.extend(props)

    merged = combine_proposals(all_props)
    if not merged:
        print("[coord] no proposals. Done.")
        return

    print(f"[coord] merged={len(merged)}; top 5:")
    for p in merged[:5]:
        print(f"  * {p.symbol} conf={p.confidence} entry={p.entry:.2f} stop={p.stop:.2f} take={p.take:.2f} [{p.agent}]")

    if DRY_RUN:
        print("[orders] DRY_RUN=True — not placing orders.")
        return

    # basic sizing: 2% of equity per trade (demo)
    risk_frac = 0.02
    per_trade = max(100.0, eq * risk_frac)
    placed = 0
    for p in merged[:5]:
        if p.entry <= 0:
            continue
        qty = max(MIN_QTY, int(per_trade / p.entry))
        try:
            place_bracket(broker, p.symbol, p.entry, qty, p.stop, p.take)
            placed += 1
            print(f"[order] {p.symbol} {qty} @~{p.entry:.2f} SL {p.stop:.2f} TP {p.take:.2f}")
        except Exception as e:
            print(f"[order×] {p.symbol} failed: {e}")

    print(f"[orders] placed={placed}")

def main_entry_once():
    import asyncio
    asyncio.run(run_once())

if __name__ == "__main__":
    main_entry_once()
